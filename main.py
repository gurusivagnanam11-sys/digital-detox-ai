"""
main.py — FastAPI application entry point.

Routes:
  POST /api/auth/register       — Create account
  POST /api/auth/login          — Get JWT token
  POST /api/usage/log           — Log usage session
  GET  /api/usage/today         — Get today's usage summary
  POST /api/chat                — Send message to AI coach (SSE stream)
  GET  /api/analytics/score     — Get today's wellness score
  GET  /api/analytics/report    — Get weekly report
  GET  /api/analytics/history   — Get usage history
  POST /api/focus/start         — Start Pomodoro session
  POST /api/focus/complete      — Mark session complete
  GET  /api/streak              — Get detox streak
"""

from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()
import json
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from database import (
    init_db,
    get_session_factory,
    User,
    UsageSession,
    WellnessScore,
    DetoxStreak,
    FocusSession,
    ChatMessage,
)
from config import get_settings
from llm_engine import DetoxLLMEngine, ConversationManager, build_user_context
from preprocessor import UsageDataPreprocessor
from feature_engineer import BehavioralFeatureEngineer
from nlp_pipeline import NLPPipeline
from habit_engine import HabitAnalysisEngine
from llm_engine import DetoxLLMEngine
from rag_system import WellnessRAGSystem

# ─── App setup ────────────────────────────────────────────────────────────────

settings = get_settings()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-powered digital wellness coaching platform",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Initialise services ──────────────────────────────────────────────────────

engine = init_db()
SessionLocal = get_session_factory(engine)

preprocessor   = UsageDataPreprocessor()
feature_eng    = BehavioralFeatureEngineer()
habit_engine   = HabitAnalysisEngine()
nlp_pipeline   = NLPPipeline()
llm_engine     = DetoxLLMEngine()
rag_system     = WellnessRAGSystem()

# Per-user conversation managers (in production: use Redis for multi-instance)
_conversations: dict[int, ConversationManager] = {}


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_conversation(user_id: int) -> ConversationManager:
    if user_id not in _conversations:
        _conversations[user_id] = ConversationManager()
    return _conversations[user_id]


# ─── Request / Response schemas ───────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str
    email: EmailStr
    password: str
    daily_limit_minutes: int = 180

class LoginRequest(BaseModel):
    username: str
    password: str

class LogUsageRequest(BaseModel):
    app_name: str
    app_category: str
    session_start: datetime
    session_end: datetime
    notification_count: int = 0

class ChatRequest(BaseModel):
    message: str
    stream: bool = True

class FocusStartRequest(BaseModel):
    duration_minutes: int = 25
    session_type: str = "focus"

class ChatResponse(BaseModel):
    reply: str
    intent: str
    wellness_score: Optional[float] = None


# ─── Auth routes ──────────────────────────────────────────────────────────────

@app.post("/api/auth/register", status_code=201)
async def register(req: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == req.username).first():
        raise HTTPException(status_code=400, detail="Username already exists")

    import hashlib
    hashed = hashlib.sha256(req.password.encode()).hexdigest()  # Use bcrypt in production

    user = User(
        username=req.username,
        email=req.email,
        hashed_password=hashed,
        daily_limit_minutes=req.daily_limit_minutes,
    )
    streak = DetoxStreak(current_streak_days=0, longest_streak_days=0, total_detox_days=0)
    user.streaks = streak

    db.add(user)
    db.commit()
    db.refresh(user)
    return {"user_id": user.id, "message": "Account created. Welcome to your digital wellness journey!"}


@app.post("/api/auth/login")
async def login(req: LoginRequest, db: Session = Depends(get_db)):
    import hashlib
    user = db.query(User).filter(User.username == req.username).first()
    hashed = hashlib.sha256(req.password.encode()).hexdigest()

    if not user or user.hashed_password != hashed:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # In production: return a signed JWT. Simplified here.
    return {"token": f"mock_token_{user.id}", "user_id": user.id, "username": user.username}


# ─── Usage tracking routes ────────────────────────────────────────────────────

@app.post("/api/usage/log")
async def log_usage(
    user_id: int,
    req: LogUsageRequest,
    db: Session = Depends(get_db),
):
    session = UsageSession(
        user_id=user_id,
        app_name=req.app_name,
        app_category=req.app_category,
        session_start=req.session_start,
        session_end=req.session_end,
        duration_minutes=(req.session_end - req.session_start).total_seconds() / 60,
        notification_count=req.notification_count,
        is_late_night=_is_late_night(req.session_start),
        is_social_media=req.app_name in nlp_pipeline.PRESERVE_TERMS,
    )
    db.add(session)
    db.commit()

    # Check if late-night warning should fire
    warning = None
    if session.is_late_night:
        warning = await llm_engine.generate_detox_reminder(
            usage_minutes=session.duration_minutes,
            limit_minutes=30,
        )

    return {"logged": True, "session_id": session.id, "warning": warning}


@app.get("/api/usage/today")
async def get_today_usage(user_id: int, db: Session = Depends(get_db)):
    today = datetime.now(timezone.utc).date()
    sessions = (
        db.query(UsageSession)
        .filter(
            UsageSession.user_id == user_id,
            UsageSession.session_start >= datetime.combine(today, datetime.min.time()),
        )
        .all()
    )
    total_minutes = sum(s.duration_minutes for s in sessions)
    social_minutes = sum(s.duration_minutes for s in sessions if s.is_social_media)

    user = db.query(User).filter(User.id == user_id).first()
    limit = user.daily_limit_minutes if user else 180

    return {
        "total_minutes": round(total_minutes, 1),
        "total_hours": round(total_minutes / 60, 2),
        "session_count": len(sessions),
        "social_media_ratio": round(social_minutes / max(total_minutes, 1), 3),
        "over_limit": total_minutes > limit,
        "limit_minutes": limit,
        "usage_pct_of_limit": round((total_minutes / limit) * 100, 1),
    }


# ─── Chat route ───────────────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat(
    user_id: int,
    req: ChatRequest,
    db: Session = Depends(get_db),
):
    # NLP processing
    processed = nlp_pipeline.process(req.message)

    # Build behavioral context
    user = db.query(User).filter(User.id == user_id).first()
    streak = db.query(DetoxStreak).filter(DetoxStreak.user_id == user_id).first()
    wellness_score = (
        db.query(WellnessScore)
        .filter(WellnessScore.user_id == user_id)
        .order_by(WellnessScore.id.desc())
        .first()
    )

    context = build_user_context(
        wellness_score={
            "score": wellness_score.score if wellness_score else None,
            "grade": wellness_score.grade if wellness_score else "N/A",
            "category": wellness_score.category if wellness_score else "N/A",
        } if wellness_score else None,
        streak={
            "current_streak_days": streak.current_streak_days if streak else 0,
            "longest_streak_days": streak.longest_streak_days if streak else 0,
        } if streak else None,
    )

    # RAG retrieval
    rag_context = rag_system.retrieve_for_intent(req.message, processed.detected_intent.value)

    # Conversation state
    conversation = get_conversation(user_id)

    # Save user message
    db.add(ChatMessage(
        user_id=user_id,
        session_id=f"session_{user_id}",
        role="user",
        content=req.message,
        intent=processed.detected_intent.value,
    ))

    if req.stream:
        # Server-Sent Events streaming response
        async def event_stream():
            full_reply = ""
            async for chunk in llm_engine.chat_stream(
                user_message=req.message,
                intent=processed.detected_intent,
                user_context=context,
                conversation=conversation,
                rag_context=rag_context,
            ):
                full_reply += chunk
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"

            # Save assistant message
            db_session = SessionLocal()
            db_session.add(ChatMessage(
                user_id=user_id,
                session_id=f"session_{user_id}",
                role="assistant",
                content=full_reply,
                intent=processed.detected_intent.value,
            ))
            db_session.commit()
            db_session.close()
            yield f"data: {json.dumps({'done': True})}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")
    else:
        reply = await llm_engine.chat(
            user_message=req.message,
            intent=processed.detected_intent,
            user_context=context,
            conversation=conversation,
            rag_context=rag_context,
        )
        db.add(ChatMessage(
            user_id=user_id, session_id=f"session_{user_id}",
            role="assistant", content=reply, intent=processed.detected_intent.value,
        ))
        db.commit()
        return ChatResponse(reply=reply, intent=processed.detected_intent.value)


# ─── Analytics routes ─────────────────────────────────────────────────────────

@app.get("/api/analytics/score")
async def get_wellness_score(user_id: int, db: Session = Depends(get_db)):
    score = (
        db.query(WellnessScore)
        .filter(WellnessScore.user_id == user_id)
        .order_by(WellnessScore.id.desc())
        .first()
    )
    if not score:
        raise HTTPException(status_code=404, detail="No wellness score yet. Log some usage first.")
    return {
        "score": score.score,
        "grade": score.grade,
        "category": score.category,
        "date": score.date,
        "breakdown": json.loads(score.breakdown) if score.breakdown else {},
        "recommendation": score.recommendation,
    }


@app.get("/api/analytics/report")
async def get_weekly_report(user_id: int, db: Session = Depends(get_db)):
    """Generate and return weekly wellness report."""
    # In production: run this as a nightly background task and cache it
    scores = (
        db.query(WellnessScore)
        .filter(WellnessScore.user_id == user_id)
        .order_by(WellnessScore.id.desc())
        .limit(7)
        .all()
    )

    if not scores:
        return {"message": "Not enough data for a weekly report yet. Use the app for a few days first."}

    weekly_data = {
        "scores": [{"date": s.date, "score": s.score, "grade": s.grade} for s in reversed(scores)],
        "avg_score": round(sum(s.score for s in scores) / len(scores), 1),
        "trend": "improving" if scores[0].score > scores[-1].score else "worsening",
    }

    report_text = await llm_engine.generate_weekly_report(weekly_data, habit_profiles=[])
    return {"report": report_text, "data": weekly_data}


@app.get("/api/streak")
async def get_streak(user_id: int, db: Session = Depends(get_db)):
    streak = db.query(DetoxStreak).filter(DetoxStreak.user_id == user_id).first()
    if not streak:
        return {"current_streak": 0, "longest_streak": 0, "total_detox_days": 0}
    return {
        "current_streak": streak.current_streak_days,
        "longest_streak": streak.longest_streak_days,
        "total_detox_days": streak.total_detox_days,
        "last_success_date": streak.last_success_date,
    }


# ─── Focus mode routes ────────────────────────────────────────────────────────

@app.post("/api/focus/start")
async def start_focus(user_id: int, req: FocusStartRequest, db: Session = Depends(get_db)):
    session = FocusSession(
        user_id=user_id,
        started_at=datetime.now(timezone.utc),
        duration_minutes=req.duration_minutes,
        session_type=req.session_type,
        completed=False,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return {
        "session_id": session.id,
        "duration_minutes": session.duration_minutes,
        "started_at": session.started_at.isoformat(),
        "message": f"Focus session started! {session.duration_minutes} minutes of deep work. You've got this 💪",
    }


@app.post("/api/focus/complete/{session_id}")
async def complete_focus(user_id: int, session_id: int, db: Session = Depends(get_db)):
    session = db.query(FocusSession).filter(
        FocusSession.id == session_id,
        FocusSession.user_id == user_id,
    ).first()

    if not session:
        raise HTTPException(status_code=404, detail="Focus session not found")

    session.completed = True
    session.completed_at = datetime.now(timezone.utc)
    db.commit()

    return {
        "completed": True,
        "duration_minutes": session.duration_minutes,
        "message": "Excellent focus session! 🎉 Take a well-deserved break.",
    }


# ─── Health check ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "healthy", "app": settings.APP_NAME, "version": settings.APP_VERSION}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _is_late_night(dt: datetime) -> bool:
    hour = dt.hour
    return hour >= settings.LATE_NIGHT_START_HOUR or hour < settings.LATE_NIGHT_END_HOUR
