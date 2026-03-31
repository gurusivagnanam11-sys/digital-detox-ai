"""
database.py — SQLAlchemy ORM models for Digital Detox AI.

Tables:
  users                — account + preferences
  usage_sessions       — raw screen-time events
  daily_feature_vectors— engineered feature row per user-day
  wellness_scores      — daily digital wellness score history
  detox_streaks        — current and historical streaks
  focus_sessions       — Pomodoro session records
  chat_history         — conversation turns
  detox_goals          — user-defined reduction targets
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey,
    Integer, String, Text, Index, create_engine
)
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker
from sqlalchemy.sql import func

from config import get_settings


settings = get_settings()


class Base(DeclarativeBase):
    pass


# ─── Users ────────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id              = Column(Integer, primary_key=True, index=True)
    username        = Column(String(64), unique=True, nullable=False, index=True)
    email           = Column(String(255), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    created_at      = Column(DateTime(timezone=True), server_default=func.now())
    is_active       = Column(Boolean, default=True)

    # Preferences
    daily_limit_minutes = Column(Integer, default=180)  # 3-hour default target
    timezone            = Column(String(64), default="UTC")
    late_night_warning  = Column(Boolean, default=True)

    # Relationships
    usage_sessions  = relationship("UsageSession",    back_populates="user", lazy="dynamic")
    wellness_scores = relationship("WellnessScore",   back_populates="user", lazy="dynamic")
    streaks         = relationship("DetoxStreak",     back_populates="user", uselist=False)
    focus_sessions  = relationship("FocusSession",    back_populates="user", lazy="dynamic")
    chat_history    = relationship("ChatMessage",     back_populates="user", lazy="dynamic")
    goals           = relationship("DetoxGoal",       back_populates="user", lazy="dynamic")


# ─── Usage Sessions ───────────────────────────────────────────────────────────

class UsageSession(Base):
    __tablename__ = "usage_sessions"

    id                 = Column(Integer, primary_key=True, index=True)
    user_id            = Column(Integer, ForeignKey("users.id"), nullable=False)
    app_name           = Column(String(128), nullable=False)
    app_category       = Column(String(64), nullable=False)
    session_start      = Column(DateTime(timezone=True), nullable=False)
    session_end        = Column(DateTime(timezone=True), nullable=False)
    duration_minutes   = Column(Float, nullable=False)
    notification_count = Column(Integer, default=0)
    is_late_night      = Column(Boolean, default=False)
    is_social_media    = Column(Boolean, default=False)

    user = relationship("User", back_populates="usage_sessions")

    __table_args__ = (
        Index("ix_usage_user_date", "user_id", "session_start"),
    )


# ─── Daily Feature Vectors ────────────────────────────────────────────────────

class DailyFeatureVector(Base):
    __tablename__ = "daily_feature_vectors"

    id                         = Column(Integer, primary_key=True)
    user_id                    = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    date                       = Column(String(10), nullable=False)   # YYYY-MM-DD

    # Core features
    total_daily_minutes        = Column(Float)
    session_count              = Column(Integer)
    avg_session_minutes        = Column(Float)
    social_media_ratio         = Column(Float)
    late_night_ratio           = Column(Float)
    notification_rate_per_hour = Column(Float)
    app_switch_frequency       = Column(Float)
    distinct_apps_used         = Column(Integer)
    peak_usage_hour            = Column(Float)
    evening_usage_ratio        = Column(Float)

    # Trend features
    usage_trend_slope          = Column(Float, default=0.0)
    social_trend_slope         = Column(Float, default=0.0)
    late_night_trend_slope     = Column(Float, default=0.0)

    # Flags
    excessive_usage_flag       = Column(Boolean, default=False)
    social_dependency_flag     = Column(Boolean, default=False)
    night_addiction_flag       = Column(Boolean, default=False)
    active_warning_count       = Column(Integer, default=0)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_feature_user_date", "user_id", "date", unique=True),
    )


# ─── Wellness Scores ─────────────────────────────────────────────────────────

class WellnessScore(Base):
    __tablename__ = "wellness_scores"

    id              = Column(Integer, primary_key=True)
    user_id         = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    date            = Column(String(10), nullable=False)
    score           = Column(Float, nullable=False)   # 0-100
    grade           = Column(String(2))               # A+ through F
    category        = Column(String(32))              # "excellent" / "good" / "fair" / "poor"
    breakdown       = Column(Text)                    # JSON: component scores
    recommendation  = Column(Text)                    # Generated recommendation text

    user = relationship("User", back_populates="wellness_scores")

    __table_args__ = (
        Index("ix_wellness_user_date", "user_id", "date", unique=True),
    )


# ─── Detox Streaks ────────────────────────────────────────────────────────────

class DetoxStreak(Base):
    __tablename__ = "detox_streaks"

    id                   = Column(Integer, primary_key=True)
    user_id              = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    current_streak_days  = Column(Integer, default=0)
    longest_streak_days  = Column(Integer, default=0)
    last_success_date    = Column(String(10))   # YYYY-MM-DD
    total_detox_days     = Column(Integer, default=0)
    streak_started_at    = Column(DateTime(timezone=True))
    updated_at           = Column(DateTime(timezone=True), onupdate=func.now())

    user = relationship("User", back_populates="streaks")


# ─── Focus Sessions ───────────────────────────────────────────────────────────

class FocusSession(Base):
    __tablename__ = "focus_sessions"

    id              = Column(Integer, primary_key=True)
    user_id         = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    started_at      = Column(DateTime(timezone=True), nullable=False)
    completed_at    = Column(DateTime(timezone=True))
    duration_minutes= Column(Integer, nullable=False)
    session_type    = Column(String(16), default="focus")  # focus / short_break / long_break
    completed       = Column(Boolean, default=False)
    interrupted_at  = Column(Float)  # minutes into session when interrupted (if any)

    user = relationship("User", back_populates="focus_sessions")


# ─── Chat History ─────────────────────────────────────────────────────────────

class ChatMessage(Base):
    __tablename__ = "chat_history"

    id              = Column(Integer, primary_key=True)
    user_id         = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    session_id      = Column(String(64), nullable=False, index=True)
    role            = Column(String(16), nullable=False)  # "user" | "assistant"
    content         = Column(Text, nullable=False)
    intent          = Column(String(64))
    created_at      = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="chat_history")


# ─── Detox Goals ──────────────────────────────────────────────────────────────

class DetoxGoal(Base):
    __tablename__ = "detox_goals"

    id                    = Column(Integer, primary_key=True)
    user_id               = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    goal_type             = Column(String(64))  # "daily_limit" / "social_media_limit" / "no_late_night"
    target_value          = Column(Float)
    current_value         = Column(Float, default=0.0)
    start_date            = Column(String(10))
    target_date           = Column(String(10))
    achieved              = Column(Boolean, default=False)

    user = relationship("User", back_populates="goals")


# ─── DB Initialisation ────────────────────────────────────────────────────────

def create_db_engine():
    return create_engine(
        settings.DATABASE_URL,
        connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {},
        echo=settings.DEBUG,
    )

def get_session_factory(engine=None):
    if engine is None:
        engine = create_db_engine()
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    engine = create_db_engine()
    Base.metadata.create_all(bind=engine)
    return engine
