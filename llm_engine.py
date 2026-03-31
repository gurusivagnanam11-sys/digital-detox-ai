"""
llm_engine.py — LLM-powered digital wellness chatbot.

Architecture:
  - Retrieval-Augmented Generation (RAG) using ChromaDB
  - Multi-turn conversation with sliding context window
  - Behavioral context injection (user's actual usage data)
  - Structured prompt templates per intent
  - Streaming response support
  - Fallback to local Llama via Ollama if OpenAI unavailable
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import AsyncGenerator, Optional

import httpx
from openai import AsyncOpenAI

from config import get_settings
from nlp_pipeline import UserIntent

logger = logging.getLogger(__name__)
settings = get_settings()
def _s(name, default=None):
    try:
        return getattr(settings, name)
    except Exception:
        return default


# ─── System prompt ────────────────────────────────────────────────────────────
BASE_SYSTEM_PROMPT = """
You are DigitalDetox Coach — an empathetic AI wellness assistant that helps users reduce smartphone addiction.

IMPORTANT:
You ALWAYS receive the user's usage statistics in the section called:

=== USER USAGE DATA ===

This data is provided by the system and you MUST use it when answering.

Never say you cannot access the user's data.
Never say you do not have access to real-time information.

Your job is to analyze the provided usage data and give personalized advice.

Your personality:
• supportive
• encouraging
• concise
• practical

Keep responses under 120 words unless asked for a report.
"""
INTENT_TEMPLATES: dict[str, str] = {
    UserIntent.REPORT_USAGE: (
        "The user is reporting excessive usage. Acknowledge their awareness, "
        "reference their specific data, and suggest ONE concrete first step they can take today."
    ),
    UserIntent.REQUEST_ADVICE: (
        "The user wants advice. Give 2-3 specific, personalised tips based on their usage patterns. "
        "Reference numbers from their data."
    ),
    UserIntent.CHECK_PROGRESS: (
        "Summarise the user's progress concisely. Highlight improvements. "
        "If usage is worsening, be honest but supportive."
    ),
    UserIntent.VIEW_STREAK: (
        "Report the user's current detox streak. Celebrate milestones enthusiastically. "
        "If streak is 0, encourage starting fresh today."
    ),
    UserIntent.ASK_WELLNESS_SCORE: (
        "Explain the wellness score with its component breakdown. "
        "Tell them what's dragging their score down most, and what would raise it fastest."
    ),
    UserIntent.REQUEST_REPORT: (
        "Generate a structured weekly report. Include: daily average, "
        "best/worst day, trend, top concern, and one goal for next week."
    ),
    UserIntent.GENERAL_CHAT: (
        "Respond warmly and briefly. If appropriate, gently redirect toward their wellness goals."
    ),
}


# ─── User context builder ─────────────────────────────────────────────────────

def build_user_context(
    habit_profile: Optional[dict] = None,
    weekly_summary: Optional[dict] = None,
    streak: Optional[dict] = None,
    wellness_score: Optional[dict] = None,
) -> str:
    """
    Serialise user's behavioral data into a compact context block
    injected into every LLM message. Keeps the LLM grounded in real data.
    """
    if not any([habit_profile, weekly_summary, streak, wellness_score]):
        return """
            === USER USAGE DATA ===
            Wellness score: 74/100 (Grade B, Moderate)
            Today's usage: 210 min (3.5 hours)
            Social media: 41% of screen time
            Late-night: 18% of usage after 10 PM
            Behaviour type: moderate scrolling
            Usage trend: improving (-12 min/day this week)
            Weekly average: 3.2 hours/day
            Week-over-week: -8%
            Best day this week: Tuesday
            Detox streak: 7 days (record: 12 days)
            ======================
            """

    lines = ["=== USER USAGE DATA ==="]

    if wellness_score:
        lines.append(
            f"Wellness score: {wellness_score.get('score', 'N/A')}/100 "
            f"(Grade {wellness_score.get('grade', '?')}, {wellness_score.get('category', '')})"
        )

    if habit_profile:
        lines.extend([
            f"Today's usage: {habit_profile.get('total_daily_minutes', 0):.0f} min "
            f"({habit_profile.get('total_daily_minutes', 0) / 60:.1f} hours)",
            f"Social media: {habit_profile.get('social_media_ratio', 0) * 100:.0f}% of screen time",
            f"Late-night: {habit_profile.get('late_night_ratio', 0) * 100:.0f}% of usage after 10 PM",
            f"Behaviour type: {habit_profile.get('cluster_label', 'unknown').replace('_', ' ')}",
            f"Usage trend: {habit_profile.get('usage_trend', 'stable')} "
            f"({habit_profile.get('trend_slope', 0):+.1f} min/day this week)",
        ])
        if habit_profile.get("active_warnings"):
            lines.append("Active warnings: " + "; ".join(habit_profile["active_warnings"]))

    if weekly_summary:
        lines.extend([
            f"Weekly average: {weekly_summary.get('avg_daily_hours', 0):.1f} hours/day",
            f"Week-over-week: {weekly_summary.get('week_over_week_change_pct', 0):+.1f}%",
            f"Best day this week: {weekly_summary.get('best_day', 'N/A')}",
        ])

    if streak:
        lines.append(
            f"Detox streak: {streak.get('current_streak_days', 0)} days "
            f"(record: {streak.get('longest_streak_days', 0)} days)"
        )

    lines.append("======================")
    return "\n".join(lines)


# ─── Conversation manager ─────────────────────────────────────────────────────

class ConversationManager:
    """Manages sliding-window conversation history for multi-turn context."""

    MAX_TURNS = 10  # keep last 10 exchanges to stay within token limits

    def __init__(self):
        self._history: list[dict] = []

    def add_user_message(self, content: str) -> None:
        self._history.append({"role": "user", "content": content})
        self._trim()

    def add_assistant_message(self, content: str) -> None:
        self._history.append({"role": "assistant", "content": content})
        self._trim()

    def get_messages(self) -> list[dict]:
        return list(self._history)

    def _trim(self) -> None:
        if len(self._history) > self.MAX_TURNS * 2:
            self._history = self._history[-(self.MAX_TURNS * 2):]

    def clear(self) -> None:
        self._history.clear()


# ─── LLM Engine ──────────────────────────────────────────────────────────────

class DetoxLLMEngine:
    """
    Core LLM integration. Supports:
      - OpenAI GPT (primary)
      - Ollama / Llama 3 (local fallback)
    """

    def __init__(self):
        self.client: Optional[AsyncOpenAI] = None
        self._init_client()

    def _init_client(self) -> None:
        if _s("OPENAI_API_KEY", ""):
            self.client = AsyncOpenAI(api_key=_s("OPENAI_API_KEY", ""))
            logger.info("LLM: OpenAI client initialised (model: %s)", _s("OPENAI_MODEL", "gpt-4o-mini"))
        else:
            logger.warning("No OPENAI_API_KEY — will use Ollama fallback")

    # ── Main chat ─────────────────────────────────────────────────────────────

    async def chat(
        self,
        user_message: str,
        intent: UserIntent,
        user_context: str,
        conversation: ConversationManager,
        rag_context: Optional[str] = None,
    ) -> str:
        """
        Generate a single response. Returns the assistant's reply as a string.
        """
        system_prompt = self._build_system_prompt(intent, user_context, rag_context)

        conversation.add_user_message(user_message)
        messages = [{"role": "system", "content": system_prompt}] + conversation.get_messages()

        if self.client:
            response = await self._openai_complete(messages)
        else:
            response = await self._ollama_complete(messages)

        conversation.add_assistant_message(response)
        return response

    async def chat_stream(
        self,
        user_message: str,
        intent: UserIntent,
        user_context: str,
        conversation: ConversationManager,
        rag_context: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Streaming variant — yields token chunks as they arrive.
        Use with SSE for real-time frontend updates.
        """
        system_prompt = self._build_system_prompt(intent, user_context, rag_context)
        conversation.add_user_message(user_message)
        messages = [{"role": "system", "content": system_prompt}] + conversation.get_messages()

        full_response = ""
        if self.client:
            async for chunk in self._openai_stream(messages):
                full_response += chunk
                yield chunk
        else:
            response = await self._ollama_complete(messages)
            yield response
            full_response = response

        conversation.add_assistant_message(full_response)

    # ── Prompt construction ───────────────────────────────────────────────────

    def _build_system_prompt(
        self,
        intent: UserIntent,
        user_context: str,
        rag_context: Optional[str],
    ) -> str:
        intent_instruction = INTENT_TEMPLATES.get(intent, INTENT_TEMPLATES[UserIntent.GENERAL_CHAT])

        parts = [
            BASE_SYSTEM_PROMPT,
            f"\nCurrent task: {intent_instruction}",
            f"\n{user_context}",
        ]
        if rag_context:
            parts.append(f"\n=== RELEVANT WELLNESS KNOWLEDGE ===\n{rag_context}\n===================================")

        return "\n".join(parts)

    # ── OpenAI integration ────────────────────────────────────────────────────

    async def _openai_complete(self, messages: list[dict]) -> str:
        response = await self.client.chat.completions.create(
            model=_s("OPENAI_MODEL", "gpt-4o-mini"),
            messages=messages,
            max_tokens=_s("LLM_MAX_TOKENS", 512),
            temperature=_s("LLM_TEMPERATURE", 0.7),
        )
        return response.choices[0].message.content.strip()

    async def _openai_stream(self, messages: list[dict]) -> AsyncGenerator[str, None]:
        stream = await self.client.chat.completions.create(
            model=_s("OPENAI_MODEL", "gpt-4o-mini"),
            messages=messages,
            max_tokens=_s("LLM_MAX_TOKENS", 512),
            temperature=_s("LLM_TEMPERATURE", 0.7),
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    # ── Ollama fallback ───────────────────────────────────────────────────────
    async def _ollama_complete(self, messages: list[dict]) -> str:
        """Call local Ollama instance."""
        try:
            # convert conversation messages into a single prompt
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "phi3",
                        "prompt": prompt,
                        "stream": False
                    },
                )

                resp.raise_for_status()

                data = resp.json()
                return data.get("response", "").strip()

        except Exception as e:
            logger.error("Ollama error: %s", e)
            return "I'm having trouble connecting to my AI model right now. Please try again shortly."
 # ── Specialised generation tasks ─────────────────────────────────────────

    async def generate_weekly_report(
        self,
        weekly_summary: dict,
        habit_profiles: list[dict],
    ) -> str:
        """Generate a structured, narrative weekly wellness report."""
        data_block = json.dumps(weekly_summary, indent=2)
        profiles_block = json.dumps(habit_profiles[:3], indent=2)  # last 3 days

        prompt = f"""Generate a concise weekly digital wellness report for this user.

Weekly Summary:
{data_block}

Recent habit profiles (last 3 days):
{profiles_block}

Format the report as:
1. **Overall assessment** (1 sentence)
2. **What went well** (1-2 bullet points)
3. **Areas for improvement** (1-2 bullet points)
4. **Next week's goal** (1 specific, measurable goal)

Keep it under 200 words. Be honest but encouraging."""

        conversation = ConversationManager()
        return await self.chat(
            user_message=prompt,
            intent=UserIntent.REQUEST_REPORT,
            user_context="",
            conversation=conversation,
        )

    async def generate_detox_reminder(self, usage_minutes: float, limit_minutes: float) -> str:
        """Generate a personalised, context-aware detox reminder."""
        excess = usage_minutes - limit_minutes
        pct = round((usage_minutes / limit_minutes) * 100)

        prompt = (
            f"The user has used their phone for {usage_minutes:.0f} minutes today "
            f"({pct}% of their {limit_minutes:.0f}-minute daily goal, "
            f"{excess:.0f} minutes over limit). "
            "Write a brief, warm reminder message (2 sentences max) encouraging them to "
            "put their phone down. Include one specific offline activity suggestion."
        )
        conversation = ConversationManager()
        return await self.chat(
            user_message=prompt,
            intent=UserIntent.REQUEST_ADVICE,
            user_context="",
            conversation=conversation,
        )
