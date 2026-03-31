"""
streak_tracker.py — Detox streak calculation and Pomodoro focus mode engine.

Streak Logic:
  - A "successful detox day" = total usage < user's daily limit
  - Streak increments at midnight if yesterday was successful
  - Streak resets if a day is missed or limit exceeded
  - Milestone notifications at: 3, 7, 14, 30, 60, 100 days

Focus Mode:
  - Standard Pomodoro: 25 min focus / 5 min short break / 15 min long break
  - Custom intervals supported
  - Tracks completion rate for wellness score contribution
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)


STREAK_MILESTONES = {3: "🌱", 7: "🔥", 14: "⭐", 30: "🏆", 60: "💎", 100: "🦋"}


@dataclass
class StreakStatus:
    current_streak: int
    longest_streak: int
    total_detox_days: int
    last_success_date: Optional[str]
    today_on_track: bool          # True if today's usage so far is within limit
    milestone_reached: Optional[int]
    milestone_emoji: str
    message: str


class StreakTracker:

    def __init__(self, daily_limit_minutes: float = 180.0):
        self.daily_limit = daily_limit_minutes

    def update_streak(
        self,
        current_streak: int,
        longest_streak: int,
        total_detox_days: int,
        last_success_date: Optional[str],
        today_total_minutes: float,
        today_date: Optional[date] = None,
    ) -> StreakStatus:
        """
        Call this at the end of each day (or on demand) to update streak state.
        Returns new streak values and milestone information.
        """
        today = today_date or date.today()
        today_str = str(today)
        yesterday_str = str(today - timedelta(days=1))

        today_success = today_total_minutes <= self.daily_limit

        if today_success:
            # Check if this is a consecutive day
            if last_success_date in (yesterday_str, today_str):
                if last_success_date != today_str:
                    current_streak += 1
                    total_detox_days += 1
            else:
                # Streak reset or first day
                current_streak = 1
                total_detox_days += 1

            longest_streak = max(longest_streak, current_streak)
            last_success_date = today_str
        else:
            # Day failed — reset streak only if we've passed the day
            if last_success_date and last_success_date < yesterday_str:
                current_streak = 0  # missed a day

        # Check milestones
        milestone_reached = None
        for milestone, emoji in STREAK_MILESTONES.items():
            if current_streak == milestone:
                milestone_reached = milestone
                break

        message = self._generate_streak_message(current_streak, today_success, milestone_reached)

        return StreakStatus(
            current_streak=current_streak,
            longest_streak=longest_streak,
            total_detox_days=total_detox_days,
            last_success_date=last_success_date,
            today_on_track=today_success,
            milestone_reached=milestone_reached,
            milestone_emoji=STREAK_MILESTONES.get(milestone_reached, "") if milestone_reached else "",
            message=message,
        )

    def _generate_streak_message(
        self,
        streak: int,
        today_success: bool,
        milestone: Optional[int],
    ) -> str:
        if milestone:
            emoji = STREAK_MILESTONES[milestone]
            msgs = {
                3:   f"{emoji} 3-day streak! Your digital detox habit is forming.",
                7:   f"{emoji} One week streak! Your brain is adapting to less screen time.",
                14:  f"{emoji} Two weeks! You've broken the worst of the habit loop.",
                30:  f"{emoji} 30-DAY STREAK! You've transformed your relationship with technology.",
                60:  f"{emoji} 60 days. Digital wellness is now part of who you are.",
                100: f"{emoji} 100 DAYS! You are a digital wellness champion.",
            }
            return msgs.get(milestone, f"{emoji} Amazing {milestone}-day streak!")

        if not today_success:
            return f"Today went over the limit. Your {streak}-day streak is at risk — finish the day mindfully."
        if streak == 0:
            return "Fresh start! Every expert was once a beginner."
        if streak < 3:
            return f"Day {streak} ✓ — keep going!"
        return f"{streak}-day streak — consistency is building!"


# ─── Pomodoro Focus Timer ─────────────────────────────────────────────────────

@dataclass
class PomodoroConfig:
    focus_minutes: int = 25
    short_break_minutes: int = 5
    long_break_minutes: int = 15
    sessions_before_long_break: int = 4


@dataclass
class PomodoroSession:
    session_number: int         # 1, 2, 3, 4, then long break
    phase: str                  # "focus" | "short_break" | "long_break"
    duration_minutes: int
    started_at: datetime
    ends_at: datetime
    completed: bool = False
    interrupted_at_minute: Optional[float] = None


class PomodoroEngine:
    """
    Stateful Pomodoro engine per user. Tracks session sequences and calculates
    focus productivity scores.
    """

    def __init__(self, config: Optional[PomodoroConfig] = None):
        self.config = config or PomodoroConfig()
        self._completed_today: int = 0
        self._session_history: list[PomodoroSession] = []

    def start_session(self, current_count_today: int = 0) -> PomodoroSession:
        """
        Start the next appropriate session (focus or break) based on current count.
        """
        self._completed_today = current_count_today
        now = datetime.now(timezone.utc)

        if self._completed_today > 0 and self._completed_today % self.config.sessions_before_long_break == 0:
            phase = "long_break"
            duration = self.config.long_break_minutes
        elif self._completed_today % self.config.sessions_before_long_break != 0:
            # Alternating: if last was focus, take short break
            phase = "short_break" if self._last_was_focus() else "focus"
            duration = (
                self.config.short_break_minutes
                if phase == "short_break"
                else self.config.focus_minutes
            )
        else:
            phase = "focus"
            duration = self.config.focus_minutes

        session = PomodoroSession(
            session_number=self._completed_today + 1,
            phase=phase,
            duration_minutes=duration,
            started_at=now,
            ends_at=now + timedelta(minutes=duration),
        )
        self._session_history.append(session)
        return session

    def _last_was_focus(self) -> bool:
        for s in reversed(self._session_history):
            if s.completed:
                return s.phase == "focus"
        return True  # default: first session is always focus

    def complete_session(self) -> dict:
        if not self._session_history:
            return {"error": "No active session"}

        last = self._session_history[-1]
        last.completed = True

        if last.phase == "focus":
            self._completed_today += 1

        focus_completed = sum(1 for s in self._session_history if s.phase == "focus" and s.completed)
        productivity_score = min(100, focus_completed * 25)  # 4 sessions = 100%

        return {
            "phase_completed": last.phase,
            "focus_sessions_today": focus_completed,
            "productivity_score": productivity_score,
            "next_phase": self._next_phase_info(),
            "encouragement": self._get_encouragement(focus_completed),
        }

    def _next_phase_info(self) -> dict:
        focus_done = sum(1 for s in self._session_history if s.phase == "focus" and s.completed)
        sessions_until_long = (
            self.config.sessions_before_long_break - (focus_done % self.config.sessions_before_long_break)
        )
        return {
            "sessions_until_long_break": sessions_until_long,
            "next_focus_number": focus_done + 1,
        }

    def _get_encouragement(self, focus_sessions: int) -> str:
        messages = {
            1: "First focus session done! You're building momentum.",
            2: "Two sessions in — your brain is in the zone.",
            3: "Three sessions! One more for a long break.",
            4: "Full Pomodoro cycle! Take a proper 15-minute break. You earned it 🎉",
            5: "5 sessions — extraordinary focus today!",
            6: "Elite-level focus. Your phone can wait.",
            8: "8 Pomodoros? That's a master-level productive day.",
        }
        return messages.get(focus_sessions, f"{focus_sessions} focus sessions today. Excellent work!")

    def get_daily_stats(self) -> dict:
        focus_done = sum(1 for s in self._session_history if s.phase == "focus" and s.completed)
        interrupted = sum(1 for s in self._session_history if s.interrupted_at_minute is not None)
        completion_rate = (
            focus_done / max(len([s for s in self._session_history if s.phase == "focus"]), 1)
        )
        return {
            "focus_sessions_completed": focus_done,
            "sessions_interrupted": interrupted,
            "completion_rate": round(completion_rate, 2),
            "total_focus_minutes": focus_done * self.config.focus_minutes,
            "productivity_score": min(100, focus_done * 25),
        }
