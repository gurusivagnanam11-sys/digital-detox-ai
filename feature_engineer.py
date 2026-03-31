"""
feature_engineer.py — Behavioral feature engineering for digital wellness.

Produces one feature vector per (user, date) that captures:
  - Daily usage intensity
  - Social media dependency signals
  - Late-night usage patterns
  - Focus productivity performance
  - Week-over-week growth trends
  - App-switching cognitive load proxy
  - Digital wellness indicators
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import linregress

logger = logging.getLogger(__name__)


class BehavioralFeatureEngineer:
    """
    Transforms preprocessed usage DataFrame into a rich feature matrix.
    Each row = one user-day. Output is ready for ML models or the LLM context.
    """

    def __init__(
        self,
        late_night_start: int = 22,
        late_night_end: int = 6,
        social_apps: Optional[set[str]] = None,
    ):
        self.late_night_start = late_night_start
        self.late_night_end = late_night_end
        self.social_apps = social_apps or {
            "Instagram", "TikTok", "Twitter", "Facebook",
            "Snapchat", "YouTube", "Reddit", "Pinterest",
        }

    def engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main entry point. Returns a per-user-day feature DataFrame.
        """
        logger.info("Starting feature engineering on %d session records", len(df))

        features = (
            df.groupby(["user_id", "date"])
            .apply(self._compute_daily_features)
            .reset_index()
        )

        # Week-over-week trend features (require > 7 days of history per user)
        features = features.sort_values(["user_id", "date"])
        features = self._add_trend_features(features)
        features = self._add_wellness_indicators(features)

        logger.info("Feature engineering complete: %d user-days, %d features",
                    len(features), features.shape[1])
        return features

    # ── Per-day aggregate features ────────────────────────────────────────────

    def _compute_daily_features(self, group: pd.DataFrame) -> pd.Series:
        total_minutes = group["duration_minutes"].sum()
        session_count = len(group)
        avg_session = total_minutes / max(session_count, 1)

        social_minutes = group.loc[group["is_social_media"] == 1, "duration_minutes"].sum()
        social_ratio = social_minutes / max(total_minutes, 1)

        late_night_minutes = group.loc[group["is_late_night"] == 1, "duration_minutes"].sum()
        late_night_ratio = late_night_minutes / max(total_minutes, 1)

        notification_rate = (
            group["notification_count"].sum() / max(total_minutes / 60, 0.1)
        )  # notifications per hour

        # App-switch frequency as cognitive load proxy:
        # count distinct apps * sessions / time window
        distinct_apps = group["app_name"].nunique()
        app_switch_freq = (distinct_apps * session_count) / max(total_minutes / 60, 0.1)

        # Peak usage hour — hour with most total minutes
        hour_usage = group.groupby("session_start_hour")["duration_minutes"].sum()
        peak_hour = float(hour_usage.idxmax()) if len(hour_usage) > 0 else 12.0

        # Evening usage concentration (6 PM–midnight)
        evening_minutes = group.loc[
            (group["session_start_hour"] >= 18) & (group["session_start_hour"] < 24),
            "duration_minutes",
        ].sum()
        evening_ratio = evening_minutes / max(total_minutes, 1)

        return pd.Series({
            "total_daily_minutes": round(total_minutes, 2),
            "session_count": session_count,
            "avg_session_minutes": round(avg_session, 2),
            "social_media_ratio": round(social_ratio, 4),
            "social_media_minutes": round(social_minutes, 2),
            "late_night_ratio": round(late_night_ratio, 4),
            "late_night_minutes": round(late_night_minutes, 2),
            "notification_rate_per_hour": round(notification_rate, 2),
            "app_switch_frequency": round(app_switch_freq, 3),
            "distinct_apps_used": distinct_apps,
            "peak_usage_hour": peak_hour,
            "evening_usage_ratio": round(evening_ratio, 4),
        })

    # ── Trend features (rolling windows) ─────────────────────────────────────

    def _add_trend_features(self, features: pd.DataFrame) -> pd.DataFrame:
        def rolling_growth(series: pd.Series, window: int = 7) -> pd.Series:
            """Week-over-week growth rate using linear regression slope."""
            def slope(s: pd.Series) -> float:
                if len(s) < 3:
                    return 0.0
                x = np.arange(len(s), dtype=float)
                slope_val, *_ = linregress(x, s.values)
                return float(slope_val)

            return series.rolling(window, min_periods=3).apply(slope, raw=False)

        for user_id, group in features.groupby("user_id"):
            idx = group.index
            features.loc[idx, "usage_trend_slope"] = (
                rolling_growth(group["total_daily_minutes"])
            )
            features.loc[idx, "social_trend_slope"] = (
                rolling_growth(group["social_media_ratio"])
            )
            features.loc[idx, "late_night_trend_slope"] = (
                rolling_growth(group["late_night_ratio"])
            )
            # 7-day rolling average for smoothed comparisons
            features.loc[idx, "weekly_avg_minutes"] = (
                group["total_daily_minutes"].rolling(7, min_periods=1).mean()
            )
            features.loc[idx, "weekly_avg_social_ratio"] = (
                group["social_media_ratio"].rolling(7, min_periods=1).mean()
            )

        features["usage_trend_slope"] = features["usage_trend_slope"].fillna(0.0)
        features["social_trend_slope"] = features["social_trend_slope"].fillna(0.0)
        features["late_night_trend_slope"] = features["late_night_trend_slope"].fillna(0.0)
        return features

    # ── Wellness indicator flags ───────────────────────────────────────────────

    def _add_wellness_indicators(self, features: pd.DataFrame) -> pd.DataFrame:
        # Binary indicator: excessive daily usage (> 4 hours)
        features["excessive_usage_flag"] = (
            features["total_daily_minutes"] > 240
        ).astype(int)

        # Binary: high social media dependency
        features["social_dependency_flag"] = (
            features["social_media_ratio"] > 0.40
        ).astype(int)

        # Binary: late-night addiction signal
        features["night_addiction_flag"] = (
            features["late_night_ratio"] > 0.20
        ).astype(int)

        # Binary: productivity decline (upward trend AND high usage)
        features["productivity_decline_flag"] = (
            (features["usage_trend_slope"] > 2) &
            (features["total_daily_minutes"] > 180)
        ).astype(int)

        # Composite: number of active warning flags
        flag_cols = [
            "excessive_usage_flag", "social_dependency_flag",
            "night_addiction_flag", "productivity_decline_flag",
        ]
        features["active_warning_count"] = features[flag_cols].sum(axis=1)

        return features

    def get_feature_names(self) -> list[str]:
        return [
            "total_daily_minutes", "session_count", "avg_session_minutes",
            "social_media_ratio", "social_media_minutes",
            "late_night_ratio", "late_night_minutes",
            "notification_rate_per_hour", "app_switch_frequency",
            "distinct_apps_used", "peak_usage_hour", "evening_usage_ratio",
            "usage_trend_slope", "social_trend_slope", "late_night_trend_slope",
            "weekly_avg_minutes", "weekly_avg_social_ratio",
            "excessive_usage_flag", "social_dependency_flag",
            "night_addiction_flag", "productivity_decline_flag",
            "active_warning_count",
        ]
