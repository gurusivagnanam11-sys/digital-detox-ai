"""
habit_engine.py — ML-powered digital habit analysis.

Models:
  1. K-Means clustering     — user behavior archetype discovery
  2. Isolation Forest       — anomaly / addiction spike detection
  3. Linear Regression      — usage trend forecasting
  4. Rule-based classifier  — habit severity classification
  5. Wellness Score         — composite 0–100 metric
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


# ─── Result types ─────────────────────────────────────────────────────────────

@dataclass
class HabitProfile:
    user_id: str
    date: str
    cluster_id: int
    cluster_label: str         # "balanced_user", "social_heavy", "night_owl", etc.
    is_anomaly: bool
    anomaly_score: float       # Isolation Forest score (negative = anomalous)
    usage_trend: str           # "increasing" | "stable" | "decreasing"
    trend_slope: float
    active_warnings: list[str]
    wellness_score: float
    wellness_grade: str
    wellness_category: str


@dataclass
class WellnessScoreBreakdown:
    total_score: float          # 0–100
    usage_component: float      # 40 pts max
    social_component: float     # 20 pts max
    night_component: float      # 20 pts max
    trend_component: float      # 10 pts max
    focus_component: float      # 10 pts max
    grade: str
    category: str


# ─── Cluster label map ────────────────────────────────────────────────────────

CLUSTER_ARCHETYPES = {
    "balanced_user":     "You maintain healthy digital habits with moderate usage.",
    "social_heavy":      "High social media usage detected. Consider limiting social apps.",
    "night_owl":         "Significant late-night phone usage. Try a digital sunset routine.",
    "productivity_drop": "Increasing screen time trend affecting productivity.",
    "high_notif":        "Very high notification frequency causing constant interruptions.",
    "overuser":          "Daily usage significantly above healthy limits.",
}

FEATURE_COLS_FOR_ML = [
    "total_daily_minutes", "social_media_ratio", "late_night_ratio",
    "notification_rate_per_hour", "app_switch_frequency",
    "usage_trend_slope", "evening_usage_ratio",
]


# ─── Habit Analysis Engine ────────────────────────────────────────────────────

class HabitAnalysisEngine:
    """
    Stateful engine — fit once on population data, then predict per user.
    Intended for nightly batch jobs + real-time single-user inference.
    """

    N_CLUSTERS = 6  # one per archetype

    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=self.N_CLUSTERS, random_state=42, n_init=10)
        self.isolation_forest = IsolationForest(
            contamination=0.1,       # ~10% of days expected anomalous
            random_state=42,
            n_estimators=200,
        )
        self._fitted = False
        self._cluster_labels: dict[int, str] = {}

    # ── Training ─────────────────────────────────────────────────────────────

    def fit(self, features_df: pd.DataFrame) -> None:
        """
        Fit clustering and anomaly models on population feature data.
        Call once with historical data for all users.
        """
        X = self._prepare_matrix(features_df)
        X_scaled = self.scaler.fit_transform(X)

        # K-Means clustering
        self.kmeans.fit(X_scaled)
        self._assign_cluster_labels(features_df, X_scaled)

        # Isolation Forest for anomaly detection
        self.isolation_forest.fit(X_scaled)

        self._fitted = True
        logger.info("HabitAnalysisEngine fitted on %d records, %d clusters", len(X), self.N_CLUSTERS)

    def _assign_cluster_labels(self, df: pd.DataFrame, X_scaled: np.ndarray) -> None:
        """
        Map cluster IDs to human-readable archetypes by inspecting cluster centroids.
        """
        df = df.copy()
        df["_cluster"] = self.kmeans.labels_

        cluster_stats = df.groupby("_cluster")[FEATURE_COLS_FOR_ML].mean()
        archetype_list = list(CLUSTER_ARCHETYPES.keys())

        for cluster_id, row in cluster_stats.iterrows():
            # Heuristic assignment based on dominant signal in centroid
            if row["late_night_ratio"] > 0.25:
                label = "night_owl"
            elif row["social_media_ratio"] > 0.40:
                label = "social_heavy"
            elif row["total_daily_minutes"] > 300:
                label = "overuser"
            elif row["usage_trend_slope"] > 3 and row["total_daily_minutes"] > 200:
                label = "productivity_drop"
            elif row["notification_rate_per_hour"] > 15:
                label = "high_notif"
            else:
                label = "balanced_user"
            self._cluster_labels[int(cluster_id)] = label

        logger.debug("Cluster assignments: %s", self._cluster_labels)

    # ── Inference ────────────────────────────────────────────────────────────

    def analyze(self, feature_row: pd.Series) -> HabitProfile:
        """
        Analyze a single user-day feature vector. Returns full habit profile.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before analyze().")

        X = feature_row[FEATURE_COLS_FOR_ML].values.reshape(1, -1)
        X = np.nan_to_num(X, nan=0.0)
        X_scaled = self.scaler.transform(X)

        cluster_id = int(self.kmeans.predict(X_scaled)[0])
        cluster_label = self._cluster_labels.get(cluster_id, "balanced_user")

        anomaly_pred = self.isolation_forest.predict(X_scaled)[0]  # -1 = anomaly
        anomaly_score = float(self.isolation_forest.score_samples(X_scaled)[0])
        is_anomaly = anomaly_pred == -1

        trend_slope = float(feature_row.get("usage_trend_slope", 0.0))
        trend = (
            "increasing" if trend_slope > 1.5
            else "decreasing" if trend_slope < -1.5
            else "stable"
        )

        warnings = self._generate_warnings(feature_row)
        wellness = self._compute_wellness_score(feature_row)

        return HabitProfile(
            user_id=str(feature_row.get("user_id", "")),
            date=str(feature_row.get("date", "")),
            cluster_id=cluster_id,
            cluster_label=cluster_label,
            is_anomaly=is_anomaly,
            anomaly_score=round(anomaly_score, 4),
            usage_trend=trend,
            trend_slope=round(trend_slope, 3),
            active_warnings=warnings,
            wellness_score=wellness.total_score,
            wellness_grade=wellness.grade,
            wellness_category=wellness.category,
        )

    def analyze_batch(self, features_df: pd.DataFrame) -> list[HabitProfile]:
        return [self.analyze(row) for _, row in features_df.iterrows()]

    # ── Warning generation ────────────────────────────────────────────────────

    def _generate_warnings(self, row: pd.Series) -> list[str]:
        warnings = []
        minutes = float(row.get("total_daily_minutes", 0))
        social_ratio = float(row.get("social_media_ratio", 0))
        late_ratio = float(row.get("late_night_ratio", 0))
        notif_rate = float(row.get("notification_rate_per_hour", 0))
        trend = float(row.get("usage_trend_slope", 0))

        if minutes > 240:
            hours = round(minutes / 60, 1)
            warnings.append(f"Excessive usage: {hours}h today (limit 4h)")
        if social_ratio > 0.40:
            pct = round(social_ratio * 100)
            warnings.append(f"Social media: {pct}% of screen time")
        if late_ratio > 0.20:
            pct = round(late_ratio * 100)
            warnings.append(f"Late-night usage: {pct}% after 10 PM")
        if notif_rate > 20:
            warnings.append(f"High notification rate: {notif_rate:.0f}/hour")
        if trend > 3:
            warnings.append(f"Usage increasing {trend:.1f} min/day this week")

        return warnings

    # ── Wellness score ────────────────────────────────────────────────────────

    def _compute_wellness_score(self, row: pd.Series) -> WellnessScoreBreakdown:
        """
        Composite wellness score (0–100).

        Component weights:
          Usage intensity    — 40 pts (penalised for > 2h, severe penalty > 6h)
          Social media ratio — 20 pts (penalised linearly above 25%)
          Late-night usage   — 20 pts (penalised linearly above 10%)
          Usage trend        — 10 pts (bonus for decreasing, penalty for increasing)
          Focus sessions     — 10 pts (bonus for completing Pomodoros)
        """
        minutes = float(row.get("total_daily_minutes", 0))
        social_ratio = float(row.get("social_media_ratio", 0))
        late_ratio = float(row.get("late_night_ratio", 0))
        trend = float(row.get("usage_trend_slope", 0))
        focus_count = int(row.get("focus_sessions_today", 0))

        # Usage component (40 pts)
        if minutes <= 120:
            usage_pts = 40.0
        elif minutes <= 240:
            usage_pts = 40.0 - (minutes - 120) * (20 / 120)
        elif minutes <= 480:
            usage_pts = 20.0 - (minutes - 240) * (15 / 240)
        else:
            usage_pts = max(0.0, 5.0 - (minutes - 480) * 0.01)

        # Social component (20 pts)
        if social_ratio <= 0.25:
            social_pts = 20.0
        else:
            social_pts = max(0.0, 20.0 - (social_ratio - 0.25) * 80)

        # Night component (20 pts)
        if late_ratio <= 0.10:
            night_pts = 20.0
        else:
            night_pts = max(0.0, 20.0 - (late_ratio - 0.10) * 100)

        # Trend component (10 pts)
        if trend <= -1.0:
            trend_pts = 10.0   # improving
        elif trend <= 0:
            trend_pts = 7.0
        elif trend <= 2.0:
            trend_pts = 5.0
        else:
            trend_pts = max(0.0, 5.0 - trend)

        # Focus component (10 pts, max 2 Pomodoros per day for full score)
        focus_pts = min(10.0, focus_count * 5.0)

        total = round(usage_pts + social_pts + night_pts + trend_pts + focus_pts, 1)

        if total >= 90:
            grade, category = "A+", "excellent"
        elif total >= 80:
            grade, category = "A",  "excellent"
        elif total >= 70:
            grade, category = "B+", "good"
        elif total >= 60:
            grade, category = "B",  "good"
        elif total >= 50:
            grade, category = "C",  "fair"
        elif total >= 40:
            grade, category = "D",  "poor"
        else:
            grade, category = "F",  "critical"

        return WellnessScoreBreakdown(
            total_score=total,
            usage_component=round(usage_pts, 1),
            social_component=round(social_pts, 1),
            night_component=round(night_pts, 1),
            trend_component=round(trend_pts, 1),
            focus_component=round(focus_pts, 1),
            grade=grade,
            category=category,
        )

    # ── Weekly report data ────────────────────────────────────────────────────

    def generate_weekly_summary(self, features_df: pd.DataFrame, user_id: str) -> dict:
        """
        Aggregate weekly statistics for report generation.
        Expects 7 rows of daily features for the given user.
        """
        user_data = features_df[features_df["user_id"] == user_id].tail(7)
        if user_data.empty:
            return {}

        avg_daily = float(user_data["total_daily_minutes"].mean())
        avg_social = float(user_data["social_media_ratio"].mean())
        avg_night = float(user_data["late_night_ratio"].mean())
        best_day = user_data.loc[user_data["total_daily_minutes"].idxmin(), "date"]
        worst_day = user_data.loc[user_data["total_daily_minutes"].idxmax(), "date"]
        week_trend = float(user_data["usage_trend_slope"].iloc[-1])

        # Week-over-week comparison
        prev_7 = features_df[features_df["user_id"] == user_id].iloc[-14:-7]
        prev_avg = float(prev_7["total_daily_minutes"].mean()) if not prev_7.empty else avg_daily
        wow_change_pct = round((avg_daily - prev_avg) / max(prev_avg, 1) * 100, 1)

        return {
            "user_id": user_id,
            "week_end_date": str(user_data["date"].iloc[-1]),
            "avg_daily_minutes": round(avg_daily, 1),
            "avg_daily_hours": round(avg_daily / 60, 2),
            "avg_social_ratio_pct": round(avg_social * 100, 1),
            "avg_late_night_ratio_pct": round(avg_night * 100, 1),
            "best_day": best_day,
            "worst_day": worst_day,
            "week_over_week_change_pct": wow_change_pct,
            "trend": "improving" if wow_change_pct < -5 else "worsening" if wow_change_pct > 5 else "stable",
            "total_warnings": int(user_data["active_warning_count"].sum()),
        }
