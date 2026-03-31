"""
preprocessor.py — Production-grade data preprocessing pipeline.

Stages:
  1. Schema validation & type coercion
  2. Missing value imputation (median/mode strategy)
  3. Duplicate removal with timestamp deduplication
  4. Outlier detection  — IQR method + Z-score cross-check
  5. Noise filtering    — rolling-window smoothing for time-series
  6. Timestamp alignment to UTC + daily bucketing
  7. Normalization      — MinMax for bounded features, StandardScaler for unbounded
  8. Categorical encoding for app categories
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

logger = logging.getLogger(__name__)


# ─── Data contracts ────────────────────────────────────────────────────────────

@dataclass
class RawUsageEvent:
    """A single screen-time session captured from the device agent."""
    user_id: str
    app_name: str
    app_category: str          # social, productivity, entertainment, utility
    session_start: datetime
    session_end: datetime
    notification_count: int
    is_late_night: bool = field(default=False)


@dataclass
class ProcessedUsageRecord:
    """Cleaned, normalized record ready for feature engineering."""
    user_id: str
    date: str                  # YYYY-MM-DD
    app_name: str
    app_category_encoded: int
    duration_minutes: float
    session_start_hour: float  # 0-23, normalised to [0,1]
    notification_count: int
    is_late_night: int         # 0 or 1
    duration_normalized: float
    duration_zscore: float


# ─── Preprocessor ──────────────────────────────────────────────────────────────

class UsageDataPreprocessor:
    """
    Stateful preprocessor — fits scalers on training data and transforms
    new records consistently. Designed for incremental online updates.
    """

    SOCIAL_MEDIA_APPS = {
        "Instagram", "TikTok", "Twitter", "Facebook",
        "Snapchat", "YouTube", "Reddit", "Pinterest",
    }
    EXPECTED_CATEGORIES = ["social", "productivity", "entertainment", "utility", "other"]

    def __init__(self):
        self.minmax_scaler = MinMaxScaler()
        self.std_scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.EXPECTED_CATEGORIES)
        self._fitted = False
        self._outlier_bounds: dict[str, tuple[float, float]] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def fit_transform(self, events: list[RawUsageEvent]) -> pd.DataFrame:
        """Fit scalers and return cleaned DataFrame. Use on historical data."""
        df = self._to_dataframe(events)
        df = self._validate_and_coerce(df)
        df = self._remove_duplicates(df)
        df = self._impute_missing(df)
        df = self._remove_outliers(df, fit=True)
        df = self._smooth_time_series(df)
        df = self._align_timestamps(df)
        df = self._encode_categoricals(df)
        df = self._normalize(df, fit=True)
        self._fitted = True
        logger.info("Preprocessor fitted on %d records → %d cleaned", len(events), len(df))
        return df

    def transform(self, events: list[RawUsageEvent]) -> pd.DataFrame:
        """Transform new records using already-fitted scalers."""
        if not self._fitted:
            raise RuntimeError("Call fit_transform() before transform().")
        df = self._to_dataframe(events)
        df = self._validate_and_coerce(df)
        df = self._remove_duplicates(df)
        df = self._impute_missing(df)
        df = self._remove_outliers(df, fit=False)
        df = self._smooth_time_series(df)
        df = self._align_timestamps(df)
        df = self._encode_categoricals(df)
        df = self._normalize(df, fit=False)
        return df

    # ── Stage 1: Schema coercion ───────────────────────────────────────────────

    def _to_dataframe(self, events: list[RawUsageEvent]) -> pd.DataFrame:
        records = []
        for e in events:
            records.append({
                "user_id": e.user_id,
                "app_name": e.app_name.strip(),
                "app_category": e.app_category.lower().strip(),
                "session_start": e.session_start,
                "session_end": e.session_end,
                "notification_count": max(0, int(e.notification_count)),
                "is_late_night": int(e.is_late_night),
            })
        return pd.DataFrame(records)

    def _validate_and_coerce(self, df: pd.DataFrame) -> pd.DataFrame:
        df["session_start"] = pd.to_datetime(df["session_start"], utc=True, errors="coerce")
        df["session_end"] = pd.to_datetime(df["session_end"], utc=True, errors="coerce")
        # Drop rows with unparseable timestamps
        before = len(df)
        df = df.dropna(subset=["session_start", "session_end"])
        # Duration must be positive
        df["duration_minutes"] = (
            (df["session_end"] - df["session_start"]).dt.total_seconds() / 60
        )
        df = df[df["duration_minutes"] > 0]
        # Cap unrealistically long sessions (>8 h = likely error)
        df = df[df["duration_minutes"] <= 480]
        logger.debug("Validation: %d → %d rows", before, len(df))
        return df

    # ── Stage 2: Duplicate removal ────────────────────────────────────────────

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates(subset=["user_id", "app_name", "session_start"])
        logger.debug("Deduplication: removed %d duplicates", before - len(df))
        return df

    # ── Stage 3: Missing value imputation ────────────────────────────────────

    def _impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        df["notification_count"] = df["notification_count"].fillna(
            df["notification_count"].median()
        ).astype(int)
        df["app_category"] = df["app_category"].fillna("other")
        df["app_name"] = df["app_name"].fillna("Unknown")
        return df

    # ── Stage 4: Outlier detection (IQR + Z-score) ───────────────────────────

    def _remove_outliers(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        col = "duration_minutes"
        if fit:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            self._outlier_bounds[col] = (lower, upper)

        lower, upper = self._outlier_bounds.get(col, (0, 480))
        before = len(df)

        # IQR filter
        df = df[(df[col] >= lower) & (df[col] <= upper)]

        # Z-score cross-check: flag extreme outliers missed by IQR
        zscores = np.abs(stats.zscore(df[col]))
        df = df[zscores < 3.5]

        logger.debug("Outlier removal: removed %d rows", before - len(df))
        return df

    # ── Stage 5: Noise filtering (rolling window smooth) ──────────────────────

    def _smooth_time_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        For each user, compute a 7-day rolling mean of daily usage.
        Stored as `smoothed_daily_minutes` — used by trend analysis.
        """
        daily = (
            df.groupby(["user_id", pd.Grouper(key="session_start", freq="D")])
            ["duration_minutes"].sum()
            .reset_index()
            .rename(columns={"session_start": "date", "duration_minutes": "daily_minutes"})
        )
        daily["smoothed_daily_minutes"] = (
            daily.groupby("user_id")["daily_minutes"]
            .transform(lambda s: s.rolling(7, min_periods=1).mean())
        )
        df = df.merge(
            daily[["user_id", "date", "smoothed_daily_minutes"]],
            left_on=["user_id", df["session_start"].dt.normalize()],
            right_on=["user_id", "date"],
            how="left",
        ).drop(columns=["date", "key_1"], errors="ignore")
        df["smoothed_daily_minutes"] = df["smoothed_daily_minutes"].fillna(
            df["duration_minutes"]
        )
        return df

    # ── Stage 6: Timestamp alignment ──────────────────────────────────────────

    def _align_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        df["date"] = df["session_start"].dt.date.astype(str)
        df["session_start_hour"] = df["session_start"].dt.hour + df["session_start"].dt.minute / 60
        return df

    # ── Stage 7: Categorical encoding ─────────────────────────────────────────

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        # Clip unknown categories to 'other'
        df["app_category"] = df["app_category"].where(
            df["app_category"].isin(self.EXPECTED_CATEGORIES), other="other"
        )
        df["app_category_encoded"] = self.label_encoder.transform(df["app_category"])
        df["is_social_media"] = df["app_name"].isin(self.SOCIAL_MEDIA_APPS).astype(int)
        return df

    # ── Stage 8: Normalization ─────────────────────────────────────────────────

    def _normalize(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        minmax_cols = ["session_start_hour", "notification_count"]
        std_cols = ["duration_minutes"]

        if fit:
            df[[f"{c}_normalized" for c in minmax_cols]] = (
                self.minmax_scaler.fit_transform(df[minmax_cols])
            )
            df["duration_zscore"] = self.std_scaler.fit_transform(
                df[["duration_minutes"]]
            )
        else:
            df[[f"{c}_normalized" for c in minmax_cols]] = (
                self.minmax_scaler.transform(df[minmax_cols])
            )
            df["duration_zscore"] = self.std_scaler.transform(
                df[["duration_minutes"]]
            )

        # MinMax for duration
        dur_min = df["duration_minutes"].min()
        dur_max = df["duration_minutes"].max()
        df["duration_normalized"] = (
            (df["duration_minutes"] - dur_min) / (dur_max - dur_min + 1e-9)
        )
        return df

    # ── Utility ───────────────────────────────────────────────────────────────

    def get_summary_stats(self, df: pd.DataFrame) -> dict[str, Any]:
        return {
            "total_records": len(df),
            "unique_users": df["user_id"].nunique(),
            "date_range": {
                "start": str(df["session_start"].min()),
                "end": str(df["session_start"].max()),
            },
            "avg_session_minutes": round(float(df["duration_minutes"].mean()), 2),
            "social_media_ratio": round(float(df["is_social_media"].mean()), 3),
            "late_night_ratio": round(float(df["is_late_night"].mean()), 3),
        }
