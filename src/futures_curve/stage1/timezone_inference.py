"""Raw timestamp timezone inference utilities.

The raw TXT/CSV files in this project contain naive timestamps. Different data
vendors encode those timestamps in different timezones (commonly US/Eastern).

We infer the raw timezone by scoring candidate interpretations based on the
expected CME maintenance break when expressed in exchange time (US/Central).

Heuristic (CME-style products):
- If we convert raw timestamps -> US/Central and see substantial activity in
  the 16:00-16:59 CT hour (the typical maintenance break), the candidate is
  likely wrong.
- The *right* raw timezone should minimize that "maintenance hour leakage".

This is a heuristic; we persist the diagnostic histogram so assumptions are
auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from ..utils.timezone import CENTRAL_TZ


@dataclass(frozen=True)
class TimezoneInferenceResult:
    raw_timezone: str
    scores: dict[str, float]
    maintenance_leakage_pct: dict[str, float]
    sample_rows: int


def _score_candidate(ct_hours: pd.Series) -> float:
    """Score a candidate interpretation.

    Lower is better. Currently we focus on maintenance hour leakage.
    """
    total = ct_hours.size
    if total == 0:
        return float("inf")
    leak = (ct_hours == 16).mean() * 100.0
    return leak


def infer_raw_timezone(
    timestamp_strings: Iterable[str],
    candidates: list[str] | None = None,
) -> TimezoneInferenceResult:
    """Infer the raw timezone given an iterable of timestamp strings.

    Args:
        timestamp_strings: Iterable of raw timestamp strings ('YYYY-MM-DD HH:MM:SS')
        candidates: List of pytz timezone names to consider

    Returns:
        TimezoneInferenceResult
    """
    if candidates is None:
        candidates = ["US/Eastern", "US/Central", "UTC"]

    s = pd.Series(list(timestamp_strings), dtype="string")
    ts = pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S", errors="coerce").dropna()

    scores: dict[str, float] = {}
    leakage: dict[str, float] = {}

    for tz in candidates:
        try:
            localized = ts.dt.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
        except Exception:
            # fall back: drop ambiguous timestamps
            localized = ts.dt.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward").dropna()

        ct = localized.dt.tz_convert(CENTRAL_TZ)
        hours = ct.dt.hour
        score = _score_candidate(hours)
        scores[tz] = score
        leakage[tz] = score

    best = min(scores, key=scores.get)
    return TimezoneInferenceResult(
        raw_timezone=best,
        scores=scores,
        maintenance_leakage_pct=leakage,
        sample_rows=int(ts.size),
    )


def infer_raw_timezone_from_files(
    files: list[Path],
    max_files: int = 20,
    max_rows_per_file: int = 5000,
    candidates: list[str] | None = None,
) -> TimezoneInferenceResult:
    """Infer raw timezone by sampling timestamps from multiple files."""
    sample: list[str] = []
    for p in files[:max_files]:
        try:
            s = pd.read_csv(
                p,
                header=None,
                usecols=[0],
                names=["ts"],
                dtype={"ts": "string"},
                nrows=max_rows_per_file,
            )["ts"]
        except Exception:
            continue
        sample.extend([x for x in s.dropna().tolist() if isinstance(x, str)])
    return infer_raw_timezone(sample, candidates=candidates)

