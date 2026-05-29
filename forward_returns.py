from __future__ import annotations

import pandas as pd

from historical_reactions import BULLISH_EVENTS, detect_current_reaction_event, detect_event_masks
from historical_reactions import get_forward_sample, prepare_reaction_frame


MIN_SAMPLE_SIZE = 3


def safe_float(value: object) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def empty_profile(summary: str = "Forward return profile unavailable.") -> dict:
    return {
        "estimate_20d": None,
        "estimate_40d": None,
        "estimate_60d": None,
        "bear_case": None,
        "base_case": None,
        "bull_case": None,
        "confidence": "Low",
        "sample_size": 0,
        "active_event": None,
        "summary": summary,
        "drivers": ["Insufficient matching historical reactions."],
    }


def get_history_cache_key(ticker: str, history: pd.DataFrame, score_data: dict) -> tuple | None:
    if history is None or history.empty:
        return None
    close = safe_float(history["Close"].iloc[-1]) if "Close" in history.columns else None
    return (
        ticker.strip().upper(),
        len(history.index),
        str(history.index[-1]),
        round(close or 0, 4),
        score_data.get("active_reaction_event"),
        score_data.get("best_reaction_event"),
        score_data.get("signal_score"),
    )


def collect_event_samples(frame: pd.DataFrame, event_type: str) -> list[dict]:
    masks = detect_event_masks(frame)
    mask = masks.get(event_type)
    if mask is None:
        return []

    samples = []
    event_positions = [frame.index.get_loc(index_value) for index_value in frame.index[mask.fillna(False)]]
    for event_position in event_positions:
        sample = get_forward_sample(frame, event_position)
        if sample is not None:
            samples.append(sample)
    return samples


def choose_event(frame: pd.DataFrame, score_data: dict) -> str | None:
    active_event = score_data.get("active_reaction_event")
    if active_event in BULLISH_EVENTS:
        return active_event

    detected_event = detect_current_reaction_event(frame, current_setup=score_data.get("metrics", {}))
    if detected_event in BULLISH_EVENTS:
        return detected_event

    best_event = score_data.get("best_reaction_event")
    if best_event in BULLISH_EVENTS:
        return best_event

    historical_stats = score_data.get("historical_stats", {})
    bullish_stats = [stats for event, stats in historical_stats.items() if event in BULLISH_EVENTS]
    if bullish_stats:
        return max(bullish_stats, key=lambda item: item.get("historical_edge_score", 0)).get("event_type")
    return None


def get_sample_frame(frame: pd.DataFrame, event_type: str | None, score_data: dict) -> tuple[pd.DataFrame, str | None, bool]:
    if event_type:
        samples = collect_event_samples(frame, event_type)
        if len(samples) >= MIN_SAMPLE_SIZE:
            return pd.DataFrame(samples), event_type, False

    historical_stats = score_data.get("historical_stats", {})
    fallback_event = score_data.get("best_reaction_event")
    if not fallback_event and historical_stats:
        fallback_event = max(historical_stats.values(), key=lambda item: item.get("historical_edge_score", 0)).get("event_type")

    if fallback_event:
        samples = collect_event_samples(frame, fallback_event)
        if samples:
            return pd.DataFrame(samples), fallback_event, True

    return pd.DataFrame(), event_type, True


def get_setup_adjustment(score_data: dict) -> tuple[float, list[str]]:
    drivers = []
    adjustment = 0.0
    metrics = score_data.get("metrics", {})
    tags = set(score_data.get("raw_tags", score_data.get("tags", [])))

    if score_data.get("base_score", 0) >= 80:
        adjustment += 0.8
        drivers.append("Strong base quality adds a small upside adjustment.")
    if score_data.get("trigger_score", 0) >= 70:
        adjustment += 0.8
        drivers.append("Fresh trigger improves the forward profile.")
    if score_data.get("follow_through_score", 0) >= 70:
        adjustment += 0.7
        drivers.append("Follow-through score supports the setup.")
    if score_data.get("risk_score", 0) < 50:
        adjustment -= 1.0
        drivers.append("Risk score below 50 reduces the estimate.")
    if score_data.get("penalty_score", 0) > 20:
        adjustment -= 1.2
        drivers.append("Penalties reduce expected follow-through.")

    extension_pct = safe_float(metrics.get("extension_from_ema21_pct"))
    if extension_pct is not None and extension_pct > 8:
        adjustment -= min(2.0, (extension_pct - 8) * 0.35)
        drivers.append("Extension from EMA21 tempers upside.")
    if "failed-breakout-risk" in tags or "Failed Breakout Risk" in tags:
        adjustment -= 0.8
        drivers.append("Failed breakout risk lowers the profile.")

    return clamp(adjustment, -3.0, 3.0), drivers


def apply_runway_cap(value: float | None, runway_pct: float | None, drivers: list[str]) -> float | None:
    if value is None or runway_pct is None or runway_pct <= 0:
        return value
    cap = runway_pct + 2.0
    if value > cap:
        drivers.append("Upside capped near available runway to resistance.")
        return cap
    return value


def determine_confidence(sample_size: int, score_data: dict, fallback_used: bool) -> str:
    confidence_score = 0
    if sample_size >= 8:
        confidence_score += 2
    elif sample_size >= MIN_SAMPLE_SIZE:
        confidence_score += 1

    if score_data.get("historical_confidence") == "High":
        confidence_score += 1
    if fallback_used:
        confidence_score -= 1

    metrics = score_data.get("metrics", {})
    atr_pct = safe_float(metrics.get("atr_pct"))
    stop_distance_pct = safe_float(metrics.get("stop_distance_pct"))
    if atr_pct is not None and atr_pct > 7:
        confidence_score -= 1
    if stop_distance_pct is not None and stop_distance_pct > 12:
        confidence_score -= 1

    if confidence_score >= 3:
        return "High"
    if confidence_score >= 1:
        return "Medium"
    return "Low"


def build_summary(event_type: str | None, sample_size: int, fallback_used: bool) -> str:
    if not event_type:
        return "No matching historical event was available for a forward profile."
    prefix = "Fallback profile" if fallback_used else "Profile"
    return f"{prefix} based on {sample_size} historical {event_type.lower()} events."


def build_profile_from_samples(sample_frame: pd.DataFrame, event_type: str | None, fallback_used: bool, score_data: dict) -> dict:
    if sample_frame.empty:
        return empty_profile()

    adjustment, drivers = get_setup_adjustment(score_data)
    metrics = score_data.get("metrics", {})
    runway_pct = safe_float(metrics.get("runway_pct"))

    estimate_20d = safe_float(sample_frame["return_20d"].median()) + adjustment * 0.45
    estimate_40d = safe_float(sample_frame["return_40d"].median()) + adjustment
    estimate_60d = safe_float(sample_frame["return_60d"].median()) + adjustment * 1.2
    p25_40d = safe_float(sample_frame["return_40d"].quantile(0.25))
    p75_60d = safe_float(sample_frame["return_60d"].quantile(0.75))
    median_drawdown_40d = safe_float(sample_frame["max_adverse_40d"].median())

    bear_case = min(p25_40d, median_drawdown_40d) if p25_40d is not None and median_drawdown_40d is not None else p25_40d
    base_case = estimate_40d
    bull_case = apply_runway_cap(p75_60d + max(adjustment, 0), runway_pct, drivers) if p75_60d is not None else None
    estimate_60d = apply_runway_cap(estimate_60d, runway_pct, drivers)

    if not drivers:
        drivers.append("Historical median reactions drive the profile.")

    return {
        "estimate_20d": round(estimate_20d, 1) if estimate_20d is not None else None,
        "estimate_40d": round(estimate_40d, 1) if estimate_40d is not None else None,
        "estimate_60d": round(estimate_60d, 1) if estimate_60d is not None else None,
        "bear_case": round(bear_case, 1) if bear_case is not None else None,
        "base_case": round(base_case, 1) if base_case is not None else None,
        "bull_case": round(bull_case, 1) if bull_case is not None else None,
        "confidence": determine_confidence(len(sample_frame.index), score_data, fallback_used),
        "sample_size": len(sample_frame.index),
        "active_event": event_type,
        "summary": build_summary(event_type, len(sample_frame.index), fallback_used),
        "drivers": drivers[:5],
    }


def compute_forward_return_profile(ticker: str, history: pd.DataFrame | None, score_data: dict) -> dict:
    frame = prepare_reaction_frame(history)
    if frame.empty:
        return empty_profile("Insufficient price history for forward return profile.")

    event_type = choose_event(frame, score_data)
    sample_frame, selected_event, fallback_used = get_sample_frame(frame, event_type, score_data)
    if sample_frame.empty:
        return empty_profile("No usable historical event samples for forward return profile.")

    return build_profile_from_samples(sample_frame, selected_event, fallback_used, score_data)
