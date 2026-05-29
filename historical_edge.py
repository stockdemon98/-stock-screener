from __future__ import annotations

from functools import lru_cache

import pandas as pd


def _safe_float(value: object) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _prepare_history(history: pd.DataFrame | None) -> pd.DataFrame:
    if history is None or history.empty:
        return pd.DataFrame()
    required = ["High", "Low", "Close", "Volume"]
    if any(column not in history.columns for column in required):
        return pd.DataFrame()
    frame = history[required].copy()
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["High", "Low", "Close"])


def _atr_pct(frame: pd.DataFrame, length: int = 14) -> pd.Series:
    previous_close = frame["Close"].shift(1)
    true_range = pd.concat(
        [
            frame["High"] - frame["Low"],
            (frame["High"] - previous_close).abs(),
            (frame["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return (true_range.rolling(length).mean() / frame["Close"]) * 100


def _event_mask(frame: pd.DataFrame, event_type: str) -> pd.Series:
    close = frame["Close"]
    if event_type == "EMA8 reclaim":
        ma = close.ewm(span=8, adjust=False, min_periods=8).mean()
        return (close.shift(1) <= ma.shift(1)) & (close > ma)
    if event_type == "EMA21 reclaim":
        ma = close.ewm(span=21, adjust=False, min_periods=21).mean()
        return (close.shift(1) <= ma.shift(1)) & (close > ma)
    if event_type == "SMA50 reclaim":
        ma = close.rolling(50).mean()
        return (close.shift(1) <= ma.shift(1)) & (close > ma)
    if event_type == "SMA200 reclaim":
        ma = close.rolling(200).mean()
        return (close.shift(1) <= ma.shift(1)) & (close > ma)
    if event_type == "20-day high breakout":
        prior_high = frame["High"].shift(1).rolling(20).max()
        return close > prior_high
    if event_type == "50-day high breakout":
        prior_high = frame["High"].shift(1).rolling(50).max()
        return close > prior_high
    if event_type == "Compression breakout":
        atr_pct = _atr_pct(frame)
        prior_high = frame["High"].shift(1).rolling(20).max()
        return (atr_pct < atr_pct.shift(1).rolling(50).mean()) & (close > prior_high)
    return pd.Series(False, index=frame.index)


def _hit_before_drawdown(frame: pd.DataFrame, index_position: int, up_pct: float, down_pct: float, window: int) -> bool | None:
    entry = _safe_float(frame["Close"].iloc[index_position])
    if entry in (None, 0) or index_position + 1 >= len(frame.index):
        return None
    future = frame.iloc[index_position + 1:index_position + 1 + window]
    if future.empty:
        return None
    target = entry * (1 + up_pct)
    stop = entry * (1 - down_pct)
    for _, row in future.iterrows():
        high = _safe_float(row.get("High"))
        low = _safe_float(row.get("Low"))
        hit_target = high is not None and high >= target
        hit_stop = low is not None and low <= stop
        if hit_target and hit_stop:
            return False
        if hit_target:
            return True
        if hit_stop:
            return False
    return False


def calculate_historical_edge(history: pd.DataFrame | None, event_type: str | None) -> dict:
    frame = _prepare_history(history)
    event_type = event_type or "20-day high breakout"
    if frame.empty or len(frame.index) < 260:
        return {
            "available": False,
            "event_type": event_type,
            "hit_rate": None,
            "sample_size": 0,
            "confidence": "Low confidence",
            "score": 0.0,
            "summary": "Historical edge unavailable: not enough price history.",
        }

    mask = _event_mask(frame, event_type).fillna(False)
    positions = [frame.index.get_loc(index_value) for index_value in frame.index[mask]]
    positions = [position for position in positions if position + 40 < len(frame.index)]
    if not positions:
        return {
            "available": False,
            "event_type": event_type,
            "hit_rate": None,
            "sample_size": 0,
            "confidence": "Low confidence",
            "score": 0.0,
            "summary": "Historical edge unavailable: no matching historical events.",
        }

    records = []
    for position in positions:
        entry = _safe_float(frame["Close"].iloc[position])
        if entry in (None, 0):
            continue
        record = {"hit_5_before_3_30d": _hit_before_drawdown(frame, position, 0.05, 0.03, 30)}
        for days in (5, 10, 20, 40):
            future_close = _safe_float(frame["Close"].iloc[position + days])
            record[f"forward_return_{days}d"] = ((future_close / entry) - 1) * 100 if future_close is not None else None
        future_20 = frame.iloc[position + 1:position + 21]
        future_30 = frame.iloc[position + 1:position + 31]
        record["max_drawdown_20d"] = ((future_20["Low"].min() / entry) - 1) * 100 if not future_20.empty else None
        record["max_drawdown_30d"] = ((future_30["Low"].min() / entry) - 1) * 100 if not future_30.empty else None
        record["hit_10_before_5_40d"] = _hit_before_drawdown(frame, position, 0.10, 0.05, 40)
        records.append(record)

    sample_size = len(records)
    wins = [record["hit_5_before_3_30d"] for record in records if record["hit_5_before_3_30d"] is not None]
    hit_rate = (sum(1 for value in wins if value) / len(wins)) * 100 if wins else None
    confidence = "Higher confidence" if sample_size >= 30 else "Medium confidence" if sample_size >= 10 else "Low confidence"
    median_20d = pd.Series([record["forward_return_20d"] for record in records]).dropna().median()
    score = 0.0
    if sample_size >= 10 and hit_rate is not None:
        score = max(0.0, min(10.0, (hit_rate - 35.0) / 4.0))
    summary = (
        f"Historical sample: {sample_size} {event_type} events, {hit_rate:.0f}% hit rate."
        if hit_rate is not None and sample_size >= 10
        else "Historical edge unavailable: not enough historical events."
    )
    return {
        "available": sample_size >= 10 and hit_rate is not None,
        "event_type": event_type,
        "hit_rate": hit_rate if sample_size >= 10 else None,
        "sample_size": sample_size,
        "confidence": confidence,
        "median_forward_return_20d": None if pd.isna(median_20d) else float(median_20d),
        "score": round(score, 1),
        "summary": summary,
    }


@lru_cache(maxsize=512)
def calculate_historical_edge_cached(history_key: str, event_type: str) -> dict:
    return {
        "available": False,
        "event_type": event_type,
        "hit_rate": None,
        "sample_size": 0,
        "confidence": "Low confidence",
        "score": 0.0,
        "summary": "Historical Edge not calculated. Turn on Historical Edge to estimate past setup performance.",
    }
