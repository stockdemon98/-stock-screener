from __future__ import annotations

import pandas as pd


FORWARD_WINDOWS = (5, 10, 20, 40, 60)
DRAW_DOWN_WINDOWS = (20, 40, 60)
MA_RECLAIM_FORWARD_WINDOWS = (5, 10, 20, 60)
MA_RECLAIM_COLUMNS = {
    "EMA8": {"period": 8, "type": "ema"},
    "EMA21": {"period": 21, "type": "ema"},
    "SMA50": {"period": 50, "type": "sma"},
    "SMA100": {"period": 100, "type": "sma"},
    "SMA200": {"period": 200, "type": "sma"},
}
MIN_STRONG_SAMPLE = 3
REACTION_CACHE: dict[tuple, dict] = {}
MA_RECLAIM_CACHE: dict[tuple, dict] = {}
REACTION_CACHE_MAX_SIZE = 256
BULLISH_EVENTS = {
    "EMA21 Reclaim",
    "SMA50 Reclaim",
    "SMA200 Reclaim",
    "Golden Cross",
    "Pivot Breakout",
}


def safe_float(value: object) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def prepare_reaction_frame(history: pd.DataFrame | None) -> pd.DataFrame:
    if history is None or history.empty:
        return pd.DataFrame()

    required_columns = ["High", "Low", "Close"]
    if any(column not in history.columns for column in required_columns):
        return pd.DataFrame()

    frame = history[required_columns].dropna().copy()
    if len(frame.index) < 260:
        return pd.DataFrame()

    frame["EMA21"] = frame["Close"].ewm(span=21, adjust=False).mean()
    frame["SMA50"] = frame["Close"].rolling(50).mean()
    frame["SMA200"] = frame["Close"].rolling(200).mean()
    frame["Pivot60"] = frame["High"].rolling(60).max().shift(1)
    return frame


def get_history_cache_key(history: pd.DataFrame | None, current_setup: dict | None) -> tuple | None:
    if history is None or history.empty:
        return None

    close = safe_float(history["Close"].iloc[-1]) if "Close" in history.columns else None
    setup_event = current_setup.get("breakout_status") if current_setup else None
    try:
        first_index = str(history.index[0])
        last_index = str(history.index[-1])
    except IndexError:
        return None

    return (len(history.index), first_index, last_index, round(close or 0, 4), setup_event)


def cache_reaction_result(cache_key: tuple | None, result: dict) -> dict:
    if cache_key is None:
        return result
    if len(REACTION_CACHE) >= REACTION_CACHE_MAX_SIZE:
        REACTION_CACHE.clear()
    REACTION_CACHE[cache_key] = result
    return result


def get_ma_reclaim_cache_key(
    price_df: pd.DataFrame | None,
    ma_columns: tuple[str, ...],
    forward_windows: tuple[int, ...],
) -> tuple | None:
    if price_df is None or price_df.empty or "Close" not in price_df.columns:
        return None

    close = safe_float(price_df["Close"].dropna().iloc[-1]) if not price_df["Close"].dropna().empty else None
    try:
        first_index = str(price_df.index[0])
        last_index = str(price_df.index[-1])
    except IndexError:
        return None

    return (
        "ma-reclaim",
        len(price_df.index),
        first_index,
        last_index,
        round(close or 0, 4),
        ma_columns,
        forward_windows,
    )


def cache_ma_reclaim_result(cache_key: tuple | None, result: dict) -> dict:
    if cache_key is None:
        return result
    if len(MA_RECLAIM_CACHE) >= REACTION_CACHE_MAX_SIZE:
        MA_RECLAIM_CACHE.clear()
    MA_RECLAIM_CACHE[cache_key] = result
    return result


def get_reliability_label(event_count: int) -> str:
    if event_count < 8:
        return "Low sample"
    if event_count < 20:
        return "Medium sample"
    return "Strong sample"


def prepare_ma_reclaim_frame(price_df: pd.DataFrame | None, ma_columns: tuple[str, ...]) -> pd.DataFrame:
    if price_df is None or price_df.empty or "Close" not in price_df.columns:
        return pd.DataFrame()

    frame = price_df.copy()
    frame["Close"] = pd.to_numeric(frame["Close"], errors="coerce")
    if "High" in frame.columns:
        frame["High"] = pd.to_numeric(frame["High"], errors="coerce")
    else:
        frame["High"] = frame["Close"]
    if "Low" in frame.columns:
        frame["Low"] = pd.to_numeric(frame["Low"], errors="coerce")
    else:
        frame["Low"] = frame["Close"]

    frame = frame.dropna(subset=["Close"]).copy()
    if frame.empty:
        return pd.DataFrame()

    for ma_name in ma_columns:
        config = MA_RECLAIM_COLUMNS.get(ma_name)
        if not config:
            continue
        if ma_name in frame.columns:
            frame[ma_name] = pd.to_numeric(frame[ma_name], errors="coerce")
            continue
        if config["type"] == "ema":
            frame[ma_name] = frame["Close"].ewm(
                span=config["period"],
                adjust=False,
                min_periods=config["period"],
            ).mean()
        else:
            frame[ma_name] = frame["Close"].rolling(config["period"]).mean()

    return frame


def _format_event_date(index_value: object) -> str:
    if hasattr(index_value, "date"):
        return index_value.date().isoformat()
    return str(index_value)


def build_ma_reclaim_stats_for_signal(
    frame: pd.DataFrame,
    ma_name: str,
    forward_windows: tuple[int, ...],
) -> dict:
    empty_result = {
        "signal": f"{ma_name} Reclaim",
        "ma": ma_name,
        "event_count": 0,
        "reliability": get_reliability_label(0),
        "most_recent_reclaim_date": None,
        "active_recent_reclaim": None,
    }
    if frame.empty or ma_name not in frame.columns or len(frame.index) < 2:
        return empty_result

    close = frame["Close"]
    ma = frame[ma_name]
    event_mask = (close.shift(1) < ma.shift(1)) & (close > ma)
    event_dates = list(frame.index[event_mask.fillna(False)])
    if not event_dates:
        return empty_result

    max_window = max(max(forward_windows), 60)
    event_positions = [frame.index.get_loc(index_value) for index_value in event_dates]
    most_recent_position = event_positions[-1]
    most_recent_date = event_dates[-1]
    completed_samples = []

    for event_position, event_date in zip(event_positions, event_dates):
        if event_position + max_window >= len(frame.index):
            continue

        event_close = safe_float(close.iloc[event_position])
        if event_close in (None, 0):
            continue

        sample = {"event_date": _format_event_date(event_date)}
        for window in forward_windows:
            future_close = safe_float(close.iloc[event_position + window])
            if future_close is None:
                sample[f"return_{window}d"] = None
            else:
                sample[f"return_{window}d"] = (future_close / event_close) - 1

        for window in (20, 60):
            future_highs = frame["High"].iloc[event_position + 1:event_position + window + 1].dropna()
            future_lows = frame["Low"].iloc[event_position + 1:event_position + window + 1].dropna()
            sample[f"max_runup_{window}d"] = (
                (safe_float(future_highs.max()) / event_close) - 1 if not future_highs.empty else None
            )
            sample[f"max_drawdown_{window}d"] = (
                (safe_float(future_lows.min()) / event_close) - 1 if not future_lows.empty else None
            )
        completed_samples.append(sample)

    result = {
        **empty_result,
        "most_recent_reclaim_date": _format_event_date(most_recent_date),
        "total_reclaim_events": len(event_positions),
    }
    days_since_recent = len(frame.index) - most_recent_position - 1
    if days_since_recent <= max_window:
        result["active_recent_reclaim"] = {
            "signal": f"{ma_name} Reclaim",
            "date": _format_event_date(most_recent_date),
            "days_since": int(days_since_recent),
            "event_close": safe_float(close.iloc[most_recent_position]),
            "ma_value": safe_float(ma.iloc[most_recent_position]),
            "full_forward_data_available": most_recent_position + max_window < len(frame.index),
        }

    if not completed_samples:
        return result

    sample_frame = pd.DataFrame(completed_samples)
    event_count = len(sample_frame.index)
    result["event_count"] = event_count
    result["reliability"] = get_reliability_label(event_count)

    for window in forward_windows:
        column = f"return_{window}d"
        values = pd.to_numeric(sample_frame.get(column), errors="coerce").dropna()
        result[f"median_return_{window}d"] = safe_float(values.median()) if not values.empty else None
        result[f"average_return_{window}d"] = safe_float(values.mean()) if not values.empty else None
        result[f"win_rate_{window}d"] = safe_float((values > 0).mean()) if not values.empty else None

    for window in (20, 60):
        runup_values = pd.to_numeric(sample_frame.get(f"max_runup_{window}d"), errors="coerce").dropna()
        drawdown_values = pd.to_numeric(sample_frame.get(f"max_drawdown_{window}d"), errors="coerce").dropna()
        result[f"median_max_runup_{window}d"] = safe_float(runup_values.median()) if not runup_values.empty else None
        result[f"median_max_drawdown_{window}d"] = safe_float(drawdown_values.median()) if not drawdown_values.empty else None

    return result


def calculate_ma_reclaim_stats(
    price_df: pd.DataFrame | None,
    ma_columns: list[str] | tuple[str, ...] | None = None,
    forward_windows: list[int] | tuple[int, ...] | None = None,
) -> dict:
    ma_columns = tuple(ma_columns or tuple(MA_RECLAIM_COLUMNS.keys()))
    forward_windows = tuple(int(window) for window in (forward_windows or MA_RECLAIM_FORWARD_WINDOWS))
    cache_key = get_ma_reclaim_cache_key(price_df, ma_columns, forward_windows)
    if cache_key in MA_RECLAIM_CACHE:
        return MA_RECLAIM_CACHE[cache_key]

    frame = prepare_ma_reclaim_frame(price_df, ma_columns)
    if frame.empty:
        return cache_ma_reclaim_result(cache_key, {
            "stats": {},
            "rows": [],
            "active_recent_reclaims": [],
            "summary": "Insufficient price history for moving-average reclaim analysis.",
        })

    stats = {
        ma_name: build_ma_reclaim_stats_for_signal(frame, ma_name, forward_windows)
        for ma_name in ma_columns
    }
    rows = [stats[ma_name] for ma_name in ma_columns if ma_name in stats]
    active_recent_reclaims = [
        row["active_recent_reclaim"]
        for row in rows
        if row.get("active_recent_reclaim")
    ]
    return cache_ma_reclaim_result(cache_key, {
        "stats": stats,
        "rows": rows,
        "active_recent_reclaims": active_recent_reclaims,
        "summary": "Historical returns after daily moving-average reclaims.",
    })


def detect_event_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    close = frame["Close"]
    previous_close = close.shift(1)
    previous_pivot = frame["Pivot60"].shift(1)

    return {
        "EMA21 Reclaim": (previous_close <= frame["EMA21"].shift(1)) & (close > frame["EMA21"]),
        "EMA21 Break": (previous_close >= frame["EMA21"].shift(1)) & (close < frame["EMA21"]),
        "SMA50 Reclaim": (previous_close <= frame["SMA50"].shift(1)) & (close > frame["SMA50"]),
        "SMA50 Break": (previous_close >= frame["SMA50"].shift(1)) & (close < frame["SMA50"]),
        "SMA200 Reclaim": (previous_close <= frame["SMA200"].shift(1)) & (close > frame["SMA200"]),
        "SMA200 Break": (previous_close >= frame["SMA200"].shift(1)) & (close < frame["SMA200"]),
        "Golden Cross": (frame["SMA50"].shift(1) <= frame["SMA200"].shift(1)) & (frame["SMA50"] > frame["SMA200"]),
        "Death Cross": (frame["SMA50"].shift(1) >= frame["SMA200"].shift(1)) & (frame["SMA50"] < frame["SMA200"]),
        "Pivot Breakout": (previous_close <= previous_pivot) & (close > frame["Pivot60"]),
    }


def get_forward_sample(frame: pd.DataFrame, event_index: int) -> dict | None:
    if event_index + max(FORWARD_WINDOWS) >= len(frame.index):
        return None

    entry_close = safe_float(frame["Close"].iloc[event_index])
    if entry_close in (None, 0):
        return None

    sample = {}
    for window in FORWARD_WINDOWS:
        future_close = safe_float(frame["Close"].iloc[event_index + window])
        if future_close is None:
            return None
        sample[f"return_{window}d"] = ((future_close - entry_close) / entry_close) * 100

    for window in DRAW_DOWN_WINDOWS:
        future_lows = frame["Low"].iloc[event_index + 1:event_index + window + 1]
        if future_lows.empty:
            return None
        sample[f"max_adverse_{window}d"] = ((safe_float(future_lows.min()) - entry_close) / entry_close) * 100

    return sample


def score_event_stats(
    event_count: int,
    win_rate_40d: float | None,
    median_return_40d: float | None,
    win_rate_60d: float | None,
    median_return_60d: float | None,
    median_max_drawdown_40d: float | None,
) -> float:
    if event_count == 0:
        return 50.0

    sample_weight = clamp(event_count / 8, 0.35, 1.0)
    win_component = ((win_rate_40d or 0) * 0.6) + ((win_rate_60d or 0) * 0.4)
    return_component = clamp((((median_return_40d or 0) * 0.6) + ((median_return_60d or 0) * 0.4) + 8) / 18 * 100, 0, 100)
    drawdown_component = clamp(((median_max_drawdown_40d or -12) + 12) / 12 * 100, 0, 100)
    raw_score = (win_component * 0.45) + (return_component * 0.40) + (drawdown_component * 0.15)
    return clamp((raw_score * sample_weight) + (50 * (1 - sample_weight)), 0, 100)


def summarize_event(event_type: str, event_count: int, win_rate_40d: float | None, median_return_40d: float | None) -> str:
    if event_count == 0:
        return f"No usable historical {event_type.lower()} samples."
    return (
        f"{event_type}: {event_count} samples, "
        f"40D win rate {format_percent(win_rate_40d)}, "
        f"median 40D return {format_signed_percent(median_return_40d)}."
    )


def format_percent(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.0f}%"


def format_signed_percent(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:+.1f}%"


def build_event_stats(event_type: str, samples: list[dict]) -> dict:
    if not samples:
        return {
            "event_type": event_type,
            "event_count": 0,
            "win_rate_20d": None,
            "median_return_20d": None,
            "win_rate_40d": None,
            "median_return_40d": None,
            "win_rate_60d": None,
            "median_return_60d": None,
            "median_max_drawdown_40d": None,
            "median_max_drawdown_20d": None,
            "median_max_drawdown_60d": None,
            "failure_rate": None,
            "historical_edge_score": 50.0,
            "summary": summarize_event(event_type, 0, None, None),
        }

    sample_frame = pd.DataFrame(samples)
    event_count = len(sample_frame.index)
    win_rate_20d = float((sample_frame["return_20d"] > 0).mean() * 100)
    win_rate_40d = float((sample_frame["return_40d"] > 0).mean() * 100)
    win_rate_60d = float((sample_frame["return_60d"] > 0).mean() * 100)
    median_return_20d = safe_float(sample_frame["return_20d"].median())
    median_return_40d = safe_float(sample_frame["return_40d"].median())
    median_return_60d = safe_float(sample_frame["return_60d"].median())
    median_max_drawdown_20d = safe_float(sample_frame["max_adverse_20d"].median())
    median_max_drawdown_40d = safe_float(sample_frame["max_adverse_40d"].median())
    median_max_drawdown_60d = safe_float(sample_frame["max_adverse_60d"].median())
    failure_rate = float(((sample_frame["return_20d"] < 0) | (sample_frame["max_adverse_20d"] <= -8)).mean() * 100)
    historical_edge_score = score_event_stats(
        event_count=event_count,
        win_rate_40d=win_rate_40d,
        median_return_40d=median_return_40d,
        win_rate_60d=win_rate_60d,
        median_return_60d=median_return_60d,
        median_max_drawdown_40d=median_max_drawdown_40d,
    )

    return {
        "event_type": event_type,
        "event_count": event_count,
        "win_rate_20d": win_rate_20d,
        "median_return_20d": median_return_20d,
        "win_rate_40d": win_rate_40d,
        "median_return_40d": median_return_40d,
        "win_rate_60d": win_rate_60d,
        "median_return_60d": median_return_60d,
        "median_max_drawdown_20d": median_max_drawdown_20d,
        "median_max_drawdown_40d": median_max_drawdown_40d,
        "median_max_drawdown_60d": median_max_drawdown_60d,
        "failure_rate": failure_rate,
        "historical_edge_score": historical_edge_score,
        "summary": summarize_event(event_type, event_count, win_rate_40d, median_return_40d),
    }


def analyze_historical_events(history: pd.DataFrame | None) -> dict[str, dict]:
    frame = prepare_reaction_frame(history)
    if frame.empty:
        return {}

    event_masks = detect_event_masks(frame)
    event_stats = {}
    for event_type, mask in event_masks.items():
        samples = []
        event_positions = [frame.index.get_loc(index_value) for index_value in frame.index[mask.fillna(False)]]
        for event_position in event_positions:
            sample = get_forward_sample(frame, event_position)
            if sample is not None:
                samples.append(sample)
        event_stats[event_type] = build_event_stats(event_type, samples)

    return event_stats


def detect_current_reaction_event(frame: pd.DataFrame, current_setup: dict | None = None) -> str | None:
    if frame.empty or len(frame.index) < 2:
        return None

    masks = detect_event_masks(frame)
    for event_type in (
        "Pivot Breakout",
        "EMA21 Reclaim",
        "SMA50 Reclaim",
        "SMA200 Reclaim",
        "Golden Cross",
        "EMA21 Break",
        "SMA50 Break",
        "SMA200 Break",
        "Death Cross",
    ):
        if bool(masks[event_type].fillna(False).iloc[-1]):
            return event_type

    close = safe_float(frame["Close"].iloc[-1])
    if close in (None, 0):
        return None

    pivot = safe_float(frame["Pivot60"].iloc[-1])
    if pivot not in (None, 0) and -3 <= ((close - pivot) / pivot) * 100 <= 1:
        return "Pivot Breakout"

    for column, event_type in (("EMA21", "EMA21 Reclaim"), ("SMA50", "SMA50 Reclaim"), ("SMA200", "SMA200 Reclaim")):
        value = safe_float(frame[column].iloc[-1])
        if value not in (None, 0) and -2 <= ((close - value) / value) * 100 <= 2:
            return event_type

    if current_setup:
        status = current_setup.get("breakout_status")
        if status in {"Near Pivot", "Breaking Out"}:
            return "Pivot Breakout"

    return None


def get_confidence(event_count: int, score: float) -> str:
    if event_count >= 8 and score >= 65:
        return "High"
    if event_count >= MIN_STRONG_SAMPLE:
        return "Medium"
    return "Low"


def compute_historical_reaction_score(history: pd.DataFrame | None, current_setup: dict | None = None) -> dict:
    cache_key = get_history_cache_key(history, current_setup)
    if cache_key in REACTION_CACHE:
        return REACTION_CACHE[cache_key]

    frame = prepare_reaction_frame(history)
    if frame.empty:
        return cache_reaction_result(cache_key, {
            "historical_score": 50.0,
            "confidence": "Low",
            "active_event": None,
            "best_event": None,
            "stats": {},
            "summary": "Insufficient history for historical reaction analysis.",
        })

    stats = analyze_historical_events(frame)
    active_event = detect_current_reaction_event(frame, current_setup=current_setup)
    bullish_stats = [item for item in stats.values() if item.get("event_type") in BULLISH_EVENTS]
    best_stat = max(bullish_stats or stats.values(), key=lambda item: item.get("historical_edge_score", 0), default=None)
    active_stat = stats.get(active_event) if active_event else None
    selected_stat = active_stat or best_stat

    if not selected_stat:
        return cache_reaction_result(cache_key, {
            "historical_score": 50.0,
            "confidence": "Low",
            "active_event": active_event,
            "best_event": None,
            "stats": stats,
            "summary": "No usable historical reaction samples.",
        })

    event_count = selected_stat.get("event_count", 0)
    score = selected_stat.get("historical_edge_score", 50.0)
    if event_count < MIN_STRONG_SAMPLE:
        score = (score * 0.45) + 27.5

    best_event = best_stat.get("event_type") if best_stat else None
    summary_prefix = "Active setup" if active_stat else "Best historical analog"
    return cache_reaction_result(cache_key, {
        "historical_score": round(clamp(score, 0, 100), 1),
        "confidence": get_confidence(event_count, score),
        "active_event": active_event,
        "best_event": best_event,
        "stats": stats,
        "summary": f"{summary_prefix}: {selected_stat.get('summary')}",
    })
