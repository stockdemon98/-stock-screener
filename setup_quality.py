from __future__ import annotations

import pandas as pd

from historical_edge import calculate_historical_edge


def safe_float(value: object) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def score_range(value: float | None, low: float, high: float) -> float:
    if value is None or high <= low:
        return 0.0
    return clamp(((value - low) / (high - low)) * 100.0)


def score_inverse(value: float | None, low: float, high: float) -> float:
    if value is None or high <= low:
        return 0.0
    return clamp(((high - value) / (high - low)) * 100.0)


def prepare_indicators(history: pd.DataFrame | None) -> pd.DataFrame:
    if history is None or history.empty:
        return pd.DataFrame()
    required = ["Open", "High", "Low", "Close", "Volume"]
    if any(column not in history.columns for column in required):
        return pd.DataFrame()
    frame = history[required].copy()
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["High", "Low", "Close"])
    if frame.empty:
        return frame
    frame["EMA8"] = frame["Close"].ewm(span=8, adjust=False, min_periods=8).mean()
    frame["EMA21"] = frame["Close"].ewm(span=21, adjust=False, min_periods=21).mean()
    frame["SMA50"] = frame["Close"].rolling(50).mean()
    frame["SMA200"] = frame["Close"].rolling(200).mean()
    previous_close = frame["Close"].shift(1)
    true_range = pd.concat(
        [
            frame["High"] - frame["Low"],
            (frame["High"] - previous_close).abs(),
            (frame["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    frame["ATR14"] = true_range.rolling(14).mean()
    frame["ATR_PCT"] = (frame["ATR14"] / frame["Close"]) * 100
    frame["REL_VOLUME"] = frame["Volume"] / frame["Volume"].rolling(20).mean()
    day_range = frame["High"] - frame["Low"]
    frame["CLOSE_LOCATION"] = ((frame["Close"] - frame["Low"]) / day_range).where(day_range > 0)
    frame["PRIOR_HIGH_20D"] = frame["High"].shift(1).rolling(20).max()
    frame["PRIOR_HIGH_50D"] = frame["High"].shift(1).rolling(50).max()
    return frame


def pct_distance(price: object, level: object) -> float | None:
    price_value = safe_float(price)
    level_value = safe_float(level)
    if price_value is None or level_value in (None, 0):
        return None
    return ((price_value / level_value) - 1) * 100


def _rising(series: pd.Series, lookback: int = 10) -> bool:
    values = series.dropna()
    if len(values.index) <= lookback:
        return False
    latest = safe_float(values.iloc[-1])
    prior = safe_float(values.iloc[-lookback - 1])
    return latest is not None and prior is not None and latest > prior


def calculate_pivot_and_stop(indicators: pd.DataFrame) -> dict:
    if indicators.empty:
        return {"suggested_pivot": None, "suggested_stop": None, "risk_pct": None, "reward_risk": None}
    latest = indicators.iloc[-1]
    close = safe_float(latest.get("Close"))
    pivot20 = safe_float(latest.get("PRIOR_HIGH_20D"))
    pivot50 = safe_float(latest.get("PRIOR_HIGH_50D"))
    pivot = pivot20 if pivot20 is not None else pivot50
    if pivot20 is not None and pivot50 is not None:
        pivot = min(pivot20, pivot50, key=lambda level: abs((close or level) - level))
    recent_swing_low = safe_float(indicators["Low"].tail(15).min()) if len(indicators.index) >= 15 else None
    ema21 = safe_float(latest.get("EMA21"))
    atr14 = safe_float(latest.get("ATR14"))
    stop_candidates = []
    if pivot is not None:
        stop_candidates.append(pivot * 0.97)
    if recent_swing_low is not None:
        stop_candidates.append(recent_swing_low * 0.99)
    if ema21 is not None:
        stop_candidates.append(ema21 * 0.985)
    if close is not None and atr14 is not None:
        stop_candidates.append(close - atr14)
    suggested_stop = max((value for value in stop_candidates if close is not None and value < close), default=None)
    risk_pct = ((close - suggested_stop) / close) * 100 if close not in (None, 0) and suggested_stop is not None else None
    target = close * 1.10 if close is not None else None
    if pivot is not None and close is not None and pivot > close:
        target = pivot
    elif risk_pct is not None and close is not None:
        target = close * (1 + (2 * risk_pct / 100))
    reward_risk = ((target - close) / close * 100) / risk_pct if target is not None and close not in (None, 0) and risk_pct not in (None, 0) else None
    return {
        "suggested_pivot": pivot,
        "suggested_stop": suggested_stop,
        "risk_pct": risk_pct,
        "reward_risk": reward_risk,
    }


def classify_setup(indicators: pd.DataFrame, metrics: dict | None = None) -> str:
    metrics = metrics or {}
    if indicators.empty or len(indicators.index) < 60:
        return "No Clean Setup"
    latest = indicators.iloc[-1]
    close = safe_float(latest.get("Close"))
    ema8 = safe_float(latest.get("EMA8"))
    ema21 = safe_float(latest.get("EMA21"))
    sma50 = safe_float(latest.get("SMA50"))
    sma200 = safe_float(latest.get("SMA200"))
    prior_high_20 = safe_float(latest.get("PRIOR_HIGH_20D"))
    prior_high_50 = safe_float(latest.get("PRIOR_HIGH_50D"))
    rel_volume = safe_float(latest.get("REL_VOLUME"))
    close_location = safe_float(latest.get("CLOSE_LOCATION"))
    atr_pct = safe_float(latest.get("ATR_PCT"))
    atr_avg = safe_float(indicators["ATR_PCT"].tail(50).mean()) if len(indicators.index) >= 50 else None
    pivot = metrics.get("suggested_pivot")
    distance_to_pivot = pct_distance(close, pivot)
    extension_ema21 = pct_distance(close, ema21)
    near_52w_high = metrics.get("near_52w_high")

    broke_20 = close is not None and prior_high_20 is not None and close > prior_high_20
    broke_50 = close is not None and prior_high_50 is not None and close > prior_high_50
    if (broke_20 or broke_50) and (rel_volume is None or rel_volume >= 1.2) and (close_location is None or close_location >= 0.6):
        return "Fresh Breakout"

    for label, column in (("EMA8", "EMA8"), ("EMA21", "EMA21"), ("SMA50", "SMA50"), ("SMA200", "SMA200")):
        if column not in indicators.columns or len(indicators.index) < 2:
            continue
        previous_close = safe_float(indicators["Close"].iloc[-2])
        previous_ma = safe_float(indicators[column].iloc[-2])
        current_ma = safe_float(latest.get(column))
        if previous_close is not None and previous_ma is not None and close is not None and current_ma is not None:
            if previous_close <= previous_ma and close > current_ma:
                return f"EMA Reclaim ({label})"

    for days in (20, 50):
        prior_high = indicators["High"].shift(1).rolling(days).max()
        breakout_mask = indicators["Close"] > prior_high
        positions = [indicators.index.get_loc(index_value) for index_value in indicators.index[breakout_mask.fillna(False)]]
        if positions:
            last_position = positions[-1]
            days_since = len(indicators.index) - last_position - 1
            breakout_level = safe_float(prior_high.iloc[last_position])
            distance_to_breakout = pct_distance(close, breakout_level)
            breakout_volume = safe_float(indicators["Volume"].iloc[last_position])
            recent_volume = safe_float(indicators["Volume"].tail(max(days_since, 1)).mean())
            if 3 <= days_since <= 15 and distance_to_breakout is not None and -3 <= distance_to_breakout <= 5:
                if recent_volume is None or breakout_volume is None or recent_volume < breakout_volume:
                    return "Breakout Retest"

    if (
        close is not None
        and sma50 is not None
        and sma200 is not None
        and close > sma50 > sma200
        and _rising(indicators["SMA200"], 20)
        and near_52w_high
    ):
        return "Stage-2 Leader"

    compression = atr_pct is not None and atr_avg is not None and atr_pct < atr_avg
    volume_dry = safe_float(indicators["Volume"].tail(5).mean()) is not None and safe_float(indicators["Volume"].tail(5).mean()) < safe_float(indicators["Volume"].tail(20).mean())
    if compression and volume_dry and distance_to_pivot is not None and -5 <= distance_to_pivot <= 1:
        return "Compression Squeeze"

    if close is not None and (sma50 is None or close >= sma50 * 0.98) and distance_to_pivot is not None and -5 <= distance_to_pivot <= 0 and (extension_ema21 is None or extension_ema21 <= 8):
        return "Pre-Breakout Base"

    return "No Clean Setup"


def _historical_event_for_setup(setup_type: str) -> str:
    if "EMA Reclaim" in setup_type:
        if "EMA8" in setup_type:
            return "EMA8 reclaim"
        if "EMA21" in setup_type:
            return "EMA21 reclaim"
        if "SMA50" in setup_type:
            return "SMA50 reclaim"
        if "SMA200" in setup_type:
            return "SMA200 reclaim"
    if setup_type in {"Fresh Breakout", "Pre-Breakout Base", "Breakout Retest", "Stage-2 Leader"}:
        return "50-day high breakout"
    if setup_type == "Compression Squeeze":
        return "Compression breakout"
    return "20-day high breakout"


def calculate_setup_quality(
    history: pd.DataFrame | None,
    market_regime: dict | None = None,
    addv_20: float | None = None,
    calculate_edge: bool = False,
) -> dict:
    indicators = prepare_indicators(history)
    if indicators.empty:
        return {
            "setup_type": "No Clean Setup",
            "a_plus_status": "N/A",
            "signal_score": 0.0,
            "historical_edge": "Historical edge unavailable: not enough price history.",
        }
    latest = indicators.iloc[-1]
    close = safe_float(latest.get("Close"))
    ema8 = safe_float(latest.get("EMA8"))
    ema21 = safe_float(latest.get("EMA21"))
    sma50 = safe_float(latest.get("SMA50"))
    sma200 = safe_float(latest.get("SMA200"))
    rel_volume = safe_float(latest.get("REL_VOLUME"))
    atr_pct = safe_float(latest.get("ATR_PCT"))
    close_location = safe_float(latest.get("CLOSE_LOCATION"))
    pivot_risk = calculate_pivot_and_stop(indicators)
    high_52w = safe_float(indicators["High"].tail(252).max()) if len(indicators.index) >= 100 else None
    near_52w_high = close is not None and high_52w not in (None, 0) and close >= high_52w * 0.90
    metrics = {**pivot_risk, "near_52w_high": near_52w_high}
    setup_type = classify_setup(indicators, metrics)

    market_regime = market_regime or {"market_regime": "N/A", "market_regime_score": 8.0, "market_regime_penalty": 0.0}
    market_score = safe_float(market_regime.get("market_regime_score")) or 0.0
    trend_points = 0.0
    trend_points += 3 if close is not None and ema21 is not None and close > ema21 else 0
    trend_points += 3 if close is not None and sma50 is not None and close > sma50 else 0
    trend_points += 3 if close is not None and sma200 is not None and close > sma200 else 0
    trend_points += 2 if ema8 is not None and ema21 is not None and ema8 > ema21 else 0
    trend_points += 2 if _rising(indicators["SMA50"], 10) else 0
    trend_points += 2 if _rising(indicators["SMA200"], 20) else 0

    distance_to_pivot = pct_distance(close, pivot_risk.get("suggested_pivot"))
    base_width = None
    if len(indicators.index) >= 20:
        recent_high = safe_float(indicators["High"].tail(20).max())
        recent_low = safe_float(indicators["Low"].tail(20).min())
        base_width = ((recent_high - recent_low) / recent_high) * 100 if recent_high not in (None, 0) and recent_low is not None else None
    higher_lows = len(indicators.index) >= 30 and safe_float(indicators["Low"].tail(10).min()) > safe_float(indicators["Low"].iloc[-30:-10].min())
    base_quality = (
        6 * (score_inverse(base_width, 8, 35) / 100)
        + 6 * (score_range(1 if distance_to_pivot is not None and -5 <= distance_to_pivot <= 1 else 0, 0, 1) / 100)
        + 4 * (1 if higher_lows else 0)
        + 4 * (1 if near_52w_high else 0)
    )

    atr_avg = safe_float(indicators["ATR_PCT"].tail(50).mean()) if len(indicators.index) >= 50 else None
    recent_range = safe_float(indicators["High"].tail(10).max() - indicators["Low"].tail(10).min()) if len(indicators.index) >= 10 else None
    prior_range = safe_float(indicators["High"].iloc[-20:-10].max() - indicators["Low"].iloc[-20:-10].min()) if len(indicators.index) >= 20 else None
    volume_dry = safe_float(indicators["Volume"].tail(5).mean()) < safe_float(indicators["Volume"].tail(20).mean()) if len(indicators.index) >= 20 else False
    compression_quality = 0.0
    compression_quality += 6 if atr_pct is not None and atr_avg is not None and atr_pct < atr_avg else 0
    compression_quality += 5 if recent_range is not None and prior_range not in (None, 0) and recent_range < prior_range else 0
    compression_quality += 4 if volume_dry else 0

    trigger_quality = 0.0
    prior_high_20 = safe_float(latest.get("PRIOR_HIGH_20D"))
    prior_high_50 = safe_float(latest.get("PRIOR_HIGH_50D"))
    trigger_quality += 5 if close is not None and prior_high_20 is not None and close > prior_high_20 else 0
    trigger_quality += 4 if close is not None and prior_high_50 is not None and close > prior_high_50 else 0
    trigger_quality += 3 if close_location is not None and close_location >= 0.6 else 0
    trigger_quality += 3 if rel_volume is not None and rel_volume >= 1.2 else 0

    risk_pct = safe_float(pivot_risk.get("risk_pct"))
    reward_risk = safe_float(pivot_risk.get("reward_risk"))
    risk_reward_quality = 0.0
    risk_reward_quality += 5 * (score_inverse(risk_pct, 3, 12) / 100)
    risk_reward_quality += 5 * (score_range(reward_risk, 0.8, 2.0) / 100)

    event_type = _historical_event_for_setup(setup_type)
    historical = calculate_historical_edge(history, event_type) if calculate_edge else {
        "available": False,
        "event_type": event_type,
        "hit_rate": None,
        "sample_size": 0,
        "confidence": "N/A",
        "score": 0.0,
        "summary": "Historical Edge not calculated. Turn on Historical Edge to estimate past setup performance.",
    }
    historical_score = safe_float(historical.get("score")) or 0.0

    penalties = 0.0
    extension_ema21 = pct_distance(close, ema21)
    if extension_ema21 is not None and extension_ema21 > 5:
        penalties += min(20.0, (extension_ema21 - 5) * 2.5)
    penalties += safe_float(market_regime.get("market_regime_penalty")) or 0.0
    if rel_volume is not None and rel_volume < 0.8:
        penalties += min(20.0, (0.8 - rel_volume) * 25.0)
    if close is not None and sma200 is not None and close < sma200:
        penalties += 25.0
    if addv_20 is not None and addv_20 < 20_000_000:
        penalties += 20.0
    if atr_pct is not None and atr_pct > 7:
        penalties += min(15.0, (atr_pct - 7) * 3.0)
    if close_location is not None and close_location < 0.4:
        penalties += min(10.0, (0.4 - close_location) * 25.0)

    score = clamp(market_score + trend_points + base_quality + compression_quality + trigger_quality + risk_reward_quality + historical_score - penalties)
    a_plus_pass = (
        score >= 80
        and market_regime.get("market_regime") != "Risk-Off"
        and trend_points >= 12
        and setup_type != "No Clean Setup"
        and (extension_ema21 is None or extension_ema21 <= 8)
        and (atr_pct is None or atr_pct <= 7)
    )
    explanation_parts = [
        f"{setup_type}:",
        f"market regime {market_regime.get('market_regime', 'N/A')}",
        f"relative volume {rel_volume:.1f}x" if rel_volume is not None else "relative volume N/A",
        f"close strength {close_location:.2f}" if close_location is not None else "close strength N/A",
        f"risk is {risk_pct:.1f}% to suggested stop" if risk_pct is not None else "risk N/A",
    ]
    if historical.get("available") and historical.get("hit_rate") is not None:
        explanation_parts.append(f"historical sample: {historical.get('sample_size')} similar events, {historical.get('hit_rate'):.0f}% hit rate")
    else:
        explanation_parts.append(historical.get("summary", "Historical edge unavailable."))

    return {
        "setup_type": setup_type,
        "a_plus_status": "A+ Candidate" if a_plus_pass else "Does Not Qualify",
        "a_plus_qualified": a_plus_pass,
        "setup_signal_score": round(score, 1),
        "market_regime": market_regime.get("market_regime", "N/A"),
        "trend_quality": round(trend_points, 1),
        "base_quality": round(base_quality, 1),
        "compression_quality": round(compression_quality, 1),
        "trigger_quality": round(trigger_quality, 1),
        "risk_reward_quality": round(risk_reward_quality, 1),
        "historical_edge": historical.get("summary"),
        "historical_hit_rate": historical.get("hit_rate"),
        "historical_sample_size": historical.get("sample_size"),
        "historical_confidence": historical.get("confidence"),
        "historical_event_type": historical.get("event_type"),
        "suggested_pivot": pivot_risk.get("suggested_pivot"),
        "suggested_stop": pivot_risk.get("suggested_stop"),
        "risk_pct": risk_pct,
        "reward_risk": reward_risk,
        "relative_volume": rel_volume,
        "atr_pct": atr_pct,
        "close_strength": close_location,
        "a_plus_explanation": ", ".join(explanation_parts) + ".",
    }
