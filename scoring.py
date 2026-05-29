from __future__ import annotations

import pandas as pd

from historical_reactions import compute_historical_reaction_score

TAG_LABELS = {
    "tight-base": "Tight Base",
    "compressed": "Compression",
    "higher-lows": "Higher Lows",
    "near-pivot": "Near Pivot",
    "fresh-trigger": "Fresh Breakout",
    "ema21-reclaim": "EMA21 Reclaim",
    "sma50-reclaim": "SMA50 Reclaim",
    "sma200-reclaim": "SMA200 Reclaim",
    "strong-historical-reaction": "Strong Historical Reaction",
    "weak-historical-reaction": "Weak Historical Reaction",
    "extended": "Extended",
    "extended-breakout": "Extended",
    "lighter-liquidity": "Low Liquidity",
    "failed-breakout-risk": "Failed Breakout Risk",
}

SCORE_EXPLANATIONS = {
    "signal_score": {
        "title": "Signal Score",
        "description": "A technical swing-trading score that ranks how constructive the current setup looks.",
        "calculation": (
            "Combines Base Quality, Trigger Quality, Follow-through, Risk / Entry Efficiency, "
            "Historical Reaction, an early-structure bonus, and adjusted penalties. The current weights are "
            "25% base, 25% trigger, 15% follow-through, 15% risk, and 20% historical reaction."
        ),
        "why_it_matters": (
            "For a 1-3 month swing trader, it helps separate cleaner setups from stocks that are extended, "
            "weak, illiquid, or missing confirmation. It is not a guaranteed price prediction."
        ),
    },
    "base_quality": {
        "title": "Base Quality",
        "description": "Measures how clean and constructive the stock's recent price structure is.",
        "calculation": (
            "Looks at base width, ATR compression, higher lows, proximity to the 52-week high, and whether "
            "price is aligned above key moving-average trend levels."
        ),
        "why_it_matters": (
            "Clean bases usually give better risk/reward because the stock is building pressure before a "
            "possible breakout."
        ),
    },
    "trigger_quality": {
        "title": "Trigger Quality",
        "description": "Measures whether the stock is showing a fresh, actionable price trigger.",
        "calculation": (
            "Checks distance from pivot, close strength, relative volume, recency of breakout behavior, "
            "and whether price is clearing a meaningful recent high."
        ),
        "why_it_matters": (
            "Swing trades often work best when a constructive base is paired with fresh demand and a nearby "
            "entry trigger."
        ),
    },
    "follow_through": {
        "title": "Follow-through",
        "description": "Estimates whether the stock has enough trend strength to keep moving after entry.",
        "calculation": (
            "Uses 3-month and 6-month return, relative strength versus SPY, weekly trend alignment, and "
            "remaining runway to the 52-week high."
        ),
        "why_it_matters": (
            "A 1-3 month swing needs more than a one-day pop; better follow-through conditions improve the "
            "chance that momentum can persist."
        ),
    },
    "risk_entry": {
        "title": "Risk / Entry Efficiency",
        "description": "Measures whether the current entry area offers reasonable risk compared with upside.",
        "calculation": (
            "Looks at extension from EMA21, distance to a recent base low, reward-to-risk versus the 52-week "
            "high, and whether the invalidation level is practical."
        ),
        "why_it_matters": (
            "Even strong stocks can be poor trades if the entry is too extended or the stop is too wide for "
            "a swing-trading timeframe."
        ),
    },
    "penalties": {
        "title": "Penalties",
        "description": "Reduces the score when setup risks are present.",
        "calculation": (
            "Adds penalty points for lighter liquidity, late or extended breakouts, failed-breakout risk, "
            "high ATR, and wide risk. The final score subtracts 75% of the penalty value."
        ),
        "why_it_matters": (
            "Penalties keep risky or sloppy setups from ranking too highly just because one part of the "
            "chart looks strong."
        ),
    },
    "ema8": {
        "title": "EMA 8",
        "description": "Short-term momentum moving average. Reacts quickly to price changes.",
        "calculation": "Exponential moving average of the last 8 daily closes, with more weight on recent closes.",
        "why_it_matters": "Useful for judging very short-term momentum and whether a move is starting to stretch.",
    },
    "ema21": {
        "title": "EMA 21",
        "description": "Swing-trading trend guide. Often used to judge whether momentum is still healthy.",
        "calculation": "Exponential moving average of the last 21 daily closes, with more weight on recent closes.",
        "why_it_matters": "For 1-3 month swings, the EMA21 often acts as a practical momentum support or reset area.",
    },
    "sma50": {
        "title": "SMA 50",
        "description": "Medium-term institutional trend level. Reclaims or breaks can signal a change in trend quality.",
        "calculation": "Simple average of the last 50 daily closes.",
        "why_it_matters": "A reclaim can show improving sponsorship, while a break can warn that the setup is weakening.",
    },
    "sma200": {
        "title": "SMA 200",
        "description": "Long-term trend line. A reclaim can signal a major regime shift from weak to improving.",
        "calculation": "Simple average of the last 200 daily closes.",
        "why_it_matters": "Swing setups above or reclaiming the SMA200 often have a better long-term trend backdrop.",
    },
    "golden_cross": {
        "title": "Golden Cross",
        "description": "Occurs when the 50-day SMA crosses above the 200-day SMA.",
        "calculation": "Detected when SMA50 was at or below SMA200 on the prior bar and moves above SMA200.",
        "why_it_matters": "It can mark a longer-term trend improvement that supports multi-week swing setups.",
    },
    "death_cross": {
        "title": "Death Cross",
        "description": "Occurs when the 50-day SMA crosses below the 200-day SMA.",
        "calculation": "Detected when SMA50 was at or above SMA200 on the prior bar and moves below SMA200.",
        "why_it_matters": "It can warn that the broader trend is deteriorating and that long setups need more caution.",
    },
    "volume": {
        "title": "Volume",
        "description": "The number of shares traded during a period.",
        "calculation": "Uses daily share volume from the stock's price history.",
        "why_it_matters": "Volume helps confirm whether a breakout is supported by real demand rather than a thin move.",
    },
    "relative_volume": {
        "title": "Relative Volume",
        "description": "Compares current volume with the stock's normal recent trading activity.",
        "calculation": "Current daily volume divided by the 20-day average volume.",
        "why_it_matters": "Higher relative volume can confirm that institutions or active traders are participating.",
    },
    "ma_breach": {
        "title": "MA Breach",
        "description": "Triggered when the daily close moves from below a selected moving average to above it.",
        "calculation": (
            "Compares the prior close and current close with the selected EMA or SMA. Reclaims move from "
            "below to above; breaks move from above to below."
        ),
        "why_it_matters": "Moving-average reclaims can flag improving trend quality early in a swing setup.",
    },
    "historical_reaction": {
        "title": "Historical Reaction",
        "description": "Looks at how the stock performed after similar past moving-average breach events.",
        "calculation": (
            "Scores prior reaction windows using event count, win rates, median forward returns, and drawdown "
            "behavior after comparable technical events."
        ),
        "why_it_matters": (
            "A stock's own history can show whether similar setups tended to follow through or fail within "
            "a 1-3 month holding window."
        ),
    },
}


def get_score_explanation(key: str) -> dict:
    return SCORE_EXPLANATIONS.get(
        key,
        {
            "title": key.replace("_", " ").title(),
            "description": "No explanation is available for this item yet.",
            "calculation": "Not specified.",
            "why_it_matters": "Not specified.",
        },
    )


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def safe_float(value: object) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def score_from_ratio(value: float | None, low: float, high: float) -> float:
    if value is None:
        return 0.0
    if high <= low:
        return 0.0
    return clamp(((value - low) / (high - low)) * 100, 0, 100)


def score_from_inverse_ratio(value: float | None, low: float, high: float) -> float:
    if value is None:
        return 0.0
    if high <= low:
        return 0.0
    return clamp(((high - value) / (high - low)) * 100, 0, 100)


def is_pre_breakout_watch(
    base_score: float,
    trigger_score: float,
    follow_score: float,
    penalty_score: float,
) -> bool:
    return (
        base_score >= 75
        and follow_score >= 50
        and 25 <= trigger_score <= 65
        and penalty_score <= 10
    )


def get_early_structure_bonus(
    base_score: float,
    addv_20: float | None,
    tags: list[str],
) -> float:
    if base_score < 85:
        return 0.0
    if addv_20 is None or addv_20 < 100_000_000:
        return 0.0
    if "near-52w-high" not in tags:
        return 0.0
    if "compressed" not in tags and "higher-lows" not in tags:
        return 0.0

    bonus = 5.0
    if "compressed" in tags and "higher-lows" in tags:
        bonus += 2.0
    return bonus


def get_final_label(
    signal_score: float,
    base_score: float,
    trigger_score: float,
    follow_score: float,
    penalty_score: float,
) -> str:
    if is_pre_breakout_watch(
        base_score=base_score,
        trigger_score=trigger_score,
        follow_score=follow_score,
        penalty_score=penalty_score,
    ):
        return "Pre-Breakout Watch"

    return get_label(signal_score)


def get_effective_penalty(penalty_score: float) -> float:
    return penalty_score * 0.75


def get_label(signal_score: float) -> str:
    if signal_score >= 80:
        return "Prime Swing Candidate"
    if signal_score >= 70:
        return "Actionable Early Breakout"
    if signal_score >= 60:
        return "Pre-Breakout Watch"
    if signal_score >= 45:
        return "Needs More Setup"
    return "Pass"


def format_output_tags(tags: list[str]) -> list[str]:
    return [TAG_LABELS.get(tag, tag.replace("-", " ").title()) for tag in sorted(set(tags))]


def prepare_history(history: pd.DataFrame | None) -> pd.DataFrame:
    if history is None or history.empty:
        return pd.DataFrame()

    expected_columns = ["Open", "High", "Low", "Close", "Volume"]
    cleaned = history.copy()

    for column in expected_columns:
        if column not in cleaned.columns:
            return pd.DataFrame()

    cleaned = cleaned[expected_columns].dropna()
    return cleaned


def get_close_strength(history: pd.DataFrame) -> float | None:
    if history.empty:
        return None

    last_row = history.iloc[-1]
    high = safe_float(last_row["High"])
    low = safe_float(last_row["Low"])
    close = safe_float(last_row["Close"])
    if high is None or low is None:
        return None

    day_range = high - low

    if day_range is None or day_range <= 0 or close is None or low is None:
        return None

    return (close - low) / day_range


def get_atr_pct(history: pd.DataFrame, length: int = 14) -> float | None:
    if history.empty or len(history.index) < length + 1:
        return None

    highs = history["High"]
    lows = history["Low"]
    closes = history["Close"]
    previous_close = closes.shift(1)
    tr = pd.concat(
        [
            highs - lows,
            (highs - previous_close).abs(),
            (lows - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = safe_float(tr.rolling(length).mean().iloc[-1])
    price = safe_float(closes.iloc[-1])

    if atr is None or price in (None, 0):
        return None

    return (atr / price) * 100


def get_recent_pivot(history: pd.DataFrame, lookback: int = 20) -> float | None:
    if history.empty or len(history.index) < lookback + 1:
        return None

    pivot_slice = history["High"].iloc[-(lookback + 1):-1]
    if pivot_slice.empty:
        return None

    return safe_float(pivot_slice.max())


def get_pivot_context(history: pd.DataFrame, price: float | None, lookback: int = 60) -> dict:
    pivot = get_recent_pivot(history, lookback=lookback)
    distance_to_pivot_pct = None
    pivot_status = "Unavailable"
    is_breakout = False
    near_pivot = False
    extended_past_pivot = False

    if pivot not in (None, 0) and price is not None:
        distance_to_pivot_pct = ((price - pivot) / pivot) * 100
        is_breakout = price > pivot
        near_pivot = -3 <= distance_to_pivot_pct <= 0
        extended_past_pivot = distance_to_pivot_pct > 5

        if extended_past_pivot:
            pivot_status = "Extended Past Pivot"
        elif is_breakout:
            pivot_status = "Breaking Out"
        elif near_pivot:
            pivot_status = "Near Pivot"
        else:
            pivot_status = "Below Pivot"

    return {
        "pivot": pivot,
        "pivot_level": pivot,
        "pivot_lookback_days": lookback,
        "distance_to_pivot_pct": distance_to_pivot_pct,
        "is_breakout": is_breakout,
        "near_pivot": near_pivot,
        "extended_past_pivot": extended_past_pivot,
        "breakout_status": pivot_status,
        "pivot_status": pivot_status,
    }


def get_recent_base_low(history: pd.DataFrame, lookback: int = 15) -> float | None:
    if history.empty or len(history.index) < lookback:
        return None
    return safe_float(history["Low"].tail(lookback).min())


def get_weekly_trend_score(history: pd.DataFrame) -> tuple[float, dict]:
    if history.empty or len(history.index) < 60:
        return 0.0, {}

    weekly = history["Close"].resample("W-FRI").last().dropna()
    if len(weekly.index) < 12:
        return 0.0, {}

    ma_10w = safe_float(weekly.rolling(10).mean().iloc[-1])
    ma_30w = safe_float(weekly.rolling(30).mean().iloc[-1]) if len(weekly.index) >= 30 else None
    current = safe_float(weekly.iloc[-1])

    score = 0.0
    if current is not None and ma_10w is not None and current > ma_10w:
        score += 60
    if ma_10w is not None and ma_30w is not None and ma_10w > ma_30w:
        score += 40

    return score, {"weekly_close": current, "ma_10w": ma_10w, "ma_30w": ma_30w}


def score_base_quality(history: pd.DataFrame, price: float | None, high_52w: float | None) -> tuple[float, list[str], dict]:
    if history.empty or len(history.index) < 60 or price is None:
        return 0.0, ["insufficient-base-data"], {}

    closes = history["Close"]
    lows = history["Low"]
    ema_21 = safe_float(closes.ewm(span=21, adjust=False).mean().iloc[-1])
    ema_50 = safe_float(closes.ewm(span=50, adjust=False).mean().iloc[-1])
    ema_200 = safe_float(closes.ewm(span=200, adjust=False).mean().iloc[-1]) if len(closes.index) >= 200 else None

    recent_high = safe_float(history["High"].tail(20).max())
    recent_low = safe_float(lows.tail(20).min())
    base_width_pct = None
    if recent_high not in (None, 0) and recent_low is not None:
        base_width_pct = ((recent_high - recent_low) / recent_high) * 100

    atr_pct = get_atr_pct(history)
    recent_low_5 = safe_float(lows.tail(5).min())
    recent_low_20 = safe_float(lows.tail(20).min())
    higher_lows = None
    if recent_low_5 not in (None, 0) and recent_low_20 is not None:
        higher_lows = recent_low_5 / recent_low_20

    proximity_52w = None
    if high_52w not in (None, 0):
        proximity_52w = price / high_52w

    components = [
        score_from_inverse_ratio(base_width_pct, 8, 35),
        score_from_inverse_ratio(atr_pct, 2.5, 8.5),
        score_from_ratio(higher_lows, 0.98, 1.05),
        score_from_ratio(proximity_52w, 0.75, 1.00),
    ]

    trend_score = 0.0
    if ema_21 is not None and price > ema_21:
        trend_score += 40
    if ema_50 is not None and ema_21 is not None and ema_21 > ema_50:
        trend_score += 35
    if ema_200 is not None and ema_50 is not None and ema_50 > ema_200:
        trend_score += 25
    components.append(trend_score)

    tags: list[str] = []
    if base_width_pct is not None and base_width_pct <= 15:
        tags.append("tight-base")
    if atr_pct is not None and atr_pct <= 4.5:
        tags.append("compressed")
    if higher_lows is not None and higher_lows >= 1.0:
        tags.append("higher-lows")
    if proximity_52w is not None and proximity_52w >= 0.9:
        tags.append("near-52w-high")

    return sum(components) / len(components), tags, {
        "base_width_pct": base_width_pct,
        "atr_pct": atr_pct,
        "higher_lows_ratio": higher_lows,
        "proximity_52w_high": proximity_52w,
        "ema_21": ema_21,
        "ema_50": ema_50,
        "ema_200": ema_200,
    }


def score_trigger_quality(history: pd.DataFrame, price: float | None, volume_ratio: float | None) -> tuple[float, list[str], dict]:
    if history.empty or len(history.index) < 25 or price is None:
        return 0.0, ["insufficient-trigger-data"], {}

    pivot_context = get_pivot_context(history, price, lookback=60)
    pivot = safe_float(pivot_context.get("pivot"))
    close_strength = get_close_strength(history)
    recent_high_65 = safe_float(history["High"].iloc[-66:-1].max()) if len(history.index) >= 66 else pivot
    rolling_high_20 = history["High"].rolling(20).max()
    breakout_days = rolling_high_20[history["Close"] >= rolling_high_20.shift(1)].dropna()
    days_since_breakout = len(history.index)
    if not breakout_days.empty:
        days_since_breakout = len(history.loc[breakout_days.index[-1]:].index) - 1

    breakout_pct = None
    if pivot not in (None, 0):
        breakout_pct = ((price - pivot) / pivot) * 100

    components = [
        score_from_ratio(breakout_pct, -2, 4),
        score_from_ratio(close_strength, 0.45, 0.9),
        score_from_ratio(volume_ratio, 0.9, 2.2),
        score_from_inverse_ratio(days_since_breakout, 0, 10),
    ]

    level_significance = 0.0
    if recent_high_65 is not None and price >= recent_high_65:
        level_significance = 100.0
    elif breakout_pct is not None and breakout_pct >= 0:
        level_significance = 65.0
    components.append(level_significance)

    tags: list[str] = []
    if breakout_pct is not None and -1 <= breakout_pct <= 3:
        tags.append("fresh-trigger")
    if pivot_context.get("near_pivot"):
        tags.append("near-pivot")
    if pivot_context.get("extended_past_pivot"):
        tags.append("extended")
    if close_strength is not None and close_strength >= 0.7:
        tags.append("strong-close")
    if volume_ratio is not None and volume_ratio >= 1.2:
        tags.append("volume-confirmed")
    if recent_high_65 is not None and price >= recent_high_65:
        tags.append("range-expansion")

    metrics = {
        "breakout_pct": breakout_pct,
        "close_strength": close_strength,
        "days_since_breakout": days_since_breakout,
        "recent_high_65": recent_high_65,
    }
    metrics.update(pivot_context)

    return sum(components) / len(components), tags, metrics


def build_score_explanation(score_data: dict) -> list[str]:
    explanations = []
    metrics = score_data.get("metrics", {})
    tags = set(score_data.get("raw_tags", score_data.get("tags", [])))

    base_score = safe_float(score_data.get("base_score")) or 0.0
    trigger_score = safe_float(score_data.get("trigger_score")) or 0.0
    follow_score = safe_float(score_data.get("follow_through_score")) or 0.0
    risk_score = safe_float(score_data.get("risk_score")) or 0.0
    penalty_score = safe_float(score_data.get("penalty_score")) or 0.0
    historical_score = safe_float(score_data.get("historical_score")) or 50.0

    if base_score >= 75:
        explanations.append("Strong base quality")
    elif base_score >= 50:
        explanations.append("Base is constructive but not tight enough yet")
    else:
        explanations.append("Weak or incomplete base structure")

    if trigger_score >= 70:
        explanations.append("Fresh trigger or breakout characteristics are present")
    elif trigger_score >= 40:
        explanations.append("Trigger is developing but not decisive")
    else:
        explanations.append("Weak trigger / no fresh breakout")

    if follow_score >= 65:
        explanations.append("Follow-through profile improving")
    elif follow_score < 35:
        explanations.append("Follow-through profile is still weak")

    extension_pct = safe_float(metrics.get("extension_from_ema21_pct"))
    if extension_pct is not None and extension_pct > 6:
        explanations.append("Entry extended from EMA21")
    elif risk_score >= 65:
        explanations.append("Entry risk is relatively efficient")

    breakout_status = metrics.get("pivot_status") or metrics.get("breakout_status")
    distance_to_pivot_pct = safe_float(metrics.get("distance_to_pivot_pct"))
    if breakout_status == "Near Pivot":
        explanations.append("Price is within 3% below the pivot")
    elif breakout_status == "Breaking Out":
        explanations.append("Price is breaking above the pivot")
    elif breakout_status == "Extended Past Pivot":
        explanations.append("Price is extended past the pivot")
    elif distance_to_pivot_pct is not None and distance_to_pivot_pct < -3:
        explanations.append("Price remains below the pivot")

    historical_summary = score_data.get("historical_summary")
    if historical_summary:
        explanations.append(historical_summary)

    penalty_reasons = []
    if "lighter-liquidity" in tags:
        penalty_reasons.append("low liquidity")
    if "wild-atr" in tags:
        penalty_reasons.append("high ATR")
    if "extended-breakout" in tags:
        penalty_reasons.append("late move")
    if "wide-risk" in tags:
        penalty_reasons.append("wide risk")
    if penalty_score > 0 and penalty_reasons:
        explanations.append(f"Penalty from {', '.join(penalty_reasons)}")
    elif penalty_score > 0:
        explanations.append("Penalty factors reduced the score")

    if historical_score >= 70:
        explanations.append("Historical reactions favor this setup type")
    elif historical_score <= 40:
        explanations.append("Historical reactions have been weak for this setup type")

    return explanations[:6]


def is_near_ma_reclaim(price: float | None, metrics: dict) -> bool:
    if price in (None, 0):
        return False

    for key in ("ema_21", "ema_50", "ema_200"):
        moving_average = safe_float(metrics.get(key))
        if moving_average not in (None, 0) and -2 <= ((price - moving_average) / moving_average) * 100 <= 2:
            return True
    return False


def get_recent_ma_reclaim_tags(history: pd.DataFrame, price: float | None) -> list[str]:
    if history.empty or len(history.index) < 2 or price is None:
        return []

    closes = history["Close"]
    reclaim_tags = []
    ma_configs = (
        ("ema21-reclaim", closes.ewm(span=21, adjust=False).mean()),
        ("sma50-reclaim", closes.rolling(50).mean()),
        ("sma200-reclaim", closes.rolling(200).mean()),
    )
    for tag, moving_average in ma_configs:
        previous_close = safe_float(closes.iloc[-2])
        current_close = safe_float(closes.iloc[-1])
        previous_ma = safe_float(moving_average.iloc[-2])
        current_ma = safe_float(moving_average.iloc[-1])
        if previous_close is None or current_close is None or previous_ma in (None, 0) or current_ma in (None, 0):
            continue
        if previous_close <= previous_ma and current_close > current_ma:
            reclaim_tags.append(tag)
    return reclaim_tags


def score_follow_through_potential(
    history: pd.DataFrame,
    price: float | None,
    high_52w: float | None,
    benchmark_history: pd.DataFrame | None,
) -> tuple[float, list[str], dict]:
    if history.empty or len(history.index) < 126 or price is None:
        return 0.0, ["insufficient-follow-through-data"], {}

    closes = history["Close"]
    ret_3m = safe_float(((closes.iloc[-1] / closes.iloc[-63]) - 1) * 100) if len(closes.index) >= 64 else None
    ret_6m = safe_float(((closes.iloc[-1] / closes.iloc[-126]) - 1) * 100) if len(closes.index) >= 127 else None

    rs_3m = None
    if benchmark_history is not None and not benchmark_history.empty:
        bench_closes = benchmark_history["Close"].dropna()
        if len(bench_closes.index) >= 64:
            bench_ret_3m = safe_float(((bench_closes.iloc[-1] / bench_closes.iloc[-63]) - 1) * 100)
            if ret_3m is not None and bench_ret_3m is not None:
                rs_3m = ret_3m - bench_ret_3m

    weekly_score, weekly_metrics = get_weekly_trend_score(history)

    runway_pct = None
    if high_52w not in (None, 0):
        runway_pct = ((high_52w - price) / high_52w) * 100

    runway_score = 0.0
    if runway_pct is not None:
        if 2 <= runway_pct <= 12:
            runway_score = 100.0
        elif 0 <= runway_pct < 2:
            runway_score = 70.0
        elif 12 < runway_pct <= 25:
            runway_score = 55.0

    components = [
        score_from_ratio(ret_3m, 0, 25),
        score_from_ratio(ret_6m, 5, 40),
        score_from_ratio(rs_3m, -5, 15),
        weekly_score,
        runway_score,
    ]

    tags: list[str] = []
    if ret_3m is not None and ret_3m >= 10:
        tags.append("3m-momentum")
    if rs_3m is not None and rs_3m >= 3:
        tags.append("rs-leader")
    if weekly_score >= 70:
        tags.append("weekly-trend-aligned")
    if runway_pct is not None and 2 <= runway_pct <= 12:
        tags.append("runway-open")

    metrics = {
        "ret_3m_pct": ret_3m,
        "ret_6m_pct": ret_6m,
        "rs_3m_vs_spy_pct": rs_3m,
        "runway_pct": runway_pct,
    }
    metrics.update(weekly_metrics)

    return sum(components) / len(components), tags, metrics


def score_risk_entry_efficiency(history: pd.DataFrame, price: float | None, high_52w: float | None) -> tuple[float, list[str], dict]:
    if history.empty or len(history.index) < 30 or price is None:
        return 0.0, ["insufficient-risk-data"], {}

    closes = history["Close"]
    ema_21 = safe_float(closes.ewm(span=21, adjust=False).mean().iloc[-1])
    recent_base_low = get_recent_base_low(history, lookback=15)

    extension_pct = None
    if ema_21 not in (None, 0):
        extension_pct = ((price - ema_21) / ema_21) * 100

    stop_distance_pct = None
    if recent_base_low not in (None, 0):
        stop_distance_pct = ((price - recent_base_low) / price) * 100

    reward_to_risk = None
    if high_52w not in (None, 0) and stop_distance_pct not in (None, 0):
        upside_pct = ((high_52w - price) / price) * 100
        reward_to_risk = upside_pct / stop_distance_pct

    invalidation_score = 0.0
    if stop_distance_pct is not None:
        if 3 <= stop_distance_pct <= 8:
            invalidation_score = 100.0
        elif 2 <= stop_distance_pct < 3 or 8 < stop_distance_pct <= 10:
            invalidation_score = 70.0
        elif 10 < stop_distance_pct <= 14:
            invalidation_score = 35.0

    components = [
        score_from_inverse_ratio(extension_pct, 1, 8),
        score_from_inverse_ratio(stop_distance_pct, 3, 12),
        score_from_ratio(reward_to_risk, 0.8, 2.5),
        invalidation_score,
    ]

    tags: list[str] = []
    if extension_pct is not None and extension_pct <= 4:
        tags.append("near-ema21")
    if stop_distance_pct is not None and 3 <= stop_distance_pct <= 8:
        tags.append("clean-stop")
    if reward_to_risk is not None and reward_to_risk >= 1.5:
        tags.append("good-r-multiple")

    return sum(components) / len(components), tags, {
        "extension_from_ema21_pct": extension_pct,
        "stop_distance_pct": stop_distance_pct,
        "reward_to_risk": reward_to_risk,
    }


def score_penalties(
    price: float | None,
    addv_20: float | None,
    history: pd.DataFrame,
    trigger_metrics: dict,
    risk_metrics: dict,
) -> tuple[float, list[str], dict]:
    penalty = 0.0
    tags: list[str] = []

    if addv_20 is not None and addv_20 < 50_000_000:
        penalty += 6
        tags.append("lighter-liquidity")

    breakout_pct = safe_float(trigger_metrics.get("breakout_pct"))
    if breakout_pct is not None and breakout_pct > 7:
        penalty += 12
        tags.append("extended-breakout")

    pivot = safe_float(trigger_metrics.get("pivot"))
    if price is not None and pivot is not None and price < pivot * 0.985:
        penalty += 14
        tags.append("failed-breakout-risk")

    atr_pct = get_atr_pct(history)
    if atr_pct is not None and atr_pct > 7.5:
        penalty += 10
        tags.append("wild-atr")

    stop_distance_pct = safe_float(risk_metrics.get("stop_distance_pct"))
    if stop_distance_pct is not None and stop_distance_pct > 12:
        penalty += 8
        tags.append("wide-risk")

    return penalty, tags, {
        "atr_pct": atr_pct,
        "late_breakout_pct": breakout_pct,
        "stop_distance_pct": stop_distance_pct,
    }


def calculate_signal_score(
    stock_data: dict,
    benchmark_history: pd.DataFrame | None = None,
    min_addv: float | None = None,
    calculate_historical_edge: bool = True,
) -> dict:
    history = prepare_history(stock_data.get("history"))
    price = safe_float(stock_data.get("price"))
    high_52w = safe_float(stock_data.get("high_52w"))
    addv_20 = safe_float(stock_data.get("addv_20"))
    volume_ratio = safe_float(stock_data.get("volume_ratio"))

    if history.empty or price is None:
        return {
            "signal_score": 0.0,
            "label": "Pass",
            "base_score": 0.0,
            "trigger_score": 0.0,
            "follow_through_score": 0.0,
            "risk_score": 0.0,
            "historical_score": 50.0,
            "historical_confidence": "Low",
            "historical_summary": "Insufficient history for historical reaction analysis.",
            "active_reaction_event": None,
            "historical_stats": {},
            "penalty_score": 0.0,
            "tags": ["insufficient-data"],
            "metrics": {},
        }

    base_score, base_tags, base_metrics = score_base_quality(history, price, high_52w)
    trigger_score, trigger_tags, trigger_metrics = score_trigger_quality(history, price, volume_ratio)
    follow_score, follow_tags, follow_metrics = score_follow_through_potential(
        history=history,
        price=price,
        high_52w=high_52w,
        benchmark_history=benchmark_history,
    )
    risk_score, risk_tags, risk_metrics = score_risk_entry_efficiency(history, price, high_52w)
    penalty_score, penalty_tags, penalty_metrics = score_penalties(
        price=price,
        addv_20=addv_20,
        history=history,
        trigger_metrics=trigger_metrics,
        risk_metrics=risk_metrics,
    )
    current_setup = {}
    current_setup.update(base_metrics)
    current_setup.update(trigger_metrics)
    current_setup.update(risk_metrics)
    if calculate_historical_edge:
        historical_reaction = compute_historical_reaction_score(history, current_setup=current_setup)
    else:
        historical_reaction = {
            "historical_score": 50.0,
            "confidence": "N/A",
            "summary": "Historical Edge not calculated. Turn on Historical Edge to estimate past setup performance.",
            "active_event": None,
            "best_event": None,
            "stats": {},
        }
    historical_score = safe_float(historical_reaction.get("historical_score")) or 50.0
    ma_reclaim_tags = get_recent_ma_reclaim_tags(history, price)
    historical_tags = []
    if historical_score >= 70:
        historical_tags.append("strong-historical-reaction")
    elif historical_score <= 40:
        historical_tags.append("weak-historical-reaction")
    tags = sorted(set(base_tags + trigger_tags + follow_tags + risk_tags + penalty_tags + ma_reclaim_tags + historical_tags))
    early_structure_bonus = get_early_structure_bonus(
        base_score=base_score,
        addv_20=addv_20,
        tags=tags,
    )
    effective_penalty = get_effective_penalty(penalty_score)

    signal_score = clamp(
        (0.25 * base_score)
        + (0.25 * trigger_score)
        + (0.15 * follow_score)
        + (0.15 * risk_score)
        + (0.20 * historical_score)
        + early_structure_bonus
        - effective_penalty,
        0,
        100,
    )

    metrics = {}
    metrics.update(base_metrics)
    metrics.update(trigger_metrics)
    metrics.update(follow_metrics)
    metrics.update(risk_metrics)
    near_setup_level = bool(metrics.get("near_pivot")) or is_near_ma_reclaim(price, metrics)
    if base_score < 50:
        signal_score = min(signal_score, 70)
    if trigger_score < 20 and not near_setup_level:
        signal_score = min(signal_score, 65)
    if penalty_score > 25:
        signal_score = min(signal_score, 60)
    if min_addv is not None and addv_20 is not None and addv_20 < min_addv:
        signal_score = min(signal_score, 55)
    extension_pct = safe_float(risk_metrics.get("extension_from_ema21_pct"))
    volume_confirms = volume_ratio is not None and volume_ratio >= 1.2
    if extension_pct is not None and extension_pct > 10 and not (historical_score >= 75 and volume_confirms):
        signal_score = min(signal_score, 65)

    metrics.update(penalty_metrics)
    metrics["early_structure_bonus"] = early_structure_bonus
    metrics["effective_penalty"] = effective_penalty

    score_result = {
        "signal_score": round(signal_score, 1),
        "label": get_final_label(
            signal_score=signal_score,
            base_score=base_score,
            trigger_score=trigger_score,
            follow_score=follow_score,
            penalty_score=penalty_score,
        ),
        "base_score": round(base_score, 1),
        "trigger_score": round(trigger_score, 1),
        "follow_through_score": round(follow_score, 1),
        "risk_score": round(risk_score, 1),
        "historical_score": round(historical_score, 1),
        "historical_confidence": historical_reaction.get("confidence", "Low"),
        "historical_summary": historical_reaction.get("summary", ""),
        "active_reaction_event": historical_reaction.get("active_event"),
        "best_reaction_event": historical_reaction.get("best_event"),
        "historical_stats": historical_reaction.get("stats", {}),
        "penalty_score": round(penalty_score, 1),
        "pivot_level": metrics.get("pivot_level"),
        "distance_to_pivot_pct": metrics.get("distance_to_pivot_pct"),
        "pivot_status": metrics.get("pivot_status"),
        "tags": format_output_tags(tags),
        "raw_tags": tags,
        "metrics": metrics,
    }
    score_result["explanation"] = build_score_explanation(score_result)
    return score_result
