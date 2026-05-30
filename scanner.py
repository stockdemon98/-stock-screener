from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from time import perf_counter
import json

import pandas as pd

from data_fetch import chunk_tickers, get_batch_stock_data, get_batch_stock_data_chunk, get_batch_stock_data_chunk_detailed, get_benchmark_history, get_market_regime_histories
from data_fetch import load_scanner_snapshot, save_scanner_snapshot
from regime import calculate_market_regime
from scoring import calculate_signal_score
from setup_quality import calculate_setup_quality
from symbol_universe import EX_MAJOR_INDEXES_UNIVERSE, FULL_MARKET_UNIVERSE, get_nyse_nasdaq_common_stock_meta, get_symbol_metadata_map
from universes import get_universe
from utils import sort_rows_desc


UNIVERSE_CONFIG = {
    "SP500_FULL": {"min_price": 5, "min_addv": 20_000_000, "max_atr_pct": 7, "top_score": 80, "ticker_limit": 250},
    "QQQ": {"min_price": 5, "min_addv": 20_000_000, "max_atr_pct": 8, "top_score": 78, "ticker_limit": 75},
    "IWM": {"min_price": 10, "min_addv": 30_000_000, "max_atr_pct": 6, "top_score": 82, "ticker_limit": 250},
    "MDY": {"min_price": 8, "min_addv": 20_000_000, "max_atr_pct": 7, "top_score": 80, "ticker_limit": 75},
    FULL_MARKET_UNIVERSE: {"min_price": None, "min_addv": None, "max_atr_pct": None, "top_score": 75, "ticker_limit": 0},
    EX_MAJOR_INDEXES_UNIVERSE: {"min_price": None, "min_addv": None, "max_atr_pct": None, "top_score": 75, "ticker_limit": 0},
    "ELITE": {"min_price": 10, "min_addv": 30_000_000, "max_atr_pct": 7, "top_score": 82, "ticker_limit": 75},
    "CUSTOM_MOMENTUM": {"min_price": 2, "min_addv": 2_000_000, "max_atr_pct": 12, "top_score": 75, "ticker_limit": 75},
    "RUSSELL3000": {"min_price": 5, "min_addv": 5_000_000, "max_atr_pct": 10, "top_score": 78, "ticker_limit": 250},
}
SCAN_MODE_CONFIG = {
    "Fast Scan": {"candidate_limit": 250, "min_setup_score": 25, "max_extension_pct": 20},
    "Standard Scan": {"candidate_limit": 700, "min_setup_score": 10, "max_extension_pct": 30},
    "Full Scan": {"candidate_limit": None, "min_setup_score": None, "max_extension_pct": None},
}
LOOKBACK_TRADING_DAYS = {
    "1W": 5,
    "2W": 10,
    "3W": 15,
    "4W": 20,
    "8W": 40,
    "12W": 60,
}
MA_COLUMNS = {
    "EMA8": {"column": "EMA8", "period": 8, "type": "ema"},
    "EMA21": {"column": "EMA21", "period": 21, "type": "ema"},
    "SMA50": {"column": "SMA50", "period": 50, "type": "sma"},
    "SMA100": {"column": "SMA100", "period": 100, "type": "sma"},
    "SMA200": {"column": "SMA200", "period": 200, "type": "sma"},
}
TRIGGER_EVENT_MA_MAP = {
    "Crossed above EMA8": "EMA8",
    "Crossed above EMA21": "EMA21",
    "Crossed above SMA50": "SMA50",
    "Crossed above SMA100": "SMA100",
    "Crossed above SMA200": "SMA200",
}
TRIGGER_TREND_RANK = {
    "A+ Long Setup": 5,
    "Strong Long Setup": 4,
    "Constructive Long Setup": 3,
    "Watchlist / Developing": 2,
    "Needs Work": 1,
}
TRIGGER_PRIORITY_RANK = {
    "Golden Cross: SMA50 crossed above SMA200": 0,
    "Crossed above SMA200": 1,
    "Crossed above SMA100": 2,
    "Crossed above SMA50": 3,
    "Crossed above EMA21": 4,
    "Crossed above EMA8": 5,
    "Death Cross: SMA50 crossed below SMA200": 6,
}
BREAKOUT_TREND_RANK = {
    "Strong Uptrend": 4,
    "Constructive Uptrend": 3,
    "Strong Long Setup": 4,
    "Pullback Entry Setup": 3,
    "Watchlist": 2,
    "Neutral / Mixed": 1,
    "Long Watchlist": 2,
    "Too Extended": 1,
    "Below Key MAs": 0,
    "Weak / Avoid": 0,
    "Downtrend / Avoid": 0,
    "Insufficient Data": 0,
}


def get_universe_config(universe_name: str) -> dict:
    return UNIVERSE_CONFIG.get(universe_name, UNIVERSE_CONFIG["SP500_FULL"]).copy()


def is_full_market_universe(universe_name: str) -> bool:
    return str(universe_name).strip().upper() in {FULL_MARKET_UNIVERSE, EX_MAJOR_INDEXES_UNIVERSE}


def get_scan_mode_config(scan_mode: str | None) -> dict:
    return SCAN_MODE_CONFIG.get(scan_mode or "Fast Scan", SCAN_MODE_CONFIG["Fast Scan"]).copy()


def get_scan_mode_default_limit(scan_mode: str | None) -> int:
    candidate_limit = get_scan_mode_config(scan_mode).get("candidate_limit")
    return int(candidate_limit) if candidate_limit is not None else 5000


def build_scan_meta(
    universe_count: int,
    max_scan_cap: int,
    requested_count: int,
    scanned_count: int,
    missing_data_count: int,
    filter_skip_count: int,
    duration_seconds: float,
    result_count: int = 0,
    timings: dict | None = None,
    warning: str = "",
) -> dict:
    return {
        "universe_count": universe_count,
        "max_scan_cap": max_scan_cap,
        "requested_count": requested_count,
        "scanned_count": scanned_count,
        "missing_data_count": missing_data_count,
        "filter_skip_count": filter_skip_count,
        "duration_seconds": duration_seconds,
        "result_count": result_count,
        "timings": timings or {},
        "warning": warning,
    }


def build_scanned_row(stock_data: dict, score_data: dict) -> dict:
    price = stock_data.get("price") or 0.0
    change_pct = stock_data.get("change_pct") or 0.0
    volume_ratio = stock_data.get("volume_ratio") or 0.0
    addv = stock_data.get("addv_20") or 0.0
    history = stock_data.get("history", pd.DataFrame())
    long_trend = build_long_trend_snapshot(history, price, volume_ratio)

    return {
        "ticker": stock_data.get("ticker", "N/A"),
        "price": price,
        "change_pct": change_pct,
        "volume_ratio": volume_ratio,
        "avg_volume": stock_data.get("avg_volume_20") or 0.0,
        "addv": addv,
        "atr_pct": get_atr_pct(history),
        "signal_score": score_data.get("signal_score", 0.0),
        "label": score_data.get("label", "Pass"),
        "base_score": score_data.get("base_score", 0.0),
        "trigger_score": score_data.get("trigger_score", 0.0),
        "follow_through_score": score_data.get("follow_through_score", 0.0),
        "risk_score": score_data.get("risk_score", 0.0),
        "historical_score": score_data.get("historical_score", 50.0),
        "historical_confidence": score_data.get("historical_confidence", "Low"),
        "historical_summary": score_data.get("historical_summary", ""),
        "active_reaction_event": score_data.get("active_reaction_event"),
        "best_reaction_event": score_data.get("best_reaction_event"),
        "historical_stats": score_data.get("historical_stats", {}),
        "penalty_score": score_data.get("penalty_score", 0.0),
        "tags": score_data.get("tags", []),
        "metrics": score_data.get("metrics", {}),
        **long_trend,
    }


def format_snapshot_tags(tags: object) -> str:
    if isinstance(tags, list):
        return ", ".join(str(tag) for tag in tags[:8])
    if tags is None:
        return ""
    return str(tags)


def parse_snapshot_tags(tags: object) -> list[str]:
    if isinstance(tags, list):
        return tags
    if tags is None or pd.isna(tags):
        return []
    text = str(tags).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(tag) for tag in parsed]
    except json.JSONDecodeError:
        pass
    return [tag.strip() for tag in text.split(",") if tag.strip()]


def snapshot_text(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def snapshot_bool(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y"}


def build_snapshot_row(stock_data: dict, score_data: dict, setup_quality: dict | None = None) -> dict:
    history = stock_data.get("history", pd.DataFrame())
    close = history["Close"] if not history.empty and "Close" in history.columns else pd.Series(dtype=float)
    latest_price = stock_data.get("price") or 0.0
    ema8 = close.ewm(span=8, adjust=False, min_periods=8).mean().iloc[-1] if len(close.index) >= 8 else None
    ema21 = close.ewm(span=21, adjust=False, min_periods=21).mean().iloc[-1] if len(close.index) >= 21 else None
    sma50 = close.rolling(50).mean().iloc[-1] if len(close.index) >= 50 else None
    sma100 = close.rolling(100).mean().iloc[-1] if len(close.index) >= 100 else None
    sma200 = close.rolling(200).mean().iloc[-1] if len(close.index) >= 200 else None
    tags = score_data.get("tags", [])
    long_trend = build_long_trend_snapshot(history, latest_price, stock_data.get("volume_ratio"))
    breakout_metrics = build_breakout_snapshot_metrics(history, latest_price)
    setup_quality = setup_quality or {}
    trend_status = calculate_trend_status(
        {
            "price": latest_price,
            "ema8": ema8,
            "ema21": ema21,
            "sma50": sma50,
            "sma200": sma200,
            "sma50_trend": long_trend.get("sma50_trend"),
            "sma200_trend": long_trend.get("sma200_trend"),
            "trend_quality": setup_quality.get("trend_quality"),
            "distance_from_ema21": ((latest_price / ema21) - 1) * 100 if ema21 else None,
            "market_regime": setup_quality.get("market_regime"),
        }
    )

    return {
        "ticker": stock_data.get("ticker", "N/A"),
        "company": stock_data.get("company"),
        "exchange": stock_data.get("exchange"),
        "latest_price": latest_price,
        "close": latest_price,
        "price": latest_price,
        "daily_change_pct": stock_data.get("change_pct") or 0.0,
        "change_pct": stock_data.get("change_pct") or 0.0,
        "volume": stock_data.get("volume") or 0.0,
        "avg_volume_20d": stock_data.get("avg_volume_20") or 0.0,
        "avg_volume": stock_data.get("avg_volume_20") or 0.0,
        "dollar_volume": stock_data.get("addv_20") or 0.0,
        "addv": stock_data.get("addv_20") or 0.0,
        "volume_ratio": stock_data.get("volume_ratio") or 0.0,
        "ema8": float(ema8) if ema8 is not None and not pd.isna(ema8) else None,
        "ema21": float(ema21) if ema21 is not None and not pd.isna(ema21) else None,
        "sma50": float(sma50) if sma50 is not None and not pd.isna(sma50) else None,
        "sma100": float(sma100) if sma100 is not None and not pd.isna(sma100) else None,
        "sma200": float(sma200) if sma200 is not None and not pd.isna(sma200) else None,
        "distance_from_ema8": ((latest_price / ema8) - 1) * 100 if ema8 else None,
        "distance_from_ema21": ((latest_price / ema21) - 1) * 100 if ema21 else None,
        "distance_from_sma50": ((latest_price / sma50) - 1) * 100 if sma50 else None,
        "distance_from_sma100": ((latest_price / sma100) - 1) * 100 if sma100 else None,
        "distance_from_sma200": ((latest_price / sma200) - 1) * 100 if sma200 else None,
        "ema8_trend": long_trend.get("ema8_trend"),
        "ema21_trend": long_trend.get("ema21_trend"),
        "sma50_trend": long_trend.get("sma50_trend"),
        "sma200_trend": long_trend.get("sma200_trend"),
        "trend_status": trend_status,
        "ema_stack_status": long_trend.get("ema_stack_status"),
        "ema8_status": long_trend.get("ema8_status"),
        "ema21_status": long_trend.get("ema21_status"),
        "extension_status": long_trend.get("extension_status"),
        "pullback_quality": long_trend.get("pullback_quality"),
        "long_status": long_trend.get("long_status"),
        "long_summary": long_trend.get("long_summary"),
        "atr_pct": get_atr_pct(history),
        "base_score": score_data.get("base_score", 0.0),
        "trigger_score": score_data.get("trigger_score", 0.0),
        "follow_through_score": score_data.get("follow_through_score", 0.0),
        "risk_score": score_data.get("risk_score", 0.0),
        "historical_score": score_data.get("historical_score", 50.0),
        "setup_type": setup_quality.get("setup_type", "N/A"),
        "a_plus_status": setup_quality.get("a_plus_status", "N/A"),
        "a_plus_qualified": snapshot_bool(setup_quality.get("a_plus_qualified", False)),
        "setup_signal_score": setup_quality.get("setup_signal_score"),
        "market_regime": setup_quality.get("market_regime", "N/A"),
        "trend_quality": setup_quality.get("trend_quality"),
        "base_quality_new": setup_quality.get("base_quality"),
        "compression_quality": setup_quality.get("compression_quality"),
        "trigger_quality_new": setup_quality.get("trigger_quality"),
        "risk_reward_quality": setup_quality.get("risk_reward_quality"),
        "historical_edge": setup_quality.get("historical_edge"),
        "historical_hit_rate": setup_quality.get("historical_hit_rate"),
        "historical_sample_size": setup_quality.get("historical_sample_size"),
        "historical_confidence_new": setup_quality.get("historical_confidence"),
        "historical_event_type": setup_quality.get("historical_event_type"),
        "suggested_pivot": setup_quality.get("suggested_pivot"),
        "suggested_stop": setup_quality.get("suggested_stop"),
        "risk_pct": setup_quality.get("risk_pct"),
        "reward_risk": setup_quality.get("reward_risk"),
        "close_strength": setup_quality.get("close_strength"),
        "a_plus_explanation": setup_quality.get("a_plus_explanation"),
        "penalty_score": score_data.get("penalty_score", 0.0),
        "signal_score": score_data.get("signal_score", 0.0),
        "setup_label": score_data.get("label", "Pass"),
        "label": score_data.get("label", "Pass"),
        "tags": json.dumps(tags),
        "date_updated": datetime.now().isoformat(timespec="seconds"),
        **breakout_metrics,
    }


def build_lightweight_score_data(stock_data: dict) -> dict:
    return {
        "signal_score": 0.0,
        "label": "Unscored",
        "base_score": 0.0,
        "trigger_score": 0.0,
        "follow_through_score": 0.0,
        "risk_score": 0.0,
        "historical_score": 50.0,
        "historical_confidence": "N/A",
        "historical_summary": "Deep scoring skipped by fast full-market snapshot refresh.",
        "active_reaction_event": None,
        "best_reaction_event": None,
        "historical_stats": {},
        "penalty_score": 0.0,
        "tags": [],
        "raw_tags": [],
        "metrics": {},
        "explanation": [],
    }


def build_lightweight_setup_quality() -> dict:
    return {
        "setup_type": "Not Deep Scored",
        "a_plus_status": "N/A",
        "a_plus_qualified": False,
        "setup_signal_score": 0.0,
        "market_regime": "N/A",
        "trend_quality": None,
        "base_quality": None,
        "compression_quality": None,
        "trigger_quality": None,
        "risk_reward_quality": None,
        "historical_edge": None,
        "historical_hit_rate": None,
        "historical_sample_size": None,
        "historical_confidence": "N/A",
        "historical_event_type": None,
        "suggested_pivot": None,
        "suggested_stop": None,
        "risk_pct": None,
        "reward_risk": None,
        "close_strength": None,
        "a_plus_explanation": "Deep setup scoring skipped by fast full-market snapshot refresh.",
    }


def safe_pct_distance(price: object, moving_average: object) -> float | None:
    try:
        price_value = float(price)
        ma_value = float(moving_average)
    except (TypeError, ValueError):
        return None
    if ma_value == 0:
        return None
    return ((price_value / ma_value) - 1) * 100


def detect_ma_trend(series: pd.Series, lookback: int = 5, flat_threshold_pct: float = 0.15) -> str:
    values = series.dropna()
    if len(values.index) <= lookback:
        return "Insufficient Data"
    latest = values.iloc[-1]
    prior = values.iloc[-lookback - 1]
    if pd.isna(latest) or pd.isna(prior) or prior == 0:
        return "Insufficient Data"
    change_pct = ((latest / prior) - 1) * 100
    if change_pct > flat_threshold_pct:
        return "Rising"
    if change_pct < -flat_threshold_pct:
        return "Falling"
    return "Flat"


def get_latest_value(series: pd.Series) -> float | None:
    values = series.dropna()
    if values.empty:
        return None
    return safe_number(values.iloc[-1])


def get_previous_value(series: pd.Series) -> float | None:
    values = series.dropna()
    if len(values.index) < 2:
        return None
    return safe_number(values.iloc[-2])


def classify_ema_stack(price: object, ema8: object, ema21: object, sma50: object, sma200: object, ema21_trend: str) -> str:
    price_value = safe_number(price)
    ema8_value = safe_number(ema8)
    ema21_value = safe_number(ema21)
    sma50_value = safe_number(sma50)
    sma200_value = safe_number(sma200)
    if price_value is None or ema8_value is None or ema21_value is None or sma50_value is None:
        return "Insufficient Data"
    if price_value < sma50_value and (sma200_value is None or price_value < sma200_value) or ema21_trend == "Falling":
        return "Weak / Avoid"
    if price_value < ema21_value or price_value < sma50_value:
        return "Below Key MAs"
    if sma200_value is not None and price_value > ema8_value > ema21_value > sma50_value > sma200_value:
        return "Perfect Bullish Stack"
    if price_value > ema8_value and price_value > ema21_value:
        return "Bullish but Early"
    return "Mixed Trend"


def classify_extension_status(distance_from_ema21: object) -> str:
    distance = safe_number(distance_from_ema21)
    if distance is None:
        return "N/A"
    if distance < 0:
        return "Below EMA21"
    if distance <= 3:
        return "Healthy"
    if distance <= 8:
        return "Slightly Extended"
    if distance <= 12:
        return "Extended"
    return "Too Extended"


def classify_ma_position_status(
    latest_close: object,
    prior_close: object,
    latest_ma: object,
    prior_ma: object,
    label: str,
    extended_threshold_pct: float,
) -> str:
    latest_close_value = safe_number(latest_close)
    prior_close_value = safe_number(prior_close)
    latest_ma_value = safe_number(latest_ma)
    prior_ma_value = safe_number(prior_ma)
    if latest_close_value is None or latest_ma_value is None:
        return "N/A"

    distance = safe_pct_distance(latest_close_value, latest_ma_value)
    if distance is not None and distance > extended_threshold_pct:
        return f"Extended Above {label}"
    if abs(distance or 0) <= 1:
        return f"Testing {label}"
    if prior_close_value is not None and prior_ma_value is not None:
        was_above = prior_close_value > prior_ma_value
        is_above = latest_close_value > latest_ma_value
        if not was_above and is_above:
            return f"Reclaimed {label}"
        if was_above and not is_above:
            return f"Lost {label}"
    if latest_close_value > latest_ma_value:
        return f"Above {label}"
    return f"Lost {label}"


def classify_pullback_quality(
    latest_close: object,
    latest_low: object,
    ema8: object,
    ema21: object,
    sma50: object,
    sma200: object,
    ema21_trend: str,
    extension_status: str,
) -> str:
    close_value = safe_number(latest_close)
    low_value = safe_number(latest_low)
    ema8_value = safe_number(ema8)
    ema21_value = safe_number(ema21)
    sma50_value = safe_number(sma50)
    sma200_value = safe_number(sma200)
    if close_value is None or ema21_value is None:
        return "N/A"
    if extension_status in {"Extended", "Too Extended"}:
        return "Too Extended"
    if close_value < ema21_value:
        return "Lost EMA21"
    above_larger_trend = sma50_value is not None and close_value > sma50_value and (sma200_value is None or close_value > sma200_value)
    if low_value is not None and ema8_value is not None and low_value <= ema8_value <= close_value and close_value > ema21_value:
        return "Clean EMA8 Pullback"
    if low_value is not None and low_value <= ema21_value <= close_value and above_larger_trend and ema21_trend == "Rising":
        return "Clean EMA21 Pullback"
    ema21_distance = safe_pct_distance(close_value, ema21_value)
    if ema21_distance is not None and 0 <= ema21_distance <= 3 and ema21_trend == "Rising":
        return "Holding EMA21"
    return "N/A"


def classify_long_status(
    price: object,
    ema8: object,
    ema21: object,
    sma50: object,
    sma200: object,
    ema_stack_status: str,
    extension_status: str,
    pullback_quality: str,
    ema21_trend: str,
    sma50_trend: str,
    volume_ratio: object = None,
) -> str:
    price_value = safe_number(price)
    ema8_value = safe_number(ema8)
    ema21_value = safe_number(ema21)
    sma50_value = safe_number(sma50)
    sma200_value = safe_number(sma200)
    if price_value is None or ema21_value is None or sma50_value is None:
        return "Insufficient Data"

    above_ema8 = ema8_value is not None and price_value > ema8_value
    above_ema21 = price_value > ema21_value
    above_sma50 = price_value > sma50_value
    above_sma200 = sma200_value is not None and price_value > sma200_value
    volume_ok = safe_number(volume_ratio)
    volume_ok = volume_ok is None or volume_ok >= 0.8

    if not above_sma50 and (sma200_value is None or not above_sma200) and ema21_trend == "Falling":
        return "Weak / Avoid"
    if not above_ema21 or not above_sma50:
        return "Below Key MAs"
    if above_ema21 and above_sma50 and above_sma200 and extension_status == "Too Extended":
        return "Too Extended"
    if pullback_quality in {"Clean EMA8 Pullback", "Clean EMA21 Pullback", "Holding EMA21"} and above_sma50 and above_sma200 and ema21_trend == "Rising":
        return "Pullback Entry Setup"
    if (
        above_ema8
        and above_ema21
        and above_sma50
        and above_sma200
        and ema_stack_status == "Perfect Bullish Stack"
        and ema21_trend == "Rising"
        and sma50_trend == "Rising"
        and extension_status not in {"Extended", "Too Extended"}
        and volume_ok
    ):
        return "Strong Long Setup"
    return "Long Watchlist"


def build_long_summary(status: str) -> str:
    if status == "Strong Long Setup":
        return "Strong long candidate: price is above EMA21, SMA50, and SMA200 with rising trend structure."
    if status == "Long Watchlist":
        return "Watchlist only: price is above EMA21 but the moving-average stack is not fully confirmed."
    if status == "Pullback Entry Setup":
        return "Pullback setup: price is holding near EMA8 or EMA21 while the larger trend remains bullish."
    if status == "Too Extended":
        return "Too extended: trend is strong, but price is stretched above the moving averages. Wait for consolidation or pullback."
    if status == "Below Key MAs":
        return "Avoid for now: price is below EMA21 or SMA50, so this is not a clean long setup."
    if status == "Insufficient Data":
        return "Insufficient data: not enough moving-average history to classify the long setup."
    return "Weak / avoid: trend structure is not supportive for a long-only setup."


def build_long_trend_snapshot(history: pd.DataFrame, price: object, volume_ratio: object = None) -> dict:
    if history is None:
        history = pd.DataFrame()
    close = history["Close"] if not history.empty and "Close" in history.columns else pd.Series(dtype=float)
    low = history["Low"] if not history.empty and "Low" in history.columns else pd.Series(dtype=float)
    ema8_series = close.ewm(span=8, adjust=False, min_periods=8).mean() if len(close.index) >= 8 else pd.Series(dtype=float)
    ema21_series = close.ewm(span=21, adjust=False, min_periods=21).mean() if len(close.index) >= 21 else pd.Series(dtype=float)
    sma50_series = close.rolling(50).mean() if len(close.index) >= 50 else pd.Series(dtype=float)
    sma200_series = close.rolling(200).mean() if len(close.index) >= 200 else pd.Series(dtype=float)
    ema8 = get_latest_value(ema8_series)
    ema21 = get_latest_value(ema21_series)
    sma50 = get_latest_value(sma50_series)
    sma200 = get_latest_value(sma200_series)
    prior_close = get_previous_value(close)
    latest_low = get_latest_value(low)
    ema21_trend = detect_ma_trend(ema21_series)
    ema8_trend = detect_ma_trend(ema8_series)
    sma50_trend = detect_ma_trend(sma50_series)
    sma200_trend = detect_ma_trend(sma200_series)
    distance_from_ema21 = safe_pct_distance(price, ema21)
    ema_stack_status = classify_ema_stack(price, ema8, ema21, sma50, sma200, ema21_trend)
    extension_status = classify_extension_status(distance_from_ema21)
    ema8_status = classify_ma_position_status(price, prior_close, ema8, get_previous_value(ema8_series), "EMA8", 6)
    ema21_status = classify_ma_position_status(price, prior_close, ema21, get_previous_value(ema21_series), "EMA21", 12)
    pullback_quality = classify_pullback_quality(price, latest_low, ema8, ema21, sma50, sma200, ema21_trend, extension_status)
    long_status = classify_long_status(
        price,
        ema8,
        ema21,
        sma50,
        sma200,
        ema_stack_status,
        extension_status,
        pullback_quality,
        ema21_trend,
        sma50_trend,
        volume_ratio,
    )
    return {
        "ema8": ema8,
        "ema21": ema21,
        "sma50": sma50,
        "sma200": sma200,
        "distance_from_ema8": safe_pct_distance(price, ema8),
        "distance_from_ema21": distance_from_ema21,
        "distance_from_sma50": safe_pct_distance(price, sma50),
        "distance_from_sma200": safe_pct_distance(price, sma200),
        "ema8_trend": ema8_trend,
        "ema21_trend": ema21_trend,
        "sma50_trend": sma50_trend,
        "sma200_trend": sma200_trend,
        "ema_stack_status": ema_stack_status,
        "ema8_status": ema8_status,
        "ema21_status": ema21_status,
        "extension_status": extension_status,
        "pullback_quality": pullback_quality,
        "long_status": long_status,
        "long_summary": build_long_summary(long_status),
    }


def get_atr_pct(history: pd.DataFrame, length: int = 14) -> float | None:
    if history.empty or len(history.index) < length + 1:
        return None

    previous_close = history["Close"].shift(1)
    true_range = pd.concat(
        [
            history["High"] - history["Low"],
            (history["High"] - previous_close).abs(),
            (history["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(length).mean().iloc[-1]
    close = history["Close"].iloc[-1]
    if pd.isna(atr) or pd.isna(close) or close == 0:
        return None
    return float((atr / close) * 100)


def safe_number(value: object) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _snapshot_value(row: dict, *keys: str) -> object:
    for key in keys:
        value = row.get(key)
        if safe_number(value) is not None:
            return value
    return None


def _pct_distance(price: object, level: object) -> float | None:
    price_value = safe_number(price)
    level_value = safe_number(level)
    if price_value is None or level_value in (None, 0):
        return None
    return ((price_value / level_value) - 1) * 100


def _clean_ohlcv_frame(history: pd.DataFrame | None) -> pd.DataFrame:
    if history is None or history.empty:
        return pd.DataFrame()
    required_columns = ["High", "Low", "Close"]
    if any(column not in history.columns for column in required_columns):
        return pd.DataFrame()
    frame = history.copy()
    for column in ("High", "Low", "Close", "Volume"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["High", "Low", "Close"]).copy()


def _atr_pct_series(frame: pd.DataFrame, length: int = 14) -> pd.Series:
    previous_close = frame["Close"].shift(1)
    true_range = pd.concat(
        [
            frame["High"] - frame["Low"],
            (frame["High"] - previous_close).abs(),
            (frame["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(length).mean()
    return (atr / frame["Close"]) * 100


def build_breakout_snapshot_metrics(history: pd.DataFrame | None, price: object) -> dict:
    frame = _clean_ohlcv_frame(history)
    result = {
        "high_20d": None,
        "high_50d": None,
        "prior_high_20d": None,
        "prior_high_50d": None,
        "distance_to_20d_high": None,
        "distance_to_50d_high": None,
        "breakout_level_20d": None,
        "breakout_level_50d": None,
        "days_since_breakout_20d": None,
        "days_since_breakout_50d": None,
        "breakout_event_level_20d": None,
        "breakout_event_level_50d": None,
        "distance_to_breakout_event_20d": None,
        "distance_to_breakout_event_50d": None,
        "breakout_pct_20d": None,
        "breakout_pct_50d": None,
        "latest_close_position_pct": None,
        "higher_lows": None,
        "volatility_compression": None,
        "range_tightening": None,
        "atr_pct_vs_average": None,
    }
    if frame.empty:
        return result

    current_price = safe_number(price) or safe_number(frame["Close"].iloc[-1])
    latest_high = safe_number(frame["High"].iloc[-1])
    latest_low = safe_number(frame["Low"].iloc[-1])
    latest_close = safe_number(frame["Close"].iloc[-1])
    if latest_high is not None and latest_low is not None and latest_close is not None and latest_high > latest_low:
        result["latest_close_position_pct"] = ((latest_close - latest_low) / (latest_high - latest_low)) * 100

    atr_pct = _atr_pct_series(frame)
    latest_atr_pct = safe_number(atr_pct.iloc[-1]) if not atr_pct.empty else None
    average_atr_pct = safe_number(atr_pct.rolling(50).mean().iloc[-1]) if len(atr_pct.index) >= 50 else None
    if latest_atr_pct is not None and average_atr_pct not in (None, 0):
        result["atr_pct_vs_average"] = (latest_atr_pct / average_atr_pct) * 100
        result["volatility_compression"] = latest_atr_pct < average_atr_pct

    if len(frame.index) >= 25:
        recent_range = frame["High"].iloc[-10:].max() - frame["Low"].iloc[-10:].min()
        prior_range = frame["High"].iloc[-20:-10].max() - frame["Low"].iloc[-20:-10].min()
        if safe_number(prior_range) not in (None, 0):
            result["range_tightening"] = bool(recent_range < prior_range)
            result["volatility_compression"] = bool(result["volatility_compression"] or result["range_tightening"])

    if len(frame.index) >= 30:
        recent_low = safe_number(frame["Low"].iloc[-10:].min())
        prior_low = safe_number(frame["Low"].iloc[-30:-10].min())
        result["higher_lows"] = bool(recent_low is not None and prior_low is not None and recent_low > prior_low)

    for days in (20, 50):
        high_series = frame["High"].rolling(days).max()
        prior_high_series = frame["High"].shift(1).rolling(days).max()
        high_value = safe_number(high_series.iloc[-1]) if len(high_series.index) >= days else None
        prior_high_value = safe_number(prior_high_series.iloc[-1]) if len(prior_high_series.index) >= days + 1 else None
        result[f"high_{days}d"] = high_value
        result[f"prior_high_{days}d"] = prior_high_value
        result[f"breakout_level_{days}d"] = prior_high_value
        result[f"distance_to_{days}d_high"] = _pct_distance(current_price, prior_high_value)

        if prior_high_value is None:
            continue
        breakout_mask = frame["Close"] > prior_high_series
        breakout_positions = [frame.index.get_loc(index_value) for index_value in frame.index[breakout_mask.fillna(False)]]
        if not breakout_positions:
            continue
        last_position = breakout_positions[-1]
        event_level = safe_number(prior_high_series.iloc[last_position])
        event_close = safe_number(frame["Close"].iloc[last_position])
        result[f"days_since_breakout_{days}d"] = len(frame.index) - last_position - 1
        result[f"breakout_event_level_{days}d"] = event_level
        result[f"distance_to_breakout_event_{days}d"] = _pct_distance(current_price, event_level)
        result[f"breakout_pct_{days}d"] = _pct_distance(event_close, event_level)

    return result


def get_prefilter_setup_score(history: pd.DataFrame, price: float) -> float:
    if history.empty or len(history.index) < 50 or "Close" not in history.columns:
        return 0.0

    closes = history["Close"].dropna()
    if len(closes.index) < 50:
        return 0.0

    long_trend = build_long_trend_snapshot(history, price)
    status = long_trend.get("long_status")
    status_score = {
        "Strong Long Setup": 100,
        "Long Watchlist": 75,
        "Too Extended": 60,
        "Below Key MAs": 25,
        "Weak / Avoid": 0,
        "Insufficient Data": 0,
    }.get(status, 0)
    if long_trend.get("ema21_trend") == "Rising":
        status_score += 5
    if long_trend.get("sma50_trend") == "Rising":
        status_score += 5
    return min(status_score, 100)


def get_max_ma_extension_pct(history: pd.DataFrame, price: float) -> float | None:
    if history.empty or "Close" not in history.columns:
        return None

    closes = history["Close"].dropna()
    if len(closes.index) < 21:
        return None

    moving_averages = [
        safe_number(closes.ewm(span=21, adjust=False, min_periods=21).mean().iloc[-1]),
        safe_number(closes.rolling(50).mean().iloc[-1]) if len(closes.index) >= 50 else None,
        safe_number(closes.rolling(200).mean().iloc[-1]) if len(closes.index) >= 200 else None,
    ]
    extensions = [
        ((price - moving_average) / moving_average) * 100
        for moving_average in moving_averages
        if moving_average not in (None, 0) and price > moving_average
    ]
    if not extensions:
        return None
    return max(extensions)


def has_minimum_scan_history(stock_data: dict, min_history_bars: int = 200) -> bool:
    history = stock_data.get("history", pd.DataFrame())
    return not history.empty and len(history.index) >= min_history_bars


def passes_prefilter(
    stock_data: dict,
    config: dict,
    scan_mode: str | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    min_avg_volume: float | None = None,
    min_dollar_volume: float | None = None,
    min_history_bars: int = 200,
) -> tuple[bool, float, str]:
    history = stock_data.get("history", pd.DataFrame())
    if history.empty or len(history.index) < min_history_bars:
        return False, 0.0, "insufficient_history"

    price = safe_number(stock_data.get("price"))
    if price is None:
        return False, 0.0, "missing_price"

    min_price_value = min_price if min_price is not None else config.get("min_price")
    min_price_value = 0.0 if min_price_value is None else min_price_value
    if price < min_price_value:
        return False, 0.0, "price"
    if max_price is not None and price > max_price:
        return False, 0.0, "price"

    avg_volume = safe_number(stock_data.get("avg_volume_20")) or 0.0
    if min_avg_volume is not None and avg_volume < min_avg_volume:
        return False, 0.0, "avg_volume"

    min_addv_value = min_dollar_volume if min_dollar_volume is not None else config.get("min_addv")
    min_addv_value = 0.0 if min_addv_value is None else min_addv_value
    addv = safe_number(stock_data.get("addv_20")) or 0.0
    if addv < min_addv_value:
        return False, 0.0, "dollar_volume"

    min_market_cap = config.get("min_market_cap")
    market_cap = safe_number(stock_data.get("market_cap"))
    if min_market_cap is not None and market_cap is not None and market_cap < min_market_cap:
        return False, 0.0, "market_cap"

    max_atr_pct = config.get("max_atr_pct")
    atr_pct = get_atr_pct(history)
    if max_atr_pct is not None and atr_pct is not None and atr_pct > max_atr_pct:
        return False, 0.0, "atr"

    mode_config = get_scan_mode_config(scan_mode)
    setup_score = get_prefilter_setup_score(history, price)
    min_setup_score = mode_config.get("min_setup_score")
    if min_setup_score is not None and setup_score < float(min_setup_score):
        return False, setup_score, "weak_setup"

    max_extension_pct = mode_config.get("max_extension_pct")
    extension_pct = get_max_ma_extension_pct(history, price)
    if max_extension_pct is not None and extension_pct is not None and extension_pct > float(max_extension_pct):
        return False, setup_score, "extended"

    liquidity_score = min(addv / max(float(min_addv_value or 1), 1.0), 4.0) * 8
    volume_ratio = safe_number(stock_data.get("volume_ratio")) or 0.0
    prefilter_score = setup_score + liquidity_score + min(volume_ratio, 3.0) * 5
    return True, prefilter_score, ""


def prefilter_stock_data(
    stock_data_map: dict[str, dict],
    tickers: list[str],
    config: dict,
    candidate_limit: int | None,
    scan_mode: str | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    min_avg_volume: float | None = None,
    min_dollar_volume: float | None = None,
) -> tuple[list[tuple[str, dict]], dict]:
    candidates = []
    skip_reasons: dict[str, int] = {}

    for ticker in tickers:
        stock_data = stock_data_map.get(ticker)
        if not stock_data:
            skip_reasons["missing_data"] = skip_reasons.get("missing_data", 0) + 1
            continue
        passed, prefilter_score, reason = passes_prefilter(
            stock_data=stock_data,
            config=config,
            scan_mode=scan_mode,
            min_price=min_price,
            max_price=max_price,
            min_avg_volume=min_avg_volume,
            min_dollar_volume=min_dollar_volume,
        )
        if not passed:
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            continue
        candidates.append((ticker, stock_data, prefilter_score))

    candidates.sort(
        key=lambda item: (
            item[2],
            safe_number(item[1].get("addv_20")) or 0.0,
            safe_number(item[1].get("volume_ratio")) or 0.0,
        ),
        reverse=True,
    )
    if candidate_limit is None:
        limited_candidates = candidates
    else:
        limited_candidates = candidates[: max(int(candidate_limit), 0)]
    limited = [(ticker, stock_data) for ticker, stock_data, _score in limited_candidates]
    return limited, {
        "valid_data_count": len(stock_data_map),
        "prefilter_pass_count": len(candidates),
        "prefilter_scored_count": len(limited),
        "prefilter_skip_reasons": skip_reasons,
    }


def select_full_scan_candidates(
    stock_data_map: dict[str, dict],
    tickers: list[str],
    candidate_limit: int | None = None,
) -> tuple[list[tuple[str, dict]], dict]:
    candidates = []
    skip_reasons: dict[str, int] = {}

    for ticker in tickers:
        stock_data = stock_data_map.get(ticker)
        if not stock_data:
            skip_reasons["missing_data"] = skip_reasons.get("missing_data", 0) + 1
            continue
        if not has_minimum_scan_history(stock_data):
            skip_reasons["insufficient_history"] = skip_reasons.get("insufficient_history", 0) + 1
            continue
        price = safe_number(stock_data.get("price"))
        if price is None:
            skip_reasons["missing_price"] = skip_reasons.get("missing_price", 0) + 1
            continue
        prefilter_score = get_prefilter_setup_score(stock_data.get("history", pd.DataFrame()), price)
        candidates.append((ticker, stock_data, prefilter_score))

    candidates.sort(
        key=lambda item: (
            item[2],
            safe_number(item[1].get("addv_20")) or 0.0,
            safe_number(item[1].get("volume_ratio")) or 0.0,
        ),
        reverse=True,
    )
    valid_candidate_count = len(candidates)
    if candidate_limit is not None:
        candidates = candidates[: max(int(candidate_limit), 0)]

    return [(ticker, stock_data) for ticker, stock_data, _score in candidates], {
        "valid_data_count": len(stock_data_map),
        "prefilter_pass_count": valid_candidate_count,
        "prefilter_scored_count": len(candidates),
        "prefilter_skip_reasons": skip_reasons,
    }


def select_candidates_for_full_scoring(
    stock_data_map: dict[str, dict],
    tickers: list[str],
    config: dict,
    scan_mode: str | None,
    candidate_limit: int | None,
    min_price: float | None = None,
    max_price: float | None = None,
    min_avg_volume: float | None = None,
    min_dollar_volume: float | None = None,
    bypass_prefilter: bool = False,
) -> tuple[list[tuple[str, dict]], dict]:
    if bypass_prefilter or scan_mode == "Full Scan":
        return select_full_scan_candidates(
            stock_data_map=stock_data_map,
            tickers=tickers,
            candidate_limit=candidate_limit,
        )

    return prefilter_stock_data(
        stock_data_map=stock_data_map,
        tickers=tickers,
        config=config,
        candidate_limit=candidate_limit,
        scan_mode=scan_mode,
        min_price=min_price,
        max_price=max_price,
        min_avg_volume=min_avg_volume,
        min_dollar_volume=min_dollar_volume,
    )


def passes_filters(stock_data: dict, config: dict) -> bool:
    if stock_data.get("price", 0) < config.get("min_price", 0):
        return False
    if stock_data.get("addv_20", 0) < config.get("min_addv", 0):
        return False
    max_atr_pct = config.get("max_atr_pct")
    if max_atr_pct is not None:
        atr_pct = get_atr_pct(stock_data.get("history", pd.DataFrame()))
        if atr_pct is not None and atr_pct > max_atr_pct:
            return False
    return True


def split_results(ranked_rows: list[dict], min_score: int, max_results: int) -> dict:
    limited_rows = ranked_rows[:max_results]
    top_setups = []
    for row in ranked_rows:
        if row.get("long_status") != "Strong Long Setup":
            continue
        top_setups.append(row)
        if len(top_setups) >= min(10, max_results):
            break

    return {
        "top_setups": top_setups,
        "all_ranked": limited_rows,
    }


def filter_setup_mode(rows: list[dict], setup_mode: str | None) -> list[dict]:
    if not setup_mode or setup_mode == "Main Trend Scanner":
        return rows
    if setup_mode == "A+ Setup Mode":
        return [row for row in rows if bool(row.get("a_plus_qualified")) or row.get("a_plus_status") == "A+ Candidate"]
    if setup_mode == "Breakout Retests":
        return [row for row in rows if row.get("setup_type") == "Breakout Retest"]
    if setup_mode == "Pre-Breakout Watchlist":
        return [row for row in rows if row.get("setup_type") in {"Pre-Breakout Base", "Compression Squeeze"}]
    if setup_mode == "Fresh Breakouts":
        return [row for row in rows if row.get("setup_type") == "Fresh Breakout"]
    return rows


def long_status_rank(status: str | None) -> int:
    return {
        "Strong Long Setup": 0,
        "Pullback Entry Setup": 1,
        "Long Watchlist": 2,
        "Too Extended": 3,
        "Below Key MAs": 4,
        "Weak / Avoid": 5,
        "Insufficient Data": 6,
    }.get(status or "", 6)


def sort_long_rows(rows: list[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: (
            long_status_rank(row.get("long_status")),
            -(safe_number(row.get("setup_signal_score")) or safe_number(row.get("signal_score")) or 0),
            -(safe_number(row.get("volume_ratio")) or 0),
            abs(safe_number(row.get("distance_from_ema21")) or 999),
        ),
    )


def _passes_minimum(row: dict, field: str, minimum: object) -> bool:
    value = safe_number(row.get(field))
    threshold = safe_number(minimum)
    return value is not None and threshold is not None and value >= threshold


def _passes_maximum(row: dict, field: str, maximum: object) -> bool:
    value = safe_number(row.get(field))
    threshold = safe_number(maximum)
    return value is not None and threshold is not None and value <= threshold


def calculate_trend_status(row: dict) -> str:
    price = safe_number(row.get("price") if row.get("price") is not None else row.get("latest_price"))
    ema8 = safe_number(row.get("ema8"))
    ema21 = safe_number(row.get("ema21"))
    sma50 = safe_number(row.get("sma50"))
    sma200 = safe_number(row.get("sma200"))
    trend_quality = safe_number(row.get("trend_quality"))
    sma50_trend = snapshot_text(row.get("sma50_trend"))
    sma200_trend = snapshot_text(row.get("sma200_trend"))

    if price is None or ema21 is None or sma50 is None or sma200 is None:
        return "Insufficient Data"

    sma50_rising_or_flat = sma50_trend in {"Rising", "Flat", None}
    sma50_rising = sma50_trend == "Rising"
    sma200_rising = sma200_trend == "Rising"
    sma200_falling = sma200_trend == "Falling"
    if trend_quality is None:
        trend_quality = 0.0
        trend_quality += 3 if price > ema21 else 0
        trend_quality += 3 if price > sma50 else 0
        trend_quality += 3 if price > sma200 else 0
        trend_quality += 2 if ema8 is not None and ema8 > ema21 else 0
        trend_quality += 2 if sma50_rising else 0
        trend_quality += 2 if sma200_rising else 0
    trend_quality_value = trend_quality

    if price < sma200 and (sma50 < sma200 or sma200_falling or trend_quality_value < 5):
        return "Downtrend / Avoid"

    if (
        ema8 is not None
        and price > ema8
        and ema8 > ema21
        and ema21 > sma50
        and price > sma50
        and price > sma200
        and sma50_rising
        and sma200_rising
        and trend_quality_value >= 13
    ):
        return "Strong Uptrend"

    if (
        price > ema21
        and price > sma50
        and price > sma200
        and sma50_rising_or_flat
        and 10 <= trend_quality_value < 13
    ):
        return "Constructive Uptrend"

    near_ema21 = abs(safe_number(row.get("distance_from_ema21")) or 999) <= 3
    near_sma50 = abs(_pct_distance(price, sma50) or 999) <= 3
    if price > sma200 and (near_ema21 or near_sma50 or 7 <= trend_quality_value < 10):
        return "Watchlist"

    if 5 <= trend_quality_value < 7:
        return "Neutral / Mixed"

    if price < ema21 or price < sma50 or trend_quality_value < 5:
        return "Weak / Avoid"

    return "Neutral / Mixed"


def get_display_trend_status(row: dict) -> str:
    return snapshot_text(row.get("trend_status")) or calculate_trend_status(row)


def apply_optional_filters(rows: list[dict], filters: dict | None) -> tuple[list[dict], int]:
    if not filters:
        return rows, 0

    use_trend_status_filter = bool(filters.get("use_trend_status_filter"))
    selected_trend_statuses = {
        str(status).strip()
        for status in (filters.get("trend_status_filter_values") or [])
        if str(status).strip()
    }

    filtered_rows = []
    skipped_count = 0
    for row in rows:
        if use_trend_status_filter:
            if not selected_trend_statuses:
                skipped_count += 1
                continue
            if get_display_trend_status(row) not in selected_trend_statuses:
                skipped_count += 1
                continue

        if filters.get("use_price_filter"):
            if not _passes_minimum(row, "price", filters.get("min_price")):
                skipped_count += 1
                continue
            if not _passes_maximum(row, "price", filters.get("max_price")):
                skipped_count += 1
                continue

        if filters.get("use_volume_filter") and not _passes_minimum(row, "avg_volume", filters.get("min_avg_volume")):
            skipped_count += 1
            continue

        if filters.get("use_relative_volume_filter") and not _passes_minimum(row, "volume_ratio", filters.get("min_relative_volume")):
            skipped_count += 1
            continue

        # TODO: Add market-cap filtering when market cap is stored in scanner snapshot rows.
        if filters.get("use_market_cap_filter"):
            if not _passes_minimum(row, "market_cap", filters.get("min_market_cap")):
                skipped_count += 1
                continue
            if not _passes_maximum(row, "market_cap", filters.get("max_market_cap")):
                skipped_count += 1
                continue

        if filters.get("above_ema21") and not _passes_minimum(row, "distance_from_ema21", 0):
            skipped_count += 1
            continue
        if filters.get("above_sma50") and not _passes_minimum(row, "distance_from_sma50", 0):
            skipped_count += 1
            continue
        if filters.get("above_sma200") and not _passes_minimum(row, "distance_from_sma200", 0):
            skipped_count += 1
            continue

        filtered_rows.append(row)

    return filtered_rows, skipped_count


def apply_display_filters(
    scored_rows: list[dict],
    config: dict,
    min_score: int,
    max_results: int,
    min_price: float | None = None,
    max_price: float | None = None,
    min_avg_volume: float | None = None,
    min_dollar_volume: float | None = None,
    optional_filters: dict | None = None,
) -> tuple[list[dict], int, int]:
    legacy_filters = optional_filters
    if legacy_filters is None:
        legacy_filters = {
            "use_price_filter": min_price is not None or max_price is not None,
            "min_price": min_price,
            "max_price": max_price,
            "use_volume_filter": min_avg_volume is not None,
            "min_avg_volume": min_avg_volume,
            "use_relative_volume_filter": False,
            "min_relative_volume": None,
            "above_ema21": False,
            "above_sma50": False,
            "above_sma200": False,
            "use_trend_status_filter": False,
            "trend_status_filter_values": [],
        }
    filtered_rows, filter_skip_count = apply_optional_filters(scored_rows, legacy_filters)
    filtered_rows = filter_setup_mode(filtered_rows, legacy_filters.get("setup_mode"))

    return sort_long_rows(filtered_rows)[:max_results], filter_skip_count, len(filtered_rows)


def _breakout_lookback_days(resistance_lookback: str) -> tuple[int, ...]:
    if resistance_lookback == "20D high":
        return (20,)
    if resistance_lookback == "50D high":
        return (50,)
    return (20, 50)


def _breakout_trend_score(row: dict) -> int:
    return BREAKOUT_TREND_RANK.get(get_display_trend_status(row), 0)


def _breakout_quality_score(row: dict, mode: str, distance_to_breakout: float | None) -> int:
    score = 0.0
    trend_score = _breakout_trend_score(row)
    score += min(trend_score / 4, 1) * 25

    if distance_to_breakout is not None:
        if mode == "Pre-Breakout Setup":
            proximity_score = max(0.0, 1 - (abs(distance_to_breakout) / 5))
        elif mode == "Fresh Breakout":
            proximity_score = max(0.0, min(distance_to_breakout / 6, 1))
        else:
            proximity_score = 1.0 if distance_to_breakout >= 0 else max(0.0, 1 - (abs(distance_to_breakout) / 3))
        score += proximity_score * 25

    volume_ratio = safe_number(row.get("volume_ratio"))
    if volume_ratio is not None:
        score += min(volume_ratio / 1.5, 1) * 20

    extension = safe_number(row.get("distance_from_ema21"))
    if extension is not None:
        if extension <= 0:
            extension_score = 0
        elif extension <= 8:
            extension_score = 1
        elif extension <= 15:
            extension_score = max(0.0, 1 - ((extension - 8) / 7))
        else:
            extension_score = 0
        score += extension_score * 15

    base_points = 0
    if bool(row.get("volatility_compression")):
        base_points += 8
    if bool(row.get("higher_lows")):
        base_points += 7
    if safe_number(row.get("latest_close_position_pct")) is not None and safe_number(row.get("latest_close_position_pct")) >= 50:
        base_points += 3
    score += min(base_points, 15)
    return int(round(max(0, min(score, 100))))


def _build_breakout_candidate(row: dict, mode: str, lookback_days: int, breakout_level: float, distance_to_breakout: float | None) -> dict:
    return {
        "ticker": row.get("ticker", "N/A"),
        "company": row.get("company") or row.get("company_name"),
        "price": row.get("latest_price", row.get("price")),
        "ema8": row.get("ema8"),
        "ema21": row.get("ema21"),
        "sma50": row.get("sma50"),
        "sma100": row.get("sma100"),
        "sma200": row.get("sma200"),
        "high_20d": row.get("high_20d"),
        "high_50d": row.get("high_50d"),
        "prior_high_20d": row.get("prior_high_20d"),
        "prior_high_50d": row.get("prior_high_50d"),
        "distance_to_20d_high": row.get("distance_to_20d_high"),
        "distance_to_50d_high": row.get("distance_to_50d_high"),
        "breakout_mode_match": mode,
        "breakout_level": breakout_level,
        "breakout_type": f"{lookback_days}D high",
        "distance_to_breakout": distance_to_breakout,
        "days_since_breakout": row.get(f"days_since_breakout_{lookback_days}d"),
        "relative_volume": row.get("volume_ratio"),
        "volume_ratio": row.get("volume_ratio"),
        "atr_pct": row.get("atr_pct"),
        "distance_from_ema21": row.get("distance_from_ema21"),
        "distance_from_sma50": row.get("distance_from_sma50"),
        "distance_from_sma200": row.get("distance_from_sma200"),
        "trend_status": get_display_trend_status(row),
        "long_status": row.get("long_status"),
        "breakout_quality": _breakout_quality_score(row, mode, distance_to_breakout),
        "volatility_compression": row.get("volatility_compression"),
        "higher_lows": row.get("higher_lows"),
    }


def _passes_breakout_base_rules(row: dict, max_distance_above_ema21_pct: float, use_relative_volume_filter: bool, min_relative_volume: float) -> bool:
    if not _passes_minimum(row, "distance_from_ema21", 0):
        return False
    if not _passes_minimum(row, "distance_from_sma50", 0):
        return False
    if not _passes_maximum(row, "distance_from_ema21", max_distance_above_ema21_pct):
        return False
    if use_relative_volume_filter and not _passes_minimum(row, "volume_ratio", min_relative_volume):
        return False
    return True


def run_breakout_scan_from_snapshot(
    snapshot_df: pd.DataFrame,
    mode: str = "Pre-Breakout Setup",
    resistance_lookback: str = "Both",
    max_distance_below_pct: float = 5.0,
    fresh_breakout_window_days: int = 5,
    max_distance_above_ema21_pct: float = 12.0,
    use_relative_volume_filter: bool = False,
    min_relative_volume: float = 1.2,
    max_results: int = 25,
    optional_filters: dict | None = None,
) -> list[dict]:
    if snapshot_df is None or snapshot_df.empty:
        return []

    rows = snapshot_df_to_rows(snapshot_df)
    raw_by_ticker = {
        str(row.get("ticker", "")).upper(): row
        for row in snapshot_df.to_dict("records")
    }
    enriched_rows = []
    for row in rows:
        raw_row = raw_by_ticker.get(str(row.get("ticker", "")).upper(), {})
        enriched_rows.append({**raw_row, **row})

    filtered_rows, _ = apply_optional_filters(enriched_rows, optional_filters)
    candidates = []
    for row in filtered_rows:
        if not _passes_breakout_base_rules(row, max_distance_above_ema21_pct, use_relative_volume_filter, min_relative_volume):
            continue

        ema21 = safe_number(row.get("ema21"))
        sma50 = safe_number(row.get("sma50"))
        if mode == "Pre-Breakout Setup" and ema21 is not None and sma50 is not None and ema21 < sma50 * 0.98:
            continue

        row_candidates = []
        for days in _breakout_lookback_days(resistance_lookback):
            breakout_level = safe_number(row.get(f"breakout_level_{days}d") or row.get(f"prior_high_{days}d"))
            if breakout_level is None:
                continue
            distance_to_breakout = safe_number(row.get(f"distance_to_{days}d_high"))
            if distance_to_breakout is None:
                distance_to_breakout = _pct_distance(row.get("price"), breakout_level)
            days_since_breakout = safe_number(row.get(f"days_since_breakout_{days}d"))
            event_level = safe_number(row.get(f"breakout_event_level_{days}d"))
            distance_to_event_level = safe_number(row.get(f"distance_to_breakout_event_{days}d"))

            if mode == "Pre-Breakout Setup":
                if distance_to_breakout is None or distance_to_breakout > 0 or distance_to_breakout < -max_distance_below_pct:
                    continue
            elif mode == "Fresh Breakout":
                if event_level is not None:
                    breakout_level = event_level
                    distance_to_breakout = distance_to_event_level
                if days_since_breakout is None or days_since_breakout > fresh_breakout_window_days or distance_to_breakout is None or distance_to_breakout < 0:
                    continue
            else:
                if event_level is not None:
                    breakout_level = event_level
                    distance_to_breakout = distance_to_event_level
                if days_since_breakout is None or days_since_breakout < 5 or days_since_breakout > 20:
                    continue
                if distance_to_breakout is None or distance_to_breakout < -3:
                    continue
                if get_display_trend_status(row) not in {"Watchlist", "Constructive Uptrend", "Strong Uptrend"}:
                    continue

            row_candidates.append(_build_breakout_candidate(row, mode, days, breakout_level, distance_to_breakout))

        if not row_candidates:
            continue
        if mode == "Pre-Breakout Setup":
            best_candidate = sorted(row_candidates, key=lambda item: abs(safe_number(item.get("distance_to_breakout")) or 999))[0]
        elif mode == "Fresh Breakout":
            best_candidate = sorted(row_candidates, key=lambda item: (safe_number(item.get("days_since_breakout")) or 999, -(safe_number(item.get("distance_to_breakout")) or 0)))[0]
        else:
            best_candidate = sorted(row_candidates, key=lambda item: (abs(min(safe_number(item.get("distance_to_breakout")) or 0, 0)), -(safe_number(item.get("distance_to_breakout")) or 0)))[0]
        candidates.append(best_candidate)

    if mode == "Pre-Breakout Setup":
        candidates = sorted(
            candidates,
            key=lambda row: (
                abs(safe_number(row.get("distance_to_breakout")) or 999),
                -BREAKOUT_TREND_RANK.get(row.get("trend_status"), 0),
                -(safe_number(row.get("relative_volume")) or 0),
                abs(safe_number(row.get("distance_from_ema21")) or 999),
            ),
        )
    elif mode == "Fresh Breakout":
        candidates = sorted(
            candidates,
            key=lambda row: (
                safe_number(row.get("days_since_breakout")) if safe_number(row.get("days_since_breakout")) is not None else 999,
                -(safe_number(row.get("relative_volume")) or 0),
                -(safe_number(row.get("distance_to_breakout")) or 0),
                -BREAKOUT_TREND_RANK.get(row.get("trend_status"), 0),
            ),
        )
    else:
        candidates = sorted(
            candidates,
            key=lambda row: (
                abs(min(safe_number(row.get("distance_to_breakout")) or 0, 0)),
                -BREAKOUT_TREND_RANK.get(row.get("trend_status"), 0),
                -(safe_number(row.get("relative_volume")) or 0),
                abs(safe_number(row.get("distance_from_ema21")) or 999),
            ),
        )

    return candidates[:max_results]


def build_display_results(
    scan_payload: dict,
    universe_name: str,
    min_score: int,
    max_results: int,
    min_price: float | None = None,
    max_price: float | None = None,
    min_avg_volume: float | None = None,
    min_dollar_volume: float | None = None,
    optional_filters: dict | None = None,
) -> dict:
    start_time = perf_counter()
    config = get_universe_config(universe_name)
    ranked_rows, filter_skip_count, filtered_count = apply_display_filters(
        scored_rows=scan_payload.get("scored_rows", []),
        config=config,
        min_score=min_score,
        max_results=max_results,
        min_price=min_price,
        max_price=max_price,
        min_avg_volume=min_avg_volume,
        min_dollar_volume=min_dollar_volume,
        optional_filters=optional_filters,
    )
    base_meta = scan_payload.get("meta", {}).copy()
    timings = base_meta.get("timings", {}).copy()
    timings["filter_display_seconds"] = perf_counter() - start_time
    base_meta["timings"] = timings
    base_meta["filter_skip_count"] = filter_skip_count
    base_meta["result_count"] = filtered_count
    base_meta["duration_seconds"] = (
        timings.get("universe_load_seconds", 0)
        + timings.get("data_fetch_seconds", 0)
        + timings.get("prefilter_seconds", 0)
        + timings.get("scoring_seconds", 0)
        + timings.get("filter_display_seconds", 0)
    )

    return {
        **split_results(ranked_rows=ranked_rows, min_score=min_score, max_results=max_results),
        "meta": base_meta,
    }


def snapshot_df_to_rows(snapshot_df: pd.DataFrame) -> list[dict]:
    rows = []
    if snapshot_df.empty:
        return rows

    for row in snapshot_df.to_dict("records"):
        tags = parse_snapshot_tags(row.get("tags"))
        row_ema8_trend = snapshot_text(row.get("ema8_trend")) or "Insufficient Data"
        row_ema21_trend = snapshot_text(row.get("ema21_trend")) or "Insufficient Data"
        row_sma50_trend = snapshot_text(row.get("sma50_trend")) or "Insufficient Data"
        row_sma200_trend = snapshot_text(row.get("sma200_trend")) or "Insufficient Data"
        fallback_stack_status = snapshot_text(row.get("ema_stack_status")) or classify_ema_stack(
            row.get("latest_price", row.get("price")),
            row.get("ema8"),
            row.get("ema21"),
            row.get("sma50"),
            row.get("sma200"),
            row_ema21_trend,
        )
        fallback_extension_status = snapshot_text(row.get("extension_status")) or classify_extension_status(row.get("distance_from_ema21"))
        fallback_pullback_quality = snapshot_text(row.get("pullback_quality")) or "N/A"
        fallback_long_status = snapshot_text(row.get("long_status")) or classify_long_status(
            row.get("latest_price", row.get("price")),
            row.get("ema8"),
            row.get("ema21"),
            row.get("sma50"),
            row.get("sma200"),
            fallback_stack_status,
            fallback_extension_status,
            fallback_pullback_quality,
            row_ema21_trend,
            row_sma50_trend,
            row.get("volume_ratio"),
        )
        display_row = {
                "ticker": row.get("ticker", "N/A"),
                "company": row.get("company"),
                "exchange": row.get("exchange"),
                "price": row.get("latest_price", row.get("price")),
                "close": row.get("close", row.get("latest_price", row.get("price"))),
                "change_pct": row.get("daily_change_pct", row.get("change_pct", 0.0)),
                "volume_ratio": row.get("volume_ratio", 0.0),
                "avg_volume": row.get("avg_volume_20d", row.get("avg_volume", 0.0)),
                "addv": row.get("dollar_volume", row.get("addv", 0.0)),
                "atr_pct": row.get("atr_pct"),
                "ema8": row.get("ema8"),
                "ema21": row.get("ema21"),
                "sma50": row.get("sma50"),
                "sma100": row.get("sma100"),
                "sma200": row.get("sma200"),
                "distance_from_ema8": row.get("distance_from_ema8"),
                "distance_from_ema21": row.get("distance_from_ema21"),
                "distance_from_sma50": row.get("distance_from_sma50"),
                "distance_from_sma100": row.get("distance_from_sma100"),
                "distance_from_sma200": row.get("distance_from_sma200"),
                "ema8_trend": row_ema8_trend,
                "ema21_trend": row_ema21_trend,
                "sma50_trend": row_sma50_trend,
                "sma200_trend": row_sma200_trend,
                "ema_stack_status": fallback_stack_status,
                "ema8_status": snapshot_text(row.get("ema8_status")) or "N/A",
                "ema21_status": snapshot_text(row.get("ema21_status")) or "N/A",
                "extension_status": fallback_extension_status,
                "pullback_quality": fallback_pullback_quality,
                "long_status": fallback_long_status,
                "long_summary": row.get("long_summary") or build_long_summary(fallback_long_status),
                "signal_score": row.get("signal_score", 0.0),
                "label": row.get("setup_label", row.get("label", "Pass")),
                "base_score": row.get("base_score", 0.0),
                "trigger_score": row.get("trigger_score", 0.0),
                "follow_through_score": row.get("follow_through_score", 0.0),
                "risk_score": row.get("risk_score", 0.0),
                "historical_score": row.get("historical_score", 50.0),
                "setup_type": row.get("setup_type", "N/A"),
                "a_plus_status": row.get("a_plus_status", "N/A"),
                "a_plus_qualified": snapshot_bool(row.get("a_plus_qualified", False)),
                "setup_signal_score": row.get("setup_signal_score", row.get("signal_score", 0.0)),
                "market_regime": row.get("market_regime", "N/A"),
                "trend_quality": row.get("trend_quality"),
                "trend_status": snapshot_text(row.get("trend_status")),
                "base_quality_new": row.get("base_quality_new"),
                "compression_quality": row.get("compression_quality"),
                "trigger_quality_new": row.get("trigger_quality_new"),
                "risk_reward_quality": row.get("risk_reward_quality"),
                "historical_edge": row.get("historical_edge"),
                "historical_hit_rate": row.get("historical_hit_rate"),
                "historical_sample_size": row.get("historical_sample_size"),
                "historical_confidence_new": row.get("historical_confidence_new"),
                "historical_event_type": row.get("historical_event_type"),
                "suggested_pivot": row.get("suggested_pivot"),
                "suggested_stop": row.get("suggested_stop"),
                "risk_pct": row.get("risk_pct"),
                "reward_risk": row.get("reward_risk"),
                "date_updated": row.get("date_updated"),
                "close_strength": row.get("close_strength"),
                "a_plus_explanation": row.get("a_plus_explanation"),
                "historical_confidence": row.get("historical_confidence", "Low"),
                "penalty_score": row.get("penalty_score", 0.0),
                "tags": tags,
                "metrics": {
                    "ema_8": row.get("ema8"),
                    "ema_21": row.get("ema21"),
                    "sma50": row.get("sma50"),
                    "sma100": row.get("sma100"),
                    "sma200": row.get("sma200"),
                },
            }
        display_row["trend_status"] = snapshot_text(display_row.get("trend_status")) or calculate_trend_status(display_row)
        rows.append(display_row)
    return rows


def run_fast_scan_from_snapshot(
    snapshot_df: pd.DataFrame,
    snapshot_meta: dict,
    universe_name: str,
    min_score: int,
    max_results: int,
    min_price: float | None = None,
    max_price: float | None = None,
    min_avg_volume: float | None = None,
    min_dollar_volume: float | None = None,
    optional_filters: dict | None = None,
    progress_callback=None,
) -> dict:
    start_time = perf_counter()
    if progress_callback:
        progress_callback(0, 3, "Loading snapshot...")
    load_seconds = perf_counter() - start_time

    if progress_callback:
        progress_callback(1, 3, "Applying filters...")
    filter_start = perf_counter()
    rows = snapshot_df_to_rows(snapshot_df)
    scan_payload = {
        "scored_rows": rows,
        "meta": build_scan_meta(
            universe_count=int(snapshot_meta.get("universe_count", len(rows))),
            max_scan_cap=int(snapshot_meta.get("max_scan_cap", len(rows))),
            requested_count=int(snapshot_meta.get("requested_count", len(rows))),
            scanned_count=int(snapshot_meta.get("scanned_count", len(rows))),
            missing_data_count=int(snapshot_meta.get("missing_data_count", 0)),
            filter_skip_count=0,
            duration_seconds=0,
            result_count=len(rows),
            timings={},
        ),
    }
    scan_payload["meta"].update(
        {
            "scan_mode": snapshot_meta.get("scan_mode", "Fast Scan"),
            "tickers_attempted": int(snapshot_meta.get("tickers_attempted", snapshot_meta.get("requested_count", len(rows)))),
            "download_success_count": int(snapshot_meta.get("download_success_count", snapshot_meta.get("valid_data_count", len(rows)))),
            "invalid_ohlcv_count": int(snapshot_meta.get("invalid_ohlcv_count", 0)),
            "not_enough_history_count": int(snapshot_meta.get("not_enough_history_count", 0)),
            "usable_snapshot_count": int(snapshot_meta.get("usable_snapshot_count", len(rows))),
            "valid_data_count": int(snapshot_meta.get("valid_data_count", len(rows))),
            "prefilter_pass_count": int(snapshot_meta.get("prefilter_pass_count", len(rows))),
            "prefilter_scored_count": int(snapshot_meta.get("prefilter_scored_count", len(rows))),
            "prefilter_skip_reasons": snapshot_meta.get("prefilter_skip_reasons", {}),
            "market_regime": snapshot_meta.get("market_regime", {}),
            "calculate_historical_edge": bool(snapshot_meta.get("calculate_historical_edge", False)),
        }
    )
    results = build_display_results(
        scan_payload=scan_payload,
        universe_name=universe_name,
        min_score=min_score,
        max_results=max_results,
        min_price=min_price,
        max_price=max_price,
        min_avg_volume=min_avg_volume,
        min_dollar_volume=min_dollar_volume,
        optional_filters=optional_filters,
    )
    filter_seconds = perf_counter() - filter_start

    if progress_callback:
        progress_callback(2, 3, "Sorting results...")
    render_start = perf_counter()
    render_seconds = perf_counter() - render_start
    total_seconds = perf_counter() - start_time
    timings = results["meta"].get("timings", {})
    timings.update(
        {
            "snapshot_load_seconds": load_seconds,
            "filter_sort_seconds": filter_seconds,
            "render_prep_seconds": render_seconds,
        }
    )
    results["meta"]["timings"] = timings
    results["meta"]["duration_seconds"] = total_seconds
    results["meta"]["snapshot_last_updated"] = snapshot_meta.get("last_updated")
    results["meta"]["snapshot_tickers_available"] = len(rows)
    if progress_callback:
        progress_callback(3, 3, "Done.")
    return results


def load_snapshot_for_fast_scan(universe_name: str) -> tuple[pd.DataFrame, dict]:
    return load_scanner_snapshot(universe_name)


def refresh_scanner_snapshot(
    universe_name: str,
    max_tickers: int | None = None,
    scan_mode: str | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    min_avg_volume: float | None = None,
    min_dollar_volume: float | None = None,
    calculate_historical_edge: bool = False,
    progress_callback=None,
) -> dict:
    start_time = perf_counter()
    if progress_callback:
        progress_callback(0, 1, "Preparing universe...")

    config = get_universe_config(universe_name)
    full_market_scan = is_full_market_universe(universe_name)
    max_scan_cap = 0 if full_market_scan else (int(max_tickers) if max_tickers is not None else 0)
    universe_start = perf_counter()
    universe_tickers = get_universe(universe_name)
    tickers = universe_tickers[:max_scan_cap] if max_scan_cap else universe_tickers
    universe_seconds = perf_counter() - universe_start
    universe_symbol_meta = get_nyse_nasdaq_common_stock_meta() if full_market_scan else {}
    symbol_metadata = get_symbol_metadata_map() if full_market_scan else {}
    failed_tickers: list[str] = []
    failure_reasons: dict[str, str] = {}

    if not tickers:
        meta = {
            "universe_count": len(universe_tickers),
            "max_scan_cap": max_scan_cap,
            "requested_count": 0,
            "tickers_attempted": 0,
            "scanned_count": 0,
            "download_success_count": 0,
            "missing_data_count": 0,
            "invalid_ohlcv_count": 0,
            "not_enough_history_count": 0,
            "usable_snapshot_count": 0,
            "failed_tickers": [],
            "failure_reasons": {},
            "raw_symbols_loaded": universe_symbol_meta.get("raw_symbols_loaded"),
            "non_common_removed": universe_symbol_meta.get("non_common_removed"),
            "final_common_stocks": len(universe_tickers) if full_market_scan else universe_symbol_meta.get("final_common_stocks"),
            "symbol_removal_reasons": universe_symbol_meta.get("removal_reasons", {}),
            "duration_seconds": perf_counter() - start_time,
            "timings": {"universe_load_seconds": universe_seconds},
        }
        saved_meta = save_scanner_snapshot(universe_name, pd.DataFrame(), meta)
        return {"snapshot_df": pd.DataFrame(), "meta": saved_meta}

    download_start = perf_counter()
    stock_data_map = {}
    failure_reasons = {}
    download_chunk_size = 50 if full_market_scan else 75
    ticker_chunks = chunk_tickers(tuple(tickers), chunk_size=download_chunk_size)
    for chunk_index, ticker_chunk in enumerate(ticker_chunks, start=1):
        if progress_callback:
            completed_before_chunk = min((chunk_index - 1) * download_chunk_size, len(tickers))
            progress_callback(
                completed_before_chunk,
                len(tickers),
                f"Downloading Yahoo data batch {chunk_index}/{len(ticker_chunks)}...",
            )
        chunk_payload = get_batch_stock_data_chunk_detailed(ticker_chunk, period="2y", min_history_bars=200)
        stock_data_map.update(chunk_payload.get("data", {}))
        failure_reasons.update(chunk_payload.get("failures", {}))

    retry_start = perf_counter()
    retry_tickers = [
        ticker
        for ticker in tickers
        if ticker not in stock_data_map and failure_reasons.get(ticker) == "failed/no data"
    ]
    retry_recovered_count = 0
    if retry_tickers:
        retry_chunks = chunk_tickers(tuple(retry_tickers), chunk_size=10)
        for retry_index, retry_chunk in enumerate(retry_chunks, start=1):
            if progress_callback:
                progress_callback(
                    len(tickers) - len(retry_tickers) + min((retry_index - 1) * 10, len(retry_tickers)),
                    len(tickers),
                    f"Retrying missing Yahoo data {retry_index}/{len(retry_chunks)}...",
                )
            retry_payload = get_batch_stock_data_chunk_detailed(retry_chunk, period="2y", min_history_bars=200)
            retry_data = retry_payload.get("data", {})
            retry_recovered_count += len(retry_data)
            stock_data_map.update(retry_data)
            failure_reasons.update(retry_payload.get("failures", {}))
            for recovered_ticker in retry_data:
                failure_reasons.pop(recovered_ticker, None)
    retry_seconds = perf_counter() - retry_start
    download_seconds = perf_counter() - download_start
    failed_tickers = [ticker for ticker in tickers if ticker not in stock_data_map]
    valid_items = [(ticker, stock_data_map[ticker]) for ticker in tickers if ticker in stock_data_map]
    failed_no_data_count = sum(1 for ticker in failed_tickers if failure_reasons.get(ticker) == "failed/no data")
    invalid_ohlcv_count = sum(1 for ticker in failed_tickers if failure_reasons.get(ticker) == "invalid OHLCV")
    not_enough_history_count = sum(1 for ticker in failed_tickers if failure_reasons.get(ticker) == "not enough history")

    if progress_callback:
        progress_callback(0, len(valid_items), f"Calculating scanner scores 0/{len(valid_items)}...")
    benchmark_history = get_benchmark_history()
    market_regime = calculate_market_regime(get_market_regime_histories())
    prefilter_start = perf_counter()
    if full_market_scan:
        deep_score_limit = get_scan_mode_config(scan_mode).get("candidate_limit") or 1000
        deep_score_candidates, prefilter_meta = select_candidates_for_full_scoring(
            stock_data_map=stock_data_map,
            tickers=tickers,
            config=config,
            scan_mode=scan_mode or "Fast Scan",
            candidate_limit=max(int(deep_score_limit), 1000),
            bypass_prefilter=False,
        )
        deep_score_tickers = {ticker for ticker, _stock_data in deep_score_candidates}
    else:
        prefilter_meta = {
            "valid_data_count": len(stock_data_map),
            "prefilter_pass_count": len(stock_data_map),
            "prefilter_scored_count": len(stock_data_map),
            "prefilter_skip_reasons": {},
        }
        deep_score_tickers = set(stock_data_map)
    prefilter_seconds = perf_counter() - prefilter_start
    rows = []

    def score_snapshot_row(ticker: str, stock_data: dict) -> dict | None:
        if ticker in deep_score_tickers:
            score_data = calculate_signal_score(
                stock_data=stock_data,
                benchmark_history=benchmark_history,
                min_addv=config.get("min_addv"),
                calculate_historical_edge=calculate_historical_edge,
            )
            setup_quality = calculate_setup_quality(
                stock_data.get("history"),
                market_regime=market_regime,
                addv_20=stock_data.get("addv_20"),
                calculate_edge=calculate_historical_edge,
            )
        else:
            score_data = build_lightweight_score_data(stock_data)
            setup_quality = build_lightweight_setup_quality()
        row = build_snapshot_row(stock_data, score_data, setup_quality=setup_quality)
        row["deep_scored"] = ticker in deep_score_tickers
        metadata = symbol_metadata.get(ticker, {})
        if metadata:
            row["company"] = metadata.get("company")
            row["exchange"] = metadata.get("exchange")
        return row

    scoring_start = perf_counter()
    completed_count = 0
    snapshot_build_failed_count = 0
    if valid_items:
        max_workers = min(8, len(valid_items))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(score_snapshot_row, ticker, stock_data): ticker
                for ticker, stock_data in valid_items
            }
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    row = future.result()
                except Exception:
                    row = None
                if row:
                    rows.append(row)
                else:
                    failed_tickers.append(ticker)
                    failure_reasons[ticker] = "snapshot build failed"
                    snapshot_build_failed_count += 1
                completed_count += 1
                if progress_callback and (completed_count % 25 == 0 or completed_count == len(valid_items)):
                    progress_callback(completed_count, len(valid_items), f"Calculating scanner scores {completed_count}/{len(valid_items)}...")

    scoring_seconds = perf_counter() - scoring_start
    snapshot_df = pd.DataFrame(rows)

    if progress_callback:
        progress_callback(len(tickers), len(tickers), "Saving snapshot...")
    save_start = perf_counter()
    meta = {
        "universe_count": len(universe_tickers),
        "max_scan_cap": max_scan_cap or len(tickers),
        "requested_count": len(tickers),
        "tickers_attempted": len(tickers),
        "scanned_count": len(valid_items),
        "download_success_count": len(stock_data_map),
        "missing_data_count": failed_no_data_count + snapshot_build_failed_count,
        "invalid_ohlcv_count": invalid_ohlcv_count,
        "not_enough_history_count": not_enough_history_count,
        "snapshot_build_failed_count": snapshot_build_failed_count,
        "usable_snapshot_count": len(snapshot_df.index),
        "deep_scored_count": len(deep_score_tickers),
        "lightweight_snapshot_count": max(len(snapshot_df.index) - len(deep_score_tickers), 0),
        "retry_missing_data_count": len(retry_tickers),
        "retry_recovered_count": retry_recovered_count,
        "failed_tickers": failed_tickers,
        "failure_reasons": failure_reasons,
        "duration_seconds": perf_counter() - start_time,
        "result_count": len(snapshot_df.index),
        "scan_mode": scan_mode or "Fast Scan",
        "valid_data_count": prefilter_meta.get("valid_data_count", len(stock_data_map)),
        "prefilter_pass_count": prefilter_meta.get("prefilter_pass_count", len(stock_data_map)),
        "prefilter_scored_count": prefilter_meta.get("prefilter_scored_count", len(deep_score_tickers)),
        "prefilter_skip_reasons": prefilter_meta.get("prefilter_skip_reasons", {}),
        "raw_symbols_loaded": universe_symbol_meta.get("raw_symbols_loaded"),
        "non_common_removed": universe_symbol_meta.get("non_common_removed"),
        "final_common_stocks": len(universe_tickers) if full_market_scan else universe_symbol_meta.get("final_common_stocks"),
        "symbol_removal_reasons": universe_symbol_meta.get("removal_reasons", {}),
        "market_regime": market_regime,
        "calculate_historical_edge": bool(calculate_historical_edge),
        "timings": {
            "universe_load_seconds": universe_seconds,
            "data_fetch_seconds": download_seconds,
            "data_retry_seconds": retry_seconds,
            "prefilter_seconds": prefilter_seconds,
            "scoring_seconds": scoring_seconds,
        },
    }
    saved_meta = save_scanner_snapshot(universe_name, snapshot_df, meta)
    save_seconds = perf_counter() - save_start
    saved_meta["timings"]["save_seconds"] = save_seconds
    saved_meta["duration_seconds"] = perf_counter() - start_time
    save_scanner_snapshot(universe_name, snapshot_df, saved_meta)

    if progress_callback:
        progress_callback(len(tickers), len(tickers), "Done.")
    return {"snapshot_df": snapshot_df, "meta": saved_meta}


def scan_universe(
    universe_name: str,
    max_tickers: int | None = None,
    scan_mode: str | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    min_avg_volume: float | None = None,
    min_dollar_volume: float | None = None,
    progress_callback=None,
) -> dict:
    start_time = perf_counter()
    if progress_callback:
        progress_callback(0, 1, "Preparing universe...")

    config = get_universe_config(universe_name)
    full_market_scan = is_full_market_universe(universe_name)
    mode_config = get_scan_mode_config(scan_mode)
    default_candidate_limit = mode_config.get("candidate_limit")
    candidate_limit = None if full_market_scan else (max_tickers if max_tickers is not None else default_candidate_limit)
    max_scan_cap = int(candidate_limit) if candidate_limit is not None else 0
    universe_start = perf_counter()
    universe_tickers = get_universe(universe_name)
    tickers = universe_tickers
    universe_seconds = perf_counter() - universe_start
    benchmark_history = get_benchmark_history()
    scanned_rows: list[dict] = []
    missing_data_count = 0

    if not tickers:
        duration_seconds = perf_counter() - start_time
        return {
            "scored_rows": [],
            "meta": build_scan_meta(
                universe_count=len(universe_tickers),
                max_scan_cap=max_scan_cap,
                requested_count=0,
                scanned_count=0,
                missing_data_count=0,
                filter_skip_count=0,
                duration_seconds=duration_seconds,
                timings={"universe_load_seconds": universe_seconds},
            ),
        }

    if progress_callback:
        progress_callback(0, len(tickers), "Downloading Yahoo batch data...")
    data_start = perf_counter()
    stock_data_map = get_batch_stock_data(tuple(tickers))
    data_seconds = perf_counter() - data_start

    if progress_callback:
        progress_callback(0, len(tickers), "Pre-filtering liquid breakout candidates...")
    prefilter_start = perf_counter()
    valid_stock_items, prefilter_meta = select_candidates_for_full_scoring(
        stock_data_map=stock_data_map,
        tickers=tickers,
        config=config,
        scan_mode=scan_mode or "Fast Scan",
        candidate_limit=candidate_limit,
        min_price=min_price,
        max_price=max_price,
        min_avg_volume=min_avg_volume,
        min_dollar_volume=min_dollar_volume,
        bypass_prefilter=full_market_scan,
    )
    prefilter_seconds = perf_counter() - prefilter_start

    if progress_callback:
        progress_callback(0, len(valid_stock_items), f"Scoring tickers 0/{len(valid_stock_items)}...")
    def score_stock(ticker: str, stock_data: dict) -> dict | None:
        score_data = calculate_signal_score(
            stock_data=stock_data,
            benchmark_history=benchmark_history,
            min_addv=config.get("min_addv"),
        )
        return build_scanned_row(stock_data, score_data)

    scoring_start = perf_counter()
    missing_data_count = len(tickers) - len(stock_data_map)

    completed_count = 0
    if progress_callback and not valid_stock_items:
        progress_callback(0, 1, "No candidates survived the pre-filter.")

    if valid_stock_items:
        max_workers = min(8, len(valid_stock_items))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(score_stock, ticker, stock_data)
                for ticker, stock_data in valid_stock_items
            ]
            for future in as_completed(futures):
                try:
                    row = future.result()
                except Exception:
                    row = None
                    missing_data_count += 1
                if row:
                    scanned_rows.append(row)
                completed_count += 1
                if progress_callback and (completed_count % 25 == 0 or completed_count == len(valid_stock_items)):
                    progress_callback(completed_count, len(valid_stock_items), f"Scoring tickers {completed_count}/{len(valid_stock_items)}...")

    scoring_seconds = perf_counter() - scoring_start
    if progress_callback:
        progress_callback(len(tickers), len(tickers), "Done.")

    meta = build_scan_meta(
        universe_count=len(universe_tickers),
        max_scan_cap=max_scan_cap or len(valid_stock_items),
        requested_count=len(tickers),
        scanned_count=len(valid_stock_items),
        missing_data_count=missing_data_count,
        filter_skip_count=0,
        duration_seconds=perf_counter() - start_time,
        result_count=len(scanned_rows),
        timings={
            "universe_load_seconds": universe_seconds,
            "data_fetch_seconds": data_seconds,
            "prefilter_seconds": prefilter_seconds,
            "scoring_seconds": scoring_seconds,
        },
        warning="Full Scan scores every valid ticker and can take longer on large universes." if scan_mode == "Full Scan" else "",
    )
    meta.update(
        {
            "scan_mode": scan_mode or "Fast Scan",
            "valid_data_count": prefilter_meta.get("valid_data_count", len(stock_data_map)),
            "prefilter_pass_count": prefilter_meta.get("prefilter_pass_count", 0),
            "prefilter_scored_count": prefilter_meta.get("prefilter_scored_count", 0),
            "prefilter_skip_reasons": prefilter_meta.get("prefilter_skip_reasons", {}),
        }
    )
    return {
        "scored_rows": scanned_rows,
        "meta": meta,
    }


def run_scanner(
    universe_name: str,
    min_score: int,
    max_results: int,
    max_tickers: int | None = None,
    scan_mode: str | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    min_avg_volume: float | None = None,
    min_dollar_volume: float | None = None,
    progress_callback=None,
) -> dict:
    scan_payload = scan_universe(
        universe_name=universe_name,
        max_tickers=max_tickers,
        scan_mode=scan_mode,
        min_price=min_price,
        max_price=max_price,
        min_avg_volume=min_avg_volume,
        min_dollar_volume=min_dollar_volume,
        progress_callback=progress_callback,
    )
    return build_display_results(
        scan_payload=scan_payload,
        universe_name=universe_name,
        min_score=min_score,
        max_results=max_results,
        min_price=min_price,
        max_price=max_price,
        min_avg_volume=min_avg_volume,
        min_dollar_volume=min_dollar_volume,
    )


def build_ma_frame(history: pd.DataFrame, selected_mas: list[str]) -> pd.DataFrame:
    if history.empty or "Close" not in history.columns:
        return pd.DataFrame()

    frame = history.copy()
    for ma_name in selected_mas:
        config = MA_COLUMNS.get(ma_name)
        if not config:
            continue
        if config["type"] == "ema":
            frame[config["column"]] = frame["Close"].ewm(
                span=config["period"],
                adjust=False,
                min_periods=config["period"],
            ).mean()
        else:
            frame[config["column"]] = frame["Close"].rolling(config["period"]).mean()
    return frame.dropna(subset=["Close"])


def detect_ma_events(
    ticker: str,
    history: pd.DataFrame,
    selected_mas: list[str],
    lookback_days: int,
    event_type: str,
    score_data: dict,
    stock_data: dict,
) -> list[dict]:
    frame = build_ma_frame(history, selected_mas)
    if frame.empty or len(frame.index) < 2:
        return []

    rows = []
    event_frame = frame.tail(lookback_days + 1)
    if len(event_frame.index) < 2:
        return []

    for ma_name in selected_mas:
        ma_column = MA_COLUMNS.get(ma_name, {}).get("column")
        if not ma_column or ma_column not in event_frame.columns:
            continue

        for index in range(1, len(event_frame.index)):
            previous_row = event_frame.iloc[index - 1]
            current_row = event_frame.iloc[index]
            previous_close = previous_row.get("Close")
            previous_ma = previous_row.get(ma_column)
            current_close = current_row.get("Close")
            current_ma = current_row.get(ma_column)
            if pd.isna(previous_close) or pd.isna(previous_ma) or pd.isna(current_close) or pd.isna(current_ma):
                continue

            reclaim = previous_close < previous_ma and current_close > current_ma
            ma_break = previous_close > previous_ma and current_close < current_ma
            if event_type == "Reclaims only" and not reclaim:
                continue
            if event_type == "Breaks only" and not ma_break:
                continue
            if event_type == "Both" and not (reclaim or ma_break):
                continue

            event_direction = "Reclaim" if reclaim else "Break"
            event_date = event_frame.index[index]
            days_since_event = len(frame.loc[event_date:].index) - 1
            pct_from_ma = float(((current_close - current_ma) / current_ma) * 100) if current_ma else None
            rows.append(
                {
                    "ticker": ticker,
                    "event": f"{ma_name} {event_direction}",
                    "event_date": event_date.date().isoformat() if hasattr(event_date, "date") else str(event_date),
                    "days_since_event": days_since_event,
                    "close": float(current_close),
                    "ma_value": float(current_ma),
                    "pct_from_ma": pct_from_ma,
                    "volume_ratio": stock_data.get("volume_ratio"),
                    "signal_score": score_data.get("signal_score", 0.0),
                    "label": score_data.get("label", "Pass"),
                    "tags": score_data.get("tags", []),
                }
            )

    return rows


def scan_ma_events(
    universe_name: str,
    lookback_days: int,
    selected_mas: list[str],
    event_type: str,
    min_score: int,
    max_results: int,
    max_tickers: int | None = None,
    min_price: float | None = None,
    min_avg_volume: float | None = None,
    min_dollar_volume: float | None = None,
) -> list[dict]:
    if not selected_mas:
        return []

    config = get_universe_config(universe_name)
    if min_price is not None:
        config["min_price"] = min_price
    if min_dollar_volume is not None:
        config["min_addv"] = min_dollar_volume
    configured_limit = int(config.get("ticker_limit", 75))
    ticker_limit = max_tickers if max_tickers is not None else configured_limit
    tickers = get_universe(universe_name)[: max(int(ticker_limit), 0)]
    benchmark_history = get_benchmark_history()
    stock_data_map = get_batch_stock_data(tuple(tickers))
    rows: list[dict] = []

    for ticker in tickers:
        stock_data = stock_data_map.get(ticker)
        if not stock_data or not passes_filters(stock_data, config):
            continue
        if min_avg_volume is not None and stock_data.get("avg_volume_20", 0) < min_avg_volume:
            continue

        score_data = calculate_signal_score(
            stock_data=stock_data,
            benchmark_history=benchmark_history,
            min_addv=config.get("min_addv"),
        )
        if score_data.get("signal_score", 0.0) < min_score:
            continue

        rows.extend(
            detect_ma_events(
                ticker=ticker,
                history=stock_data.get("history", pd.DataFrame()),
                selected_mas=selected_mas,
                lookback_days=lookback_days,
                event_type=event_type,
                score_data=score_data,
                stock_data=stock_data,
            )
        )

    return sorted(
        rows,
        key=lambda row: (
            row.get("days_since_event", 10_000),
            -row.get("signal_score", 0.0),
            -(row.get("volume_ratio") or 0.0),
        ),
    )[:max_results]


def _trigger_required_mas(event_types: list[str]) -> list[str]:
    required = {"EMA8", "EMA21", "SMA50", "SMA100", "SMA200"}
    for event_type in event_types:
        ma_name = TRIGGER_EVENT_MA_MAP.get(event_type)
        if ma_name:
            required.add(ma_name)
        if event_type in {"Golden Cross: SMA50 crossed above SMA200", "Death Cross: SMA50 crossed below SMA200"}:
            required.update({"SMA50", "SMA200"})
    return sorted(required, key=lambda name: MA_COLUMNS[name]["period"])


def _format_trigger_date(event_date: object) -> str:
    if hasattr(event_date, "date"):
        return event_date.date().isoformat()
    return str(event_date)


def _current_distance(frame: pd.DataFrame, ma_name: str) -> float | None:
    config = MA_COLUMNS.get(ma_name, {})
    ma_column = config.get("column")
    if not ma_column or ma_column not in frame.columns or frame.empty:
        return None
    latest = frame.iloc[-1]
    return safe_pct_distance(latest.get("Close"), latest.get(ma_column))


def detect_trigger_events(
    ticker: str,
    history: pd.DataFrame,
    event_types: list[str],
    lookback_days: int,
    only_currently_above_ema21_sma50: bool,
    stock_data: dict,
) -> list[dict]:
    if not event_types:
        return []

    frame = build_ma_frame(history, _trigger_required_mas(event_types))
    if frame.empty or len(frame.index) < 2:
        return []

    latest = frame.iloc[-1]
    latest_close = latest.get("Close")
    if only_currently_above_ema21_sma50:
        latest_ema21 = latest.get("EMA21")
        latest_sma50 = latest.get("SMA50")
        if (
            pd.isna(latest_close)
            or pd.isna(latest_ema21)
            or pd.isna(latest_sma50)
            or latest_close <= latest_ema21
            or latest_close <= latest_sma50
        ):
            return []

    event_frame = frame.tail(lookback_days + 1)
    if len(event_frame.index) < 2:
        return []

    trend_snapshot = build_long_trend_snapshot(history, stock_data.get("price"), stock_data.get("volume_ratio"))
    base_row = {
        "ticker": ticker,
        "price": stock_data.get("price"),
        "volume_ratio": stock_data.get("volume_ratio"),
        "avg_volume": stock_data.get("avg_volume_20"),
        "distance_from_ema8": _current_distance(frame, "EMA8"),
        "distance_from_ema21": _current_distance(frame, "EMA21"),
        "distance_from_sma50": _current_distance(frame, "SMA50"),
        "distance_from_sma100": _current_distance(frame, "SMA100"),
        "distance_from_sma200": _current_distance(frame, "SMA200"),
        "trend_status": trend_snapshot.get("long_status", "N/A"),
        "long_status": trend_snapshot.get("long_status", "N/A"),
        "ema_stack_status": trend_snapshot.get("ema_stack_status", "N/A"),
    }
    rows: list[dict] = []

    for index in range(1, len(event_frame.index)):
        previous_row = event_frame.iloc[index - 1]
        current_row = event_frame.iloc[index]
        current_close = current_row.get("Close")
        previous_close = previous_row.get("Close")
        if pd.isna(previous_close) or pd.isna(current_close):
            continue

        event_date = event_frame.index[index]
        days_since_event = len(frame.loc[event_date:].index) - 1

        for event_type in event_types:
            ma_name = TRIGGER_EVENT_MA_MAP.get(event_type)
            if ma_name:
                ma_column = MA_COLUMNS[ma_name]["column"]
                previous_ma = previous_row.get(ma_column)
                current_ma = current_row.get(ma_column)
                if pd.isna(previous_ma) or pd.isna(current_ma):
                    continue
                if previous_close <= previous_ma and current_close > current_ma:
                    rows.append(
                        {
                            **base_row,
                            "trigger": event_type,
                            "trigger_date": _format_trigger_date(event_date),
                            "days_ago": days_since_event,
                        }
                    )
                continue

            previous_sma50 = previous_row.get("SMA50")
            current_sma50 = current_row.get("SMA50")
            previous_sma200 = previous_row.get("SMA200")
            current_sma200 = current_row.get("SMA200")
            if pd.isna(previous_sma50) or pd.isna(current_sma50) or pd.isna(previous_sma200) or pd.isna(current_sma200):
                continue
            if event_type == "Golden Cross: SMA50 crossed above SMA200" and previous_sma50 <= previous_sma200 and current_sma50 > current_sma200:
                rows.append(
                    {
                        **base_row,
                        "trigger": event_type,
                        "trigger_date": _format_trigger_date(event_date),
                        "days_ago": days_since_event,
                    }
                )
            elif event_type == "Death Cross: SMA50 crossed below SMA200" and previous_sma50 >= previous_sma200 and current_sma50 < current_sma200:
                rows.append(
                    {
                        **base_row,
                        "trigger": event_type,
                        "trigger_date": _format_trigger_date(event_date),
                        "days_ago": days_since_event,
                    }
                )

    return rows


def get_trigger_sort_key(row: dict) -> tuple:
    return (
        safe_number(row.get("days_ago")) if safe_number(row.get("days_ago")) is not None else 10_000,
        -(safe_number(row.get("volume_ratio")) or 0.0),
        -TRIGGER_TREND_RANK.get(row.get("trend_status"), 0),
        TRIGGER_PRIORITY_RANK.get(row.get("trigger"), 99),
    )


def get_trigger_dedupe_sort_key(row: dict) -> tuple:
    return (
        safe_number(row.get("days_ago")) if safe_number(row.get("days_ago")) is not None else 10_000,
        TRIGGER_PRIORITY_RANK.get(row.get("trigger"), 99),
        -(safe_number(row.get("volume_ratio")) or 0.0),
        -TRIGGER_TREND_RANK.get(row.get("trend_status"), 0),
    )


def sort_trigger_rows(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=get_trigger_sort_key)


def dedupe_trigger_rows_by_ticker(rows: list[dict]) -> list[dict]:
    deduped_rows: list[dict] = []
    seen_tickers: set[str] = set()
    for row in sorted(rows, key=get_trigger_dedupe_sort_key):
        ticker = str(row.get("ticker", "")).strip().upper()
        if not ticker or ticker in seen_tickers:
            continue
        seen_tickers.add(ticker)
        deduped_rows.append(row)
    return deduped_rows


def postprocess_trigger_rows(
    rows: list[dict],
    optional_filters: dict | None = None,
    only_currently_above_ema21_sma50: bool = False,
    one_row_per_ticker: bool = True,
    show_all_trigger_events: bool = False,
    min_days_ago: int | None = None,
    max_days_ago: int | None = None,
    max_results: int = 25,
) -> list[dict]:
    filtered_rows = rows
    if only_currently_above_ema21_sma50:
        filtered_rows, _skipped = apply_optional_filters(
            filtered_rows,
            {"above_ema21": True, "above_sma50": True},
        )

    filtered_rows, _skipped = apply_optional_filters(filtered_rows, optional_filters)

    if min_days_ago is not None or max_days_ago is not None:
        age_band_rows = []
        for row in filtered_rows:
            days_ago = safe_number(row.get("days_ago"))
            if days_ago is None:
                continue
            if min_days_ago is not None and days_ago < min_days_ago:
                continue
            if max_days_ago is not None and days_ago > max_days_ago:
                continue
            age_band_rows.append(row)
        filtered_rows = age_band_rows

    sorted_rows = sort_trigger_rows(filtered_rows)
    if one_row_per_ticker and not show_all_trigger_events:
        sorted_rows = dedupe_trigger_rows_by_ticker(sorted_rows)

    return sorted_rows[:max(int(max_results), 0)]


def scan_trigger_events(
    universe_name: str,
    event_types: list[str],
    lookback_days: int,
    only_currently_above_ema21_sma50: bool,
    max_results: int,
    max_tickers: int | None = None,
    tickers: tuple[str, ...] | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    min_avg_volume: float | None = None,
    min_dollar_volume: float | None = None,
    optional_filters: dict | None = None,
) -> list[dict]:
    if not event_types:
        return []

    config = get_universe_config(universe_name)
    if min_price is not None:
        config["min_price"] = min_price
    if min_dollar_volume is not None:
        config["min_addv"] = min_dollar_volume

    configured_limit = int(config.get("ticker_limit", 75))
    ticker_limit = max_tickers if max_tickers is not None else configured_limit
    source_tickers = list(tickers) if tickers is not None else get_universe(universe_name)
    scan_tickers = source_tickers[: max(int(ticker_limit), 0)]
    stock_data_map = get_batch_stock_data(tuple(scan_tickers))
    rows: list[dict] = []

    for ticker in scan_tickers:
        stock_data = stock_data_map.get(ticker)
        if not stock_data:
            continue

        rows.extend(
            detect_trigger_events(
                ticker=ticker,
                history=stock_data.get("history", pd.DataFrame()),
                event_types=event_types,
                lookback_days=lookback_days,
                only_currently_above_ema21_sma50=only_currently_above_ema21_sma50,
                stock_data=stock_data,
            )
        )

    if optional_filters is None:
        optional_filters = {
            "use_price_filter": min_price is not None or max_price is not None,
            "min_price": min_price,
            "max_price": max_price,
            "use_volume_filter": min_avg_volume is not None,
            "min_avg_volume": min_avg_volume,
            "use_relative_volume_filter": False,
            "min_relative_volume": None,
            "above_ema21": False,
            "above_sma50": False,
            "above_sma200": False,
            "use_trend_status_filter": False,
            "trend_status_filter_values": [],
        }
    return postprocess_trigger_rows(
        rows=rows,
        optional_filters=optional_filters,
        only_currently_above_ema21_sma50=only_currently_above_ema21_sma50,
        one_row_per_ticker=False,
        show_all_trigger_events=True,
        min_days_ago=None,
        max_days_ago=None,
        max_results=max_results,
    )
