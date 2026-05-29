from __future__ import annotations

import pandas as pd


def _safe_latest(series: pd.Series) -> float | None:
    values = series.dropna()
    if values.empty:
        return None
    try:
        return float(values.iloc[-1])
    except (TypeError, ValueError):
        return None


def _prepare_frame(history: pd.DataFrame | None) -> pd.DataFrame:
    if history is None or history.empty or "Close" not in history.columns:
        return pd.DataFrame()
    frame = history.copy()
    frame["Close"] = pd.to_numeric(frame["Close"], errors="coerce")
    return frame.dropna(subset=["Close"])


def _ticker_regime_flags(history: pd.DataFrame | None) -> dict:
    frame = _prepare_frame(history)
    result = {
        "close": None,
        "ema21": None,
        "sma50": None,
        "sma200": None,
        "above_ema21": False,
        "above_sma50": False,
        "above_sma200": False,
    }
    if len(frame.index) < 50:
        return result

    close = _safe_latest(frame["Close"])
    ema21 = _safe_latest(frame["Close"].ewm(span=21, adjust=False, min_periods=21).mean())
    sma50 = _safe_latest(frame["Close"].rolling(50).mean())
    sma200 = _safe_latest(frame["Close"].rolling(200).mean()) if len(frame.index) >= 200 else None
    result.update(
        {
            "close": close,
            "ema21": ema21,
            "sma50": sma50,
            "sma200": sma200,
            "above_ema21": close is not None and ema21 is not None and close > ema21,
            "above_sma50": close is not None and sma50 is not None and close > sma50,
            "above_sma200": close is not None and sma200 is not None and close > sma200,
        }
    )
    return result


def calculate_market_regime(market_data: dict[str, pd.DataFrame] | None) -> dict:
    """Classify broad market support for long swing setups using SPY, QQQ, and IWM."""
    market_data = market_data or {}
    spy = _ticker_regime_flags(market_data.get("SPY"))
    qqq = _ticker_regime_flags(market_data.get("QQQ"))
    iwm = _ticker_regime_flags(market_data.get("IWM"))
    flags = {"SPY": spy, "QQQ": qqq, "IWM": iwm}

    above_sma50_count = sum(1 for values in flags.values() if values.get("above_sma50"))
    spy_supportive = spy.get("above_ema21") and spy.get("above_sma50")
    qqq_supportive = qqq.get("above_ema21") and qqq.get("above_sma50")

    if spy_supportive and qqq_supportive and above_sma50_count >= 2:
        regime = "Risk-On"
        score = 15.0
        penalty = 0.0
        summary = "SPY and QQQ are above EMA21/SMA50 and broad participation is supportive."
    elif (not spy.get("above_sma50") and not qqq.get("above_sma50")) or not spy.get("above_sma200"):
        regime = "Risk-Off"
        score = 0.0
        penalty = 30.0
        summary = "SPY/QQQ trend is weak or SPY is below SMA200, which penalizes long breakout setups."
    else:
        regime = "Neutral"
        score = 8.0
        penalty = 10.0
        summary = "Broad market signals are mixed, so long setups receive a smaller score."

    return {
        "market_regime": regime,
        "market_regime_score": score,
        "market_regime_penalty": penalty,
        "above_sma50_count": above_sma50_count,
        "flags": flags,
        "summary": summary,
    }
