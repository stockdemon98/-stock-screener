from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


CHART_BG = "#ffffff"
GRID_COLOR = "#e5e7eb"
TEXT_COLOR = "#111827"
CLOSE_COLOR = "#111827"
EMA8_COLOR = "#fb7185"
EMA21_COLOR = "#f59e0b"
SMA50_COLOR = "#0284c7"
SMA100_COLOR = "#64748b"
SMA200_COLOR = "#7c3aed"
RSI_COLOR = "#0f766e"
UP_COLOR = "#16a34a"
DOWN_COLOR = "#ef4444"
MARKER_BG = "#ffffff"
MARKER_OUTLINE = "#111827"
GC_COLOR = "#ca8a04"
DC_COLOR = "#f43f5e"
TIMEFRAME_TO_DAYS = {
    "6M": 183,
    "1Y": 366,
    "2Y": 731,
    "5Y": 1827,
}


def build_chart_frame(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame()

    chart_data = history.copy().reset_index().rename(columns={"index": "Date"})
    chart_data["EMA8"] = chart_data["Close"].ewm(span=8, adjust=False, min_periods=8).mean()
    chart_data["EMA21"] = chart_data["Close"].ewm(span=21, adjust=False, min_periods=21).mean()
    chart_data["SMA50"] = chart_data["Close"].rolling(50).mean()
    chart_data["SMA100"] = chart_data["Close"].rolling(100).mean()
    chart_data["SMA200"] = chart_data["Close"].rolling(200).mean()
    chart_data["VolumeColor"] = chart_data.apply(
        lambda row: UP_COLOR if row["Close"] >= row["Open"] else DOWN_COLOR,
        axis=1,
    )
    chart_data["ATR14"] = build_atr(chart_data, length=14)
    chart_data["RSI14"] = build_rsi(chart_data["Close"], length=14)
    return chart_data


def build_atr(chart_data: pd.DataFrame, length: int) -> pd.Series:
    previous_close = chart_data["Close"].shift(1)
    true_range = pd.concat(
        [
            chart_data["High"] - chart_data["Low"],
            (chart_data["High"] - previous_close).abs(),
            (chart_data["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(length).mean()


def build_rsi(closes: pd.Series, length: int = 14) -> pd.Series:
    if closes.empty:
        return pd.Series(dtype="float64")

    changes = closes.diff()
    gains = changes.clip(lower=0)
    losses = changes.clip(upper=0).abs()
    average_gain = gains.rolling(length, min_periods=length).mean()
    average_loss = losses.rolling(length, min_periods=length).mean()
    rs = average_gain / average_loss.mask(average_loss == 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi.where(average_loss != 0, 100)


def get_indicator_value(chart_data: pd.DataFrame, column: str) -> float | None:
    if chart_data.empty or column not in chart_data.columns:
        return None

    values = chart_data[column].dropna()
    if values.empty:
        return None
    return float(values.iloc[-1])


def format_legend_value(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2f}"


def format_readout_price(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"${value:.2f}"


def build_technical_readout(df: pd.DataFrame, ticker: str | None = None) -> dict:
    if df is None or df.empty or len(df.index) < 25:
        return {
            "status": "Not enough data",
            "trend_summary": "Not enough recent data for a reliable 1-month technical readout.",
            "ma_summary": "",
            "momentum_summary": "",
            "volume_summary": "",
            "support_resistance_summary": "",
            "swing_read": "Swing read: needs more price history before making a useful read.",
        }

    if not {"Close", "High", "Low"}.issubset(set(df.columns)):
        return {
            "status": "Not enough data",
            "trend_summary": "Not enough recent price data for a reliable 1-month technical readout.",
            "ma_summary": "",
            "momentum_summary": "",
            "volume_summary": "",
            "support_resistance_summary": "",
            "swing_read": "Swing read: needs complete OHLC history before making a useful read.",
        }

    if "Volume" not in df.columns:
        df = df.copy()
        df["Volume"] = 0

    chart_data = build_chart_frame(df) if "Date" not in df.columns else df.copy()
    if chart_data.empty or len(chart_data.index) < 25:
        return {
            "status": "Not enough data",
            "trend_summary": "Not enough recent data for a reliable 1-month technical readout.",
            "ma_summary": "",
            "momentum_summary": "",
            "volume_summary": "",
            "support_resistance_summary": "",
            "swing_read": "Swing read: needs more price history before making a useful read.",
        }

    for column, span in (("EMA8", 8), ("EMA21", 21)):
        if column not in chart_data.columns:
            chart_data[column] = chart_data["Close"].ewm(span=span, adjust=False, min_periods=span).mean()
    for column, window in (("SMA50", 50), ("SMA200", 200)):
        if column not in chart_data.columns:
            chart_data[column] = chart_data["Close"].rolling(window).mean()

    recent = chart_data.tail(20).copy()
    first_half = recent.head(10)
    second_half = recent.tail(10)
    latest = chart_data.iloc[-1]
    current_close = float(latest["Close"])
    start_close = float(recent["Close"].iloc[0])
    month_change_pct = ((current_close / start_close) - 1) * 100 if start_close else 0.0

    higher_highs = second_half["High"].max() > first_half["High"].max()
    higher_lows = second_half["Low"].min() > first_half["Low"].min()
    lower_highs = second_half["High"].max() < first_half["High"].max()
    lower_lows = second_half["Low"].min() < first_half["Low"].min()
    if higher_highs and higher_lows:
        trend_label = "higher highs and higher lows"
    elif lower_highs and lower_lows:
        trend_label = "lower highs and lower lows"
    else:
        trend_label = "sideways or mixed structure"
    trend_summary = f"Short-term trend: price is showing {trend_label} over the last 20 trading days."

    ma_values = {
        "EMA8": latest.get("EMA8"),
        "EMA21": latest.get("EMA21"),
        "SMA50": latest.get("SMA50"),
        "SMA200": latest.get("SMA200"),
    }
    above_mas = [name for name, value in ma_values.items() if pd.notna(value) and current_close >= float(value)]
    below_mas = [name for name, value in ma_values.items() if pd.notna(value) and current_close < float(value)]
    ma_parts = []
    if above_mas:
        ma_parts.append(f"above {', '.join(above_mas)}")
    if below_mas:
        ma_parts.append(f"below {', '.join(below_mas)}")
    ema8 = ma_values.get("EMA8")
    ema21 = ma_values.get("EMA21")
    ema_relationship = ""
    if pd.notna(ema8) and pd.notna(ema21):
        ema_relationship = " EMA8 is above EMA21." if float(ema8) >= float(ema21) else " EMA8 is still below EMA21."
    ma_summary = f"Moving averages: price is {' and '.join(ma_parts) if ma_parts else 'without enough MA data'}." + ema_relationship

    recent_high = float(recent["High"].max())
    recent_low = float(recent["Low"].min())
    near_resistance = current_close >= recent_high * 0.97
    recent_reclaim = False
    reclaim_volume = None
    for ma_name in ("EMA8", "EMA21", "SMA50", "SMA200"):
        if ma_name not in chart_data.columns:
            continue
        frame = chart_data[["Close", "Volume", ma_name]].dropna().tail(6)
        if len(frame.index) < 2:
            continue
        crossed = (frame["Close"].shift(1) <= frame[ma_name].shift(1)) & (frame["Close"] > frame[ma_name])
        if crossed.any():
            recent_reclaim = True
            cross_index = crossed[crossed].index[-1]
            avg_volume_at_cross = chart_data.loc[:cross_index, "Volume"].tail(20).mean()
            reclaim_volume = float(frame.loc[cross_index, "Volume"]) / avg_volume_at_cross if avg_volume_at_cross else None
            break
    if month_change_pct > 3:
        momentum_action = "has improved"
    elif month_change_pct < -3:
        momentum_action = "has weakened"
    else:
        momentum_action = "is little changed"
    if recent_reclaim:
        momentum_context = "and has recently reclaimed a moving average"
    elif near_resistance:
        momentum_context = "and is pressing into the recent 1-month high"
    elif current_close < recent_high * 0.94:
        momentum_context = "but remains below recent resistance"
    else:
        momentum_context = "with no clean resistance break yet"
    momentum_summary = f"Momentum: price {momentum_action} {month_change_pct:+.1f}% over the past month {momentum_context}."

    recent_volume = float(chart_data["Volume"].tail(5).mean())
    normal_volume = float(chart_data["Volume"].tail(20).mean())
    volume_ratio = recent_volume / normal_volume if normal_volume else None
    if volume_ratio is None:
        volume_summary = "Volume: recent volume is unavailable."
    elif volume_ratio >= 1.25:
        volume_summary = "Volume: recent volume is above normal."
    elif volume_ratio <= 0.80:
        volume_summary = "Volume: recent volume is below normal."
    else:
        volume_summary = "Volume: recent volume is near normal."
    if reclaim_volume is not None:
        volume_summary += " The latest reclaim came on stronger volume." if reclaim_volume >= 1.10 else " The latest reclaim came on lighter volume."

    ma_below = [float(value) for value in ma_values.values() if pd.notna(value) and float(value) < current_close]
    ma_above = [float(value) for value in ma_values.values() if pd.notna(value) and float(value) > current_close]
    support_candidates = [recent_low] + ma_below
    resistance_candidates = [recent_high] + ma_above
    support = max(value for value in support_candidates if value < current_close) if any(value < current_close for value in support_candidates) else recent_low
    resistance = min(value for value in resistance_candidates if value > current_close) if any(value > current_close for value in resistance_candidates) else recent_high
    support_resistance_summary = (
        f"Support/resistance: nearby support appears around {format_readout_price(support)}, "
        f"with resistance near {format_readout_price(resistance)}."
    )

    above_core = {"EMA8", "EMA21"}.issubset(set(above_mas))
    above_intermediate = "SMA50" in above_mas
    below_majority = len(below_mas) >= 3
    if month_change_pct > 0 and above_core and float(ema8) >= float(ema21) and (higher_lows or above_intermediate):
        status = "Constructive"
        swing_read = "Swing read: constructive, with confirmation improving if price clears resistance on stronger volume."
    elif month_change_pct < 0 and below_majority and lower_lows:
        status = "Weak / avoid"
        swing_read = "Swing read: weak / avoid until price reclaims short-term moving averages and stabilizes."
    elif recent_reclaim or near_resistance:
        status = "Needs confirmation"
        swing_read = "Swing read: needs confirmation; a clean close above resistance with volume would improve the setup."
    else:
        status = "Watchlist"
        swing_read = "Swing read: watchlist; structure is mixed and needs clearer follow-through."

    return {
        "status": status,
        "trend_summary": trend_summary,
        "ma_summary": ma_summary,
        "momentum_summary": momentum_summary,
        "volume_summary": volume_summary,
        "support_resistance_summary": support_resistance_summary,
        "swing_read": swing_read,
    }


def render_chart_legend(chart_data: pd.DataFrame) -> None:
    close_value = get_indicator_value(chart_data, "Close")
    ema8_value = get_indicator_value(chart_data, "EMA8")
    ema21_value = get_indicator_value(chart_data, "EMA21")
    sma50_value = get_indicator_value(chart_data, "SMA50")
    sma100_value = get_indicator_value(chart_data, "SMA100")
    sma200_value = get_indicator_value(chart_data, "SMA200")

    st.markdown(
        (
            f"<div style='padding:6px 0 12px 0; font-size:0.95rem;'>"
            f"<span style='color:{CLOSE_COLOR}; margin-right:16px;'>Close: {format_legend_value(close_value)}</span>"
            f"<span style='color:{EMA8_COLOR}; margin-right:16px;'>EMA8: {format_legend_value(ema8_value)}</span>"
            f"<span style='color:{EMA21_COLOR}; margin-right:16px;'>EMA21: {format_legend_value(ema21_value)}</span>"
            f"<span style='color:{SMA50_COLOR}; margin-right:16px;'>SMA50: {format_legend_value(sma50_value)}</span>"
            f"<span style='color:{SMA100_COLOR}; margin-right:16px;'>SMA100: {format_legend_value(sma100_value)}</span>"
            f"<span style='color:{SMA200_COLOR};'>SMA200: {format_legend_value(sma200_value)}</span>"
            f"</div>"
        ),
        unsafe_allow_html=True,
    )


def render_marker_legend() -> None:
    st.caption("Markers: + = bullish reclaim above the moving average, - = bearish break below it. Golden Cross = SMA50 above SMA200. Death Cross = SMA50 below SMA200.")


def get_event_description(indicator_column: str, is_reclaim: bool) -> str:
    direction = "Reclaim" if is_reclaim else "Break"
    return f"{indicator_column} {direction}"


def render_chart_controls(key_prefix: str) -> dict:
    st.markdown("#### Chart Controls")

    control_col1, control_col2, control_col3, control_col4 = st.columns([1.0, 1.15, 1.2, 1.45])
    with control_col1:
        timeframe = st.selectbox(
            "Timeframe",
            options=["6M", "1Y", "2Y", "5Y"],
            index=1,
            key=f"{key_prefix}-timeframe",
        )
        chart_type = st.radio(
            "Chart Type",
            options=["Candlestick", "Line"],
            horizontal=True,
            key=f"{key_prefix}-chart-type",
        )
        show_volume = st.checkbox("Show volume", value=True, key=f"{key_prefix}-show-volume")
    with control_col2:
        show_ema8 = st.checkbox("Show EMA8", value=True, key=f"{key_prefix}-show-ema8")
        show_ema21 = st.checkbox("Show EMA21", value=True, key=f"{key_prefix}-show-ema21")
        show_sma50 = st.checkbox("Show SMA50", value=True, key=f"{key_prefix}-show-sma50")
        show_sma100 = st.checkbox("Show SMA100", value=True, key=f"{key_prefix}-show-sma100")
        show_sma200 = st.checkbox("Show SMA200", value=True, key=f"{key_prefix}-show-sma200")
    with control_col3:
        show_ema8_breaks = st.checkbox("Show EMA8 Breaks", value=False, key=f"{key_prefix}-show-ema8-breaks")
        show_ema21_breaks = st.checkbox("Show EMA21 Breaks", value=False, key=f"{key_prefix}-show-ema21-breaks")
        show_sma50_breaks = st.checkbox("Show SMA50 Breaks", value=False, key=f"{key_prefix}-show-sma50-breaks")
        show_sma200_breaks = st.checkbox("Show SMA200 Breaks", value=False, key=f"{key_prefix}-show-sma200-breaks")
    with control_col4:
        show_golden_cross = st.checkbox("Show Golden Cross", value=False, key=f"{key_prefix}-show-golden-cross")
        show_death_cross = st.checkbox("Show Death Cross", value=False, key=f"{key_prefix}-show-death-cross")

    return {
        "timeframe": timeframe,
        "chart_type": chart_type,
        "show_volume": show_volume,
        "show_ema8": show_ema8,
        "show_ema21": show_ema21,
        "show_sma50": show_sma50,
        "show_sma100": show_sma100,
        "show_sma200": show_sma200,
        "show_ema8_breaks": show_ema8_breaks,
        "show_ema21_breaks": show_ema21_breaks,
        "show_sma50_breaks": show_sma50_breaks,
        "show_sma200_breaks": show_sma200_breaks,
        "show_gc_markers": show_golden_cross,
        "show_dc_markers": show_death_cross,
        "show_rsi": False,
    }


def slice_chart_frame_for_timeframe(chart_data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if chart_data.empty:
        return pd.DataFrame()

    days = TIMEFRAME_TO_DAYS.get(timeframe, TIMEFRAME_TO_DAYS["1Y"])
    end_date = chart_data["Date"].max()
    start_date = end_date - pd.Timedelta(days=days)
    sliced = chart_data.loc[chart_data["Date"] >= start_date].copy()

    if sliced.empty:
        return chart_data.copy()
    return sliced


def get_indicator_marker_config(indicator_column: str, short_label: str) -> dict:
    if indicator_column == "EMA8":
        return {"label": short_label, "color": EMA8_COLOR}
    if indicator_column == "EMA21":
        return {"label": short_label, "color": EMA21_COLOR}
    if indicator_column == "SMA50":
        return {"label": short_label, "color": SMA50_COLOR}
    if indicator_column == "SMA100":
        return {"label": short_label, "color": SMA100_COLOR}
    return {"label": short_label, "color": SMA200_COLOR}


def get_cross_events(chart_data: pd.DataFrame, indicator_column: str, label_prefix: str) -> pd.DataFrame:
    if chart_data.empty or indicator_column not in chart_data.columns:
        return pd.DataFrame()

    frame = chart_data[["Date", "High", "Low", "Close", indicator_column, "ATR14"]].dropna().copy()
    if len(frame.index) < 2:
        return pd.DataFrame()

    previous_close = frame["Close"].shift(1)
    previous_indicator = frame[indicator_column].shift(1)
    marker_config = get_indicator_marker_config(indicator_column, label_prefix)
    default_offset = frame["Close"] * 0.018
    offset = frame["ATR14"].fillna(default_offset).clip(lower=default_offset)

    reclaim_mask = (previous_close <= previous_indicator) & (frame["Close"] > frame[indicator_column])
    break_mask = (previous_close >= previous_indicator) & (frame["Close"] < frame[indicator_column])

    reclaim_events = frame.loc[reclaim_mask].copy()
    reclaim_events["Event"] = f"{marker_config['label']}+"
    reclaim_events["EventType"] = get_event_description(indicator_column, is_reclaim=True)
    reclaim_events["MAName"] = indicator_column
    reclaim_events["MarkerColor"] = marker_config["color"]
    reclaim_events["Symbol"] = "triangle-up"
    reclaim_events["MarkerY"] = reclaim_events["Low"] - offset.loc[reclaim_events.index] * 1.05
    reclaim_events["TextPosition"] = "bottom center"
    reclaim_events["MAValue"] = reclaim_events[indicator_column]

    break_events = frame.loc[break_mask].copy()
    break_events["Event"] = f"{marker_config['label']}-"
    break_events["EventType"] = get_event_description(indicator_column, is_reclaim=False)
    break_events["MAName"] = indicator_column
    break_events["MarkerColor"] = marker_config["color"]
    break_events["Symbol"] = "triangle-down"
    break_events["MarkerY"] = break_events["High"] + offset.loc[break_events.index] * 1.05
    break_events["TextPosition"] = "top center"
    break_events["MAValue"] = break_events[indicator_column]

    events = pd.concat([reclaim_events, break_events], ignore_index=True)
    if events.empty:
        return events

    return events.sort_values("Date").tail(16)


def get_all_marker_events(chart_data: pd.DataFrame, controls: dict) -> pd.DataFrame:
    events = []

    if controls.get("show_ema8_breaks"):
        events.append(get_cross_events(chart_data, "EMA8", "EMA8"))
    if controls.get("show_ema21_breaks"):
        events.append(get_cross_events(chart_data, "EMA21", "EMA21"))
    if controls.get("show_sma50_breaks"):
        events.append(get_cross_events(chart_data, "SMA50", "SMA50"))
    if controls.get("show_sma200_breaks"):
        events.append(get_cross_events(chart_data, "SMA200", "SMA200"))
    if controls.get("show_gc_markers"):
        events.append(get_cross_events_between_mas(chart_data, event_name="Golden Cross", color=GC_COLOR, bullish_cross=True))
    if controls.get("show_dc_markers"):
        events.append(get_cross_events_between_mas(chart_data, event_name="Death Cross", color=DC_COLOR, bullish_cross=False))

    valid_events = [event for event in events if not event.empty]
    if not valid_events:
        return pd.DataFrame()

    return pd.concat(valid_events, ignore_index=True)


def get_cross_events_between_mas(
    chart_data: pd.DataFrame,
    event_name: str,
    color: str,
    bullish_cross: bool,
) -> pd.DataFrame:
    required_columns = ["Date", "High", "Low", "Close", "SMA50", "SMA200", "ATR14"]
    if chart_data.empty or any(column not in chart_data.columns for column in required_columns):
        return pd.DataFrame()

    frame = chart_data[required_columns].dropna().copy()
    if len(frame.index) < 2:
        return pd.DataFrame()

    previous_sma50 = frame["SMA50"].shift(1)
    previous_sma200 = frame["SMA200"].shift(1)
    default_offset = chart_data["Close"] * 0.018
    offset = frame["ATR14"].fillna(default_offset).clip(lower=default_offset.loc[frame.index])

    if bullish_cross:
        cross_mask = (previous_sma50 <= previous_sma200) & (frame["SMA50"] > frame["SMA200"])
        symbol = "diamond"
        marker_y = frame["Low"] - offset * 0.9
        text_position = "bottom center"
    else:
        cross_mask = (previous_sma50 >= previous_sma200) & (frame["SMA50"] < frame["SMA200"])
        symbol = "x"
        marker_y = frame["High"] + offset * 0.9
        text_position = "top center"

    events = frame.loc[cross_mask].copy()
    if events.empty:
        return events

    events["Event"] = event_name
    events["EventType"] = "Golden Cross" if bullish_cross else "Death Cross"
    events["MAName"] = "SMA200"
    events["MAValue"] = events["SMA200"]
    events["MarkerColor"] = color
    events["Symbol"] = symbol
    events["MarkerY"] = marker_y.loc[events.index]
    events["TextPosition"] = text_position

    return events.sort_values("Date").tail(6)


def add_price_trace(fig: go.Figure, chart_data: pd.DataFrame, chart_type: str) -> None:
    hover_template = (
        "Date=%{x|%Y-%m-%d}<br>"
        "Open=%{customdata[0]:.2f}<br>"
        "High=%{customdata[1]:.2f}<br>"
        "Low=%{customdata[2]:.2f}<br>"
        "Close=%{y:.2f}<br>"
        "Volume=%{customdata[3]:,.0f}<extra></extra>"
    )
    custom_data = chart_data[["Open", "High", "Low", "Volume"]].to_numpy()

    if chart_type == "Line":
        fig.add_trace(
            go.Scatter(
                x=chart_data["Date"],
                y=chart_data["Close"],
                mode="lines",
                name="Close",
                line={"color": CLOSE_COLOR, "width": 2},
                customdata=custom_data,
                hovertemplate=hover_template,
            ),
            row=1,
            col=1,
        )
        return

    fig.add_trace(
        go.Candlestick(
            x=chart_data["Date"],
            open=chart_data["Open"],
            high=chart_data["High"],
            low=chart_data["Low"],
            close=chart_data["Close"],
            name="Price",
            increasing={"line": {"color": UP_COLOR, "width": 1.2}, "fillcolor": UP_COLOR},
            decreasing={"line": {"color": DOWN_COLOR, "width": 1.2}, "fillcolor": DOWN_COLOR},
        ),
        row=1,
        col=1,
    )


def add_indicator_trace(fig: go.Figure, chart_data: pd.DataFrame, column: str, color: str, row: int = 1) -> None:
    fig.add_trace(
        go.Scatter(
            x=chart_data["Date"],
            y=chart_data[column],
            mode="lines",
            name=column,
            line={"color": color, "width": 1.8},
            hovertemplate=f"{column}=%{{y:.2f}}<extra></extra>",
            connectgaps=False,
        ),
        row=row,
        col=1,
    )


def get_moving_average_side_label_entries(chart_data: pd.DataFrame, controls: dict) -> list[dict]:
    configs = (
        ("EMA8", EMA8_COLOR, "show_ema8"),
        ("EMA21", EMA21_COLOR, "show_ema21"),
        ("SMA50", SMA50_COLOR, "show_sma50"),
        ("SMA100", SMA100_COLOR, "show_sma100"),
        ("SMA200", SMA200_COLOR, "show_sma200"),
    )
    entries = []
    for name, color, control_key in configs:
        if not controls.get(control_key) or name not in chart_data.columns:
            continue
        value = get_indicator_value(chart_data, name)
        if value is None:
            continue
        entries.append({"name": name, "value": value, "label_y": value, "color": color})
    return entries


def space_side_label_entries(chart_data: pd.DataFrame, entries: list[dict]) -> list[dict]:
    if not entries:
        return []

    numeric_columns = ["High", "Low", "Close"] + [entry["name"] for entry in entries if entry["name"] in chart_data.columns]
    values = chart_data[numeric_columns].select_dtypes(include="number").stack().dropna()
    if values.empty:
        return entries

    low = float(values.min())
    high = float(values.max())
    span = max(high - low, 1.0)
    min_gap = span * 0.035
    lower_bound = low - span * 0.03
    upper_bound = high + span * 0.03

    spaced = sorted(({**entry} for entry in entries), key=lambda entry: entry["value"])
    for index, entry in enumerate(spaced):
        if index == 0:
            entry["label_y"] = max(entry["value"], lower_bound)
        else:
            entry["label_y"] = max(entry["value"], spaced[index - 1]["label_y"] + min_gap)

    if spaced[-1]["label_y"] > upper_bound:
        spaced[-1]["label_y"] = upper_bound
        for index in range(len(spaced) - 2, -1, -1):
            spaced[index]["label_y"] = min(spaced[index]["label_y"], spaced[index + 1]["label_y"] - min_gap)

    return spaced


def add_moving_average_side_labels(fig: go.Figure, chart_data: pd.DataFrame, controls: dict) -> None:
    entries = space_side_label_entries(chart_data, get_moving_average_side_label_entries(chart_data, controls))
    for entry in entries:
        fig.add_annotation(
            x=1.006,
            y=entry["label_y"],
            xref="paper",
            yref="y",
            text=f"{entry['name']} {entry['value']:.2f}",
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            align="left",
            bgcolor=entry["color"],
            bordercolor=entry["color"],
            borderpad=3,
            font={"color": "#ffffff", "size": 11, "family": "Arial, sans-serif"},
            opacity=0.96,
        )


def add_volume_trace(fig: go.Figure, chart_data: pd.DataFrame) -> None:
    fig.add_trace(
        go.Bar(
            x=chart_data["Date"],
            y=chart_data["Volume"],
            name="Volume",
            marker={"color": chart_data["VolumeColor"]},
            hovertemplate="Date=%{x|%Y-%m-%d}<br>Volume=%{y:,.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )


def add_marker_traces(fig: go.Figure, marker_events: pd.DataFrame, ticker: str | None = None) -> None:
    if marker_events.empty:
        return

    show_text = len(marker_events.index) <= 24
    ticker_label = ticker or "Ticker"
    for event_name, event_frame in marker_events.groupby("Event"):
        custom_data = pd.DataFrame(
            {
                "Ticker": ticker_label,
                "EventType": event_frame.get("EventType", event_frame["Event"]),
                "Close": event_frame["Close"],
                "MAName": event_frame.get("MAName", pd.Series(["MA"] * len(event_frame.index), index=event_frame.index)),
                "MAValue": event_frame.get("MAValue", event_frame["MarkerY"]),
            }
        ).to_numpy()
        fig.add_trace(
            go.Scatter(
                x=event_frame["Date"],
                y=event_frame["MarkerY"],
                mode="markers+text" if show_text else "markers",
                name=event_name,
                text=event_frame["Event"] if show_text else None,
                textposition=event_frame["TextPosition"].iloc[0],
                textfont={"size": 11, "color": TEXT_COLOR, "family": "Arial, sans-serif"},
                marker={
                    "symbol": event_frame["Symbol"].iloc[0],
                    "size": 11 if show_text else 9,
                    "color": event_frame["MarkerColor"].iloc[0],
                    "line": {"width": 1.5, "color": MARKER_OUTLINE},
                },
                customdata=custom_data,
                hovertemplate=(
                    "%{customdata[0]}<br>"
                    "%{x|%Y-%m-%d}<br>"
                    "%{customdata[1]}<br>"
                    "Close: %{customdata[2]:.2f}<br>"
                    "%{customdata[3]}: %{customdata[4]:.2f}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )


def get_initial_range(chart_data: pd.DataFrame, timeframe: str) -> list[pd.Timestamp]:
    default_window = 120
    if timeframe == "6M":
        default_window = 90
    elif timeframe == "2Y":
        default_window = 150
    elif timeframe == "5Y":
        default_window = 180

    if len(chart_data.index) <= default_window:
        return [chart_data["Date"].iloc[0], chart_data["Date"].iloc[-1]]
    return [chart_data["Date"].iloc[-default_window], chart_data["Date"].iloc[-1]]


def add_rsi_trace(fig: go.Figure, chart_data: pd.DataFrame, row: int) -> bool:
    if chart_data.empty or "RSI14" not in chart_data.columns:
        return False

    rsi_data = chart_data[["Date", "RSI14"]].dropna()
    if rsi_data.empty:
        return False

    fig.add_trace(
        go.Scatter(
            x=rsi_data["Date"],
            y=rsi_data["RSI14"],
            mode="lines",
            name="RSI(14)",
            line={"color": RSI_COLOR, "width": 1.8},
            hovertemplate="RSI(14)=%{y:.1f}<extra></extra>",
        ),
        row=row,
        col=1,
    )
    for level, color in ((70, "#f97316"), (50, "#94a3b8"), (30, "#22c55e")):
        fig.add_hline(
            y=level,
            line={"color": color, "width": 1},
            opacity=0.7,
            row=row,
            col=1,
        )
    return True


def style_figure(fig: go.Figure, chart_data: pd.DataFrame, controls: dict, row_count: int, volume_row: int | None, rsi_row: int | None) -> go.Figure:
    show_volume = controls.get("show_volume", True)
    timeframe = controls.get("timeframe", "1Y")
    initial_range = get_initial_range(chart_data, timeframe)
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        font={"color": TEXT_COLOR},
        hovermode="x unified",
        dragmode="pan",
        margin={"l": 20, "r": 116, "t": 10, "b": 20},
        height=680 if show_volume else 560,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        xaxis_rangeslider_visible=False,
        uirevision=f"stock-chart-{controls.get('timeframe', '1Y')}",
    )
    for row in range(1, row_count + 1):
        fig.update_xaxes(
            showgrid=True,
            gridcolor=GRID_COLOR,
            showspikes=True,
            spikecolor="#64748b",
            spikesnap="cursor",
            range=initial_range,
            rangeslider_visible=False,
            row=row,
            col=1,
        )
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, fixedrange=False, zerolinecolor=GRID_COLOR, side="right", row=1, col=1)
    if volume_row is not None:
        fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, fixedrange=False, zerolinecolor=GRID_COLOR, row=volume_row, col=1)
    if rsi_row is not None:
        fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, fixedrange=False, zerolinecolor=GRID_COLOR, range=[0, 100], row=rsi_row, col=1)
    if show_volume or rsi_row is not None:
        fig.update_xaxes(showticklabels=False, row=1, col=1)
    return fig


def build_figure(chart_data: pd.DataFrame, controls: dict, ticker: str | None = None) -> go.Figure:
    show_volume = controls.get("show_volume", True)
    show_rsi = controls.get("show_rsi", False) and "RSI14" in chart_data.columns and not chart_data["RSI14"].dropna().empty
    extra_rows = int(show_volume) + int(show_rsi)
    if show_volume and show_rsi:
        row_heights = [0.64, 0.20, 0.16]
    elif show_volume:
        row_heights = [0.74, 0.26]
    elif show_rsi:
        row_heights = [0.78, 0.22]
    else:
        row_heights = [1.0]
    specs = [[{"secondary_y": False}] for _ in range(1 + extra_rows)]
    fig = make_subplots(
        rows=1 + extra_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=row_heights,
        specs=specs,
    )

    add_price_trace(fig, chart_data, controls.get("chart_type", "Candlestick"))

    if controls.get("show_ema8"):
        add_indicator_trace(fig, chart_data, "EMA8", EMA8_COLOR)
    if controls.get("show_ema21"):
        add_indicator_trace(fig, chart_data, "EMA21", EMA21_COLOR)
    if controls.get("show_sma50"):
        add_indicator_trace(fig, chart_data, "SMA50", SMA50_COLOR)
    if controls.get("show_sma100"):
        add_indicator_trace(fig, chart_data, "SMA100", SMA100_COLOR)
    if controls.get("show_sma200"):
        add_indicator_trace(fig, chart_data, "SMA200", SMA200_COLOR)

    add_moving_average_side_labels(fig, chart_data, controls)

    marker_events = get_all_marker_events(chart_data, controls)
    add_marker_traces(fig, marker_events, ticker=ticker)

    volume_row = None
    rsi_row = None
    if show_volume:
        volume_row = 2
        add_volume_trace(fig, chart_data)
    if show_rsi:
        rsi_row = 3 if show_volume else 2
        add_rsi_trace(fig, chart_data, row=rsi_row)

    return style_figure(
        fig=fig,
        chart_data=chart_data,
        controls=controls,
        row_count=1 + extra_rows,
        volume_row=volume_row,
        rsi_row=rsi_row,
    )


def render_stock_chart(history: pd.DataFrame, key_prefix: str = "detail-chart", ticker: str | None = None) -> None:
    if history.empty or len(history.index) < 30:
        st.info("No chart data available.")
        return

    controls = render_chart_controls(key_prefix)
    full_chart_data = build_chart_frame(history)
    chart_data = slice_chart_frame_for_timeframe(full_chart_data, controls.get("timeframe", "1Y"))
    if chart_data.empty or len(chart_data.index) < 30:
        st.info("Not enough chart data for the selected timeframe.")
        return

    render_chart_legend(chart_data)
    marker_controls_enabled = any(
        controls.get(key)
        for key in (
            "show_ema8_breaks",
            "show_ema21_breaks",
            "show_sma50_breaks",
            "show_sma200_breaks",
            "show_gc_markers",
            "show_dc_markers",
        )
    )
    if marker_controls_enabled:
        render_marker_legend()
    if controls.get("show_rsi") and ("RSI14" not in chart_data.columns or chart_data["RSI14"].dropna().empty):
        st.caption("RSI(14) is unavailable for the selected data window.")

    figure = build_figure(chart_data, controls, ticker=ticker)
    st.plotly_chart(
        figure,
        use_container_width=True,
        config={
            "scrollZoom": True,
            "displayModeBar": True,
            "doubleClick": "reset",
            "displaylogo": False,
        },
    )
