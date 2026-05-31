from datetime import date, datetime
from html import escape
import re

import streamlit as st

from charts import build_technical_readout
from data_fetch import build_fundamental_quality_summary, get_default_wacc_inputs, get_sec_dcf_inputs
from dcf import DcfAssumptions, build_sensitivity_table, calculate_damodaran_dcf, calculate_scenarios, safe_float
from scoring import get_score_explanation
from scanner import SCAN_MODE_CONFIG, filter_setup_mode, get_scan_mode_default_limit
from symbol_universe import EX_MAJOR_INDEXES_UNIVERSE, FULL_MARKET_UNIVERSE, get_symbol_metadata
from ticker_lookup import build_ticker_lookup, ticker_from_option
from universes import get_universe_count, get_universe_label, get_universe_note, get_universe_option_label, get_universe_warning, is_universe_available
from utils import format_change_pct, format_price, format_section_rows


def format_ratio(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2f}"


def format_tags(tags: list[str]) -> str:
    return ", ".join(tags[:6]) or "None"


def get_saved_setups() -> list[dict]:
    if "saved_setups" not in st.session_state:
        st.session_state["saved_setups"] = []
    return st.session_state["saved_setups"]


def is_setup_saved(ticker: str) -> bool:
    return any(setup.get("ticker") == ticker for setup in get_saved_setups())


def find_scan_row(ticker: str, scan_results: dict | None) -> dict | None:
    if not scan_results:
        return None
    for section_name in ("all_ranked", "top_setups"):
        for row in scan_results.get(section_name, []):
            if row.get("ticker") == ticker:
                return row
    return None


def save_setup(row: dict) -> bool:
    ticker = row.get("ticker")
    if not ticker or is_setup_saved(ticker):
        return False

    get_saved_setups().append(
        {
            "ticker": ticker,
            "date_added": date.today().isoformat(),
            "price_at_add": row.get("price"),
            "signal_score_at_add": row.get("signal_score"),
            "current_signal_score": row.get("signal_score"),
            "status": row.get("long_status") or row.get("label", "N/A"),
            "tags": format_tags(row.get("tags", [])),
            "notes": "",
        }
    )
    return True


def remove_saved_setup(ticker: str) -> None:
    st.session_state["saved_setups"] = [
        setup for setup in get_saved_setups()
        if setup.get("ticker") != ticker
    ]


def format_optional_number(value: object, decimals: int = 2) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


def format_score(value: object, max_score: int | float, decimals: int = 1) -> str:
    numeric_value = safe_numeric(value)
    if numeric_value is None:
        return "N/A"
    max_numeric = safe_numeric(max_score)
    if max_numeric is None:
        return "N/A"
    max_label = f"{max_numeric:.0f}" if float(max_numeric).is_integer() else f"{max_numeric:g}"
    return f"{numeric_value:.{decimals}f} / {max_label}"


def format_historical_edge_score(row: dict) -> str:
    edge_score = safe_numeric(row.get("historical_edge_score"))
    if edge_score is not None:
        return format_score(edge_score, 10)

    sample_size = safe_numeric(row.get("historical_sample_size"))
    hit_rate = safe_numeric(row.get("historical_hit_rate"))
    if sample_size is None or sample_size < 10 or hit_rate is None:
        return "N/A"

    display_score = max(0.0, min(10.0, (hit_rate - 35.0) / 4.0))
    return format_score(display_score, 10)


def truncate_text(value: object, max_length: int = 120) -> str:
    if value is None:
        return "N/A"
    text = str(value).strip()
    if not text:
        return "N/A"
    if len(text) <= max_length:
        return text
    return f"{text[:max_length - 3].rstrip()}..."


def format_large_number(value: object) -> str:
    if value is None:
        return "N/A"
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return "N/A"

    for threshold, suffix in ((1_000_000_000_000, "T"), (1_000_000_000, "B"), (1_000_000, "M")):
        if abs(numeric_value) >= threshold:
            return f"${numeric_value / threshold:.2f}{suffix}"
    return f"${numeric_value:,.0f}"


def format_market_cap(value: object) -> str:
    return format_large_number(value)


def format_compact_number(value: object) -> str:
    if value is None:
        return "N/A"
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return "N/A"

    for threshold, suffix, decimals in (
        (1_000_000_000_000, "T", 2),
        (1_000_000_000, "B", 1),
        (1_000_000, "M", 1),
    ):
        if abs(numeric_value) >= threshold:
            return f"{numeric_value / threshold:.{decimals}f}{suffix}"
    return f"{numeric_value:,.0f}"


def format_percent_ratio(value: object) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def safe_numeric(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_text_value(value: object) -> str:
    if value is None or value == "":
        return "N/A"
    return str(value)


def format_multiple(value: object) -> str:
    numeric_value = safe_numeric(value)
    if numeric_value is None:
        return "N/A"
    return f"{numeric_value:.2f}x"


def normalize_debt_to_equity(value: object) -> float | None:
    numeric_value = safe_numeric(value)
    if numeric_value is None:
        return None
    if abs(numeric_value) > 20:
        return numeric_value / 100
    return numeric_value


def format_debt_to_equity(value: object) -> str:
    numeric_value = normalize_debt_to_equity(value)
    if numeric_value is None:
        return "N/A"
    return f"{numeric_value:.2f}x"


def format_days(value: object) -> str:
    numeric_value = safe_numeric(value)
    if numeric_value is None:
        return "N/A"
    return f"{numeric_value:.2f} days"


def format_signed_percent_optional(value: object, decimals: int = 1) -> str:
    numeric_value = safe_numeric(value)
    if numeric_value is None:
        return "N/A"
    return f"{numeric_value:+.{decimals}f}%"


def format_percent_points(value: object, decimals: int = 1) -> str:
    numeric_value = safe_numeric(value)
    if numeric_value is None:
        return "N/A"
    return f"{numeric_value:.{decimals}f}%"


def calculate_percent_change(current_value: object, reference_value: object) -> float | None:
    current = safe_numeric(current_value)
    reference = safe_numeric(reference_value)
    if current is None or reference in (None, 0):
        return None
    return ((current / reference) - 1) * 100


def calculate_short_interest_change(fundamentals: dict) -> float | None:
    shares_short = safe_numeric(fundamentals.get("sharesShort"))
    prior_month = safe_numeric(fundamentals.get("sharesShortPriorMonth"))
    if shares_short is None or prior_month in (None, 0):
        return None
    return ((shares_short - prior_month) / prior_month) * 100


def calculate_ratio(numerator: object, denominator: object) -> float | None:
    numerator_value = safe_numeric(numerator)
    denominator_value = safe_numeric(denominator)
    if numerator_value is None or denominator_value in (None, 0):
        return None
    return numerator_value / denominator_value


def get_context_label(metric_name: str, value: object) -> str:
    numeric_value = safe_numeric(value)
    if numeric_value is None:
        return ""
    if metric_name == "short_percent_float":
        if numeric_value >= 0.20:
            return "High short interest"
        if numeric_value >= 0.08:
            return "Moderate short interest"
        return "Low short interest"
    if metric_name == "debt_to_equity":
        normalized_value = normalize_debt_to_equity(value)
        if normalized_value is None:
            return ""
        if normalized_value >= 1.5:
            return "High leverage"
        if normalized_value >= 0.6:
            return "Moderate leverage"
        return "Low leverage"
    if metric_name == "from_52w_high":
        if numeric_value >= -5:
            return "Near highs"
        return "Off highs"
    if metric_name == "dollar_volume":
        if numeric_value >= 1_000_000_000:
            return "High liquidity"
        if numeric_value >= 100_000_000:
            return "Moderate liquidity"
        return "Lower liquidity"
    return ""


def render_metric_with_context(label: str, value: str, context: str = "") -> None:
    st.metric(label, value)
    if context:
        st.caption(context)


def format_compact_money(value: object, signed: bool = False) -> str:
    if value is None:
        return "N/A"
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return "N/A"

    sign = ""
    if signed and numeric_value > 0:
        sign = "+"
    elif signed and numeric_value < 0:
        sign = "-"
    absolute_value = abs(numeric_value)

    for threshold, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")):
        if absolute_value >= threshold:
            return f"{sign}${absolute_value / threshold:.1f}{suffix}"
    return f"{sign}${absolute_value:,.0f}"


def format_percent_optional(value: object) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def format_filing_month(value: str | None) -> str:
    if not value:
        return ""
    try:
        return datetime.strptime(value, "%Y-%m-%d").strftime("%b %Y")
    except ValueError:
        return value


def format_signed_percent(value: object) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):+.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def format_plain_percent(value: object) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.0f}%"
    except (TypeError, ValueError):
        return "N/A"


def render_explanation_box(
    title: str,
    description: str,
    calculation: str,
    why_it_matters: str,
) -> None:
    st.markdown(f"**{title}**")
    st.write(description)
    st.caption(f"Calculation: {calculation}")
    st.caption(f"Why it matters: {why_it_matters}")


def render_metric_explanation(key: str) -> None:
    explanation = get_score_explanation(key)
    render_explanation_box(
        title=explanation.get("title", key.replace("_", " ").title()),
        description=explanation.get("description", "No explanation is available for this item yet."),
        calculation=explanation.get("calculation", "Not specified."),
        why_it_matters=explanation.get("why_it_matters", "Not specified."),
    )


def render_signal_score_methodology() -> None:
    with st.expander("How the Signal Score is Calculated"):
        st.markdown(
            """
Signal Score /100: Market Regime /15 + Trend Quality /15 + Base Quality /20 + Compression /15 + Trigger /15 + Risk/Reward /10 + Historical Edge /10, minus penalties.

Market Regime /15: Measures whether SPY, QQQ, and IWM support long trades. Risk-On markets improve breakout reliability. Risk-Off markets penalize long setups.

Trend Quality /15: Measures whether the stock is above key daily moving averages and whether those moving averages are rising. Stocks above EMA21, SMA50, and SMA200 with rising averages score higher.

Base Quality /20: Measures whether the stock has formed a clean trading base. Better bases are controlled, not too deep, have clear resistance, show higher lows, and are not overly volatile.

Compression /15: Measures whether volatility and range are tightening before a potential breakout. Compression is based on ATR%, recent trading range, and volume drying up.

Trigger /15: Measures whether the stock is actually breaking out or reclaiming an important level. Strong triggers include closes above resistance, strong close location, and above-average volume.

Risk/Reward /10: Measures whether the entry is still efficient. Stocks too far above support or EMA21 score lower because the stop would be too wide.

Historical Edge /10: Measures how similar setups performed in the stock's own historical data. This includes hit rate, median forward return, and sample size when enough historical examples exist.
            """
        )


def render_scanner_explanations() -> None:
    with st.expander("How to read these scanner results"):
        st.write(
            "Scanner filters favor liquid stocks with usable price history, acceptable volatility, and enough "
            "technical structure to evaluate. Rows stay compact so the ranked lists remain fast to scan."
        )
        for key in ("signal_score", "base_quality", "trigger_quality", "risk_entry", "relative_volume"):
            render_metric_explanation(key)


def render_ma_scanner_explanations() -> None:
    with st.expander("Moving-average and reaction terms"):
        for key in ("ema21", "sma50", "sma200", "golden_cross", "death_cross", "ma_breach", "historical_reaction"):
            render_metric_explanation(key)


def render_technical_metric_explanations() -> None:
    with st.expander("Key technical metric definitions"):
        for key in (
            "ema8",
            "ema21",
            "sma50",
            "sma200",
            "golden_cross",
            "death_cross",
            "volume",
            "relative_volume",
            "ma_breach",
            "historical_reaction",
        ):
            render_metric_explanation(key)


def get_default_ticker_limit(universe_name: str, scan_mode: str = "Fast Scan") -> int:
    if str(universe_name).strip().upper() in {FULL_MARKET_UNIVERSE, EX_MAJOR_INDEXES_UNIVERSE}:
        return 0
    universe_count = get_universe_count(universe_name)
    if universe_count:
        return universe_count
    return get_scan_mode_default_limit(scan_mode)


def get_average_volume_presets() -> dict[str, int]:
    return {
        "Off": 0,
        "100K+": 100_000,
        "300K+": 300_000,
        "500K+": 500_000,
        "1M+": 1_000_000,
        "2M+": 2_000_000,
    }


def get_dollar_volume_presets() -> dict[str, float | None]:
    return {
        "Off": 0.0,
        "$500K+": 500_000.0,
        "$1M+": 1_000_000.0,
        "$2M+": 2_000_000.0,
        "$5M+": 5_000_000.0,
        "$10M+": 10_000_000.0,
        "$25M+": 25_000_000.0,
        "Custom": None,
    }


def get_max_price_presets() -> dict[str, float | None]:
    return {
        "Off": None,
        "$10": 10.0,
        "$20": 20.0,
        "$50": 50.0,
        "$100": 100.0,
        "$250": 250.0,
        "Custom": None,
    }


def parse_dollar_volume_input(value: str) -> float | None:
    cleaned_value = str(value).strip().upper().replace("$", "").replace(",", "")
    if not cleaned_value:
        return None

    multiplier = 1.0
    if cleaned_value.endswith("K"):
        multiplier = 1_000.0
        cleaned_value = cleaned_value[:-1]
    elif cleaned_value.endswith("M"):
        multiplier = 1_000_000.0
        cleaned_value = cleaned_value[:-1]
    elif cleaned_value.endswith("B"):
        multiplier = 1_000_000_000.0
        cleaned_value = cleaned_value[:-1]

    try:
        return max(float(cleaned_value) * multiplier, 0.0)
    except ValueError:
        return None


def render_min_average_volume_control(key_prefix: str = "scan") -> int:
    presets = get_average_volume_presets()
    selected_preset = st.selectbox(
        "Min Average Volume",
        list(presets.keys()),
        index=2,
        key=f"{key_prefix}-min-average-volume-preset",
    )
    return presets[selected_preset]


def render_min_dollar_volume_control(universe_name: str, key_prefix: str = "scan") -> float:
    presets = get_dollar_volume_presets()
    default_label = "$5M+" if universe_name == "RUSSELL3000" else "$2M+"
    default_index = list(presets.keys()).index(default_label)
    selected_preset = st.selectbox(
        "Min Dollar Volume",
        list(presets.keys()),
        index=default_index,
        key=f"{key_prefix}-min-dollar-volume-preset",
        help="Dollar volume = price x volume. This helps filter for liquid, tradable stocks.",
    )

    if selected_preset != "Custom":
        return presets[selected_preset]

    custom_value = st.text_input(
        "Custom Min Dollar Volume",
        value="2M",
        key=f"{key_prefix}-min-dollar-volume-custom",
        help="Use shorthand like 500K, 2M, or 25M.",
    )
    parsed_value = parse_dollar_volume_input(custom_value)
    if parsed_value is None:
        st.warning("Enter a valid dollar volume such as 500K, 2M, or 25M. Falling back to $2M.")
        return 2_000_000.0
    return parsed_value


def render_max_price_control() -> float | None:
    presets = get_max_price_presets()
    selected_preset = st.selectbox(
        "Max Price ($)",
        list(presets.keys()),
        index=0,
        key="scan-max-price-preset",
    )
    if selected_preset == "Off":
        return None
    if selected_preset != "Custom":
        return presets[selected_preset]

    return st.number_input(
        "Custom Max Price ($)",
        min_value=0.0,
        value=100.0,
        step=1.0,
        format="%.2f",
        key="scan-max-price-custom",
    )


def render_controls(universe_names: list[str]) -> tuple[str, str, int, int, int, str, dict, bool, bool, bool]:
    st.title("Stock Screener")
    raw_search_ticker = render_ticker_search()

    selected_universe = st.selectbox("Universe", universe_names, index=0, format_func=get_universe_option_label)
    is_full_market = str(selected_universe).strip().upper() in {FULL_MARKET_UNIVERSE, EX_MAJOR_INDEXES_UNIVERSE}
    scan_mode = "Fast Scan"
    universe_note = get_universe_note(selected_universe)
    universe_warning = get_universe_warning(selected_universe)
    if universe_note:
        st.caption(universe_note)
    if universe_warning:
        st.warning(universe_warning)
    if is_full_market:
        st.warning("Full-market updates can take several minutes. Show Saved Results is the faster path once a snapshot exists.")

    min_score = 0
    st.markdown("### Scan")
    col1, col2 = st.columns([1.0, 2.6])
    with col1:
        max_results = st.slider("Max results", min_value=3, max_value=50, value=10, step=1)
    with col2:
        if is_full_market:
            max_tickers = 0
        else:
            universe_count = get_universe_count(selected_universe)
            max_tickers = min(get_default_ticker_limit(selected_universe, scan_mode), max(10, universe_count or 2000))
        run_label = "Show Saved Results" if is_full_market else "Show Results"
        refresh_label = "Update Full-Market Data" if is_full_market else "Update Market Data"
        action_col, refresh_col = st.columns([1.0, 1.2])
        with action_col:
            run_scan = st.button(run_label, type="primary", disabled=not is_universe_available(selected_universe))
        with refresh_col:
            refresh_snapshot = st.button(
                refresh_label,
                disabled=not is_universe_available(selected_universe),
            )

    with st.expander("Advanced Scanner Filters", expanded=False):
        filter_reset_version = st.session_state.get("advanced_filter_reset_version", 0)
        filter_key_prefix = f"scan-filter-{selected_universe}-{filter_reset_version}"
        if st.button("Reset Filters", key=f"{filter_key_prefix}-reset"):
            st.session_state["advanced_filter_reset_version"] = filter_reset_version + 1
            st.rerun()
        calculate_historical_edge = st.checkbox(
            "Calculate Historical Edge on refresh",
            value=False,
            key="calculate-historical-edge",
            help="Optional and slower. Used only when rebuilding market data.",
        )
        if is_full_market and calculate_historical_edge:
            st.warning("Historical Edge on the full NYSE + Nasdaq universe can be slow.")
        price_col, volume_col, trend_col = st.columns([1.2, 1.2, 1.0])
        with price_col:
            use_price_filter = st.checkbox("Use price filter", value=False, key=f"{filter_key_prefix}-use-price-filter")
            min_price = st.number_input(
                "Min price ($)",
                min_value=0.0,
                value=2.00,
                step=1.00,
                format="%.2f",
                key=f"{filter_key_prefix}-min-price",
                disabled=not use_price_filter,
            )
            max_price = st.number_input(
                "Max price ($)",
                min_value=0.0,
                value=1000.00,
                step=1.00,
                format="%.2f",
                key=f"{filter_key_prefix}-max-price",
                disabled=not use_price_filter,
            )
            if use_price_filter and max_price < min_price:
                st.warning("Max price is below Min price. No scanner rows will match until the range is corrected.")
        with volume_col:
            use_volume_filter = st.checkbox("Use volume filter", value=False, key=f"{filter_key_prefix}-use-volume-filter")
            min_avg_volume = st.number_input(
                "Minimum average volume",
                min_value=0,
                value=500_000,
                step=50_000,
                key=f"{filter_key_prefix}-min-avg-volume",
                disabled=not use_volume_filter,
            )
            use_relative_volume_filter = st.checkbox("Use relative volume filter", value=False, key=f"{filter_key_prefix}-use-relative-volume-filter")
            min_relative_volume = st.number_input(
                "Minimum relative volume",
                min_value=0.0,
                value=1.0,
                step=0.1,
                format="%.2f",
                key=f"{filter_key_prefix}-min-relative-volume",
                disabled=not use_relative_volume_filter,
            )
        with trend_col:
            above_ema21 = st.checkbox("Only show stocks above EMA21", value=False, key=f"{filter_key_prefix}-above-ema21")
            above_sma50 = st.checkbox("Only show stocks above SMA50", value=False, key=f"{filter_key_prefix}-above-sma50")
            above_sma200 = st.checkbox("Only show stocks above SMA200", value=False, key=f"{filter_key_prefix}-above-sma200")
            trend_status_filter = st.selectbox(
                "Trend Status",
                [
                    "All",
                    "Strong Uptrend",
                    "Constructive Uptrend",
                    "Watchlist",
                    "Neutral / Mixed",
                    "Weak / Avoid",
                    "Downtrend / Avoid",
                    "Too Extended",
                    "Insufficient Data",
                ],
                index=0,
                key=f"{filter_key_prefix}-trend-status-filter",
                help="Filters by the plain-English trend label computed from price, moving averages, alignment, and slope.",
            )

    scanner_filters = {
        "use_price_filter": use_price_filter,
        "min_price": float(min_price) if use_price_filter else None,
        "max_price": float(max_price) if use_price_filter else None,
        "use_volume_filter": use_volume_filter,
        "min_avg_volume": int(min_avg_volume) if use_volume_filter else None,
        "use_relative_volume_filter": use_relative_volume_filter,
        "min_relative_volume": float(min_relative_volume) if use_relative_volume_filter else None,
        "use_market_cap_filter": False,
        "min_market_cap": None,
        "max_market_cap": None,
        "above_ema21": above_ema21,
        "above_sma50": above_sma50,
        "above_sma200": above_sma200,
        "use_trend_status_filter": trend_status_filter != "All",
        "trend_status_filter_values": [trend_status_filter] if trend_status_filter != "All" else [],
        "display_limit": max_results,
    }

    return raw_search_ticker, selected_universe, min_score, max_results, max_tickers, scan_mode, scanner_filters, refresh_snapshot, run_scan, calculate_historical_edge


def render_ma_scanner_controls(universe_names: list[str], lookback_options: list[str]) -> tuple[str, str, str, list[str], int, int, int, bool]:
    st.markdown("### MA Reclaim / Break Scanner")
    st.caption(
        "This scanner looks for stocks that recently closed back above or below selected moving averages. "
        "Reclaims can show improving momentum, while breaks can show weakening trend structure."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        universe_name = st.selectbox("MA Universe", universe_names, index=0, key="ma-universe", format_func=get_universe_label)
        lookback_label = st.selectbox("Lookback", lookback_options, index=1, key="ma-lookback")
        universe_warning = get_universe_warning(universe_name)
        if get_universe_note(universe_name):
            st.caption(get_universe_note(universe_name))
        if universe_warning:
            st.warning(universe_warning)
    with col2:
        event_type = st.selectbox(
            "Event Type",
            ["Reclaims only", "Breaks only", "Both"],
            index=0,
            key="ma-event-type",
        )
        selected_mas = st.multiselect(
            "Moving Averages",
            ["EMA8", "EMA21", "SMA50", "SMA200"],
            default=["EMA21", "SMA50"],
            key="ma-selected-mas",
        )
    with col3:
        min_score = st.slider("MA Min signal score", min_value=0, max_value=100, value=40, step=5, key="ma-min-score")
        max_results = st.slider("MA Max results", min_value=3, max_value=50, value=15, step=1, key="ma-max-results")
        max_tickers = st.slider(
            "MA Max tickers",
            min_value=10,
            max_value=1000,
            value=get_default_ticker_limit(universe_name),
            step=10,
            key=f"ma-max-tickers-{universe_name}",
        )
        run_scan = st.button("Run MA scan", disabled=not is_universe_available(universe_name), key="ma-run-scan")

    return universe_name, lookback_label, event_type, selected_mas, min_score, max_results, max_tickers, run_scan


def render_ticker_search() -> str:
    lookup_rows = build_ticker_lookup()
    labels = [row["label"] for row in lookup_rows]
    selected_option = st.selectbox(
        "Search Ticker",
        labels,
        index=None,
        placeholder="Enter ticker or company name (e.g. MSFT or Microsoft)",
        accept_new_options=True,
        key="ticker-search",
    )
    if not selected_option:
        return ""

    return ticker_from_option(selected_option)


def open_ticker(ticker: str) -> None:
    selected_ticker = str(ticker or "").strip().upper()
    if selected_ticker and selected_ticker != "N/A":
        st.session_state["selected_ticker"] = selected_ticker


def render_search_result(
    ticker: str,
    memberships: list[str],
    result_row: dict | None,
    error_message: str | None = None,
) -> None:
    if not ticker:
        return

    if error_message:
        st.error(error_message)
        return

    if not result_row:
        return

    metadata = get_symbol_metadata(ticker)
    if metadata:
        st.caption(
            f"Manual lookup: In NYSE/Nasdaq common stock list? Yes | "
            f"Exchange: {metadata.get('exchange') or 'N/A'} | Company: {metadata.get('company') or 'N/A'} | "
            f"Score: {format_score(result_row.get('signal_score'), 100)} | Trend Status: {result_row.get('long_status', 'N/A')}"
        )
    else:
        st.caption(
            f"Manual lookup: In NYSE/Nasdaq common stock list? No | "
            f"Score: {format_score(result_row.get('signal_score'), 100)} | Trend Status: {result_row.get('long_status', 'N/A')}"
        )
    st.button(f"Open {ticker}", key=f"search-open-{ticker}", on_click=open_ticker, args=(ticker,))


def render_empty_state(message: str = "No results matched the current filters.") -> None:
    st.warning(message)


def render_scan_summary(scan_results: dict) -> None:
    meta = scan_results.get("meta", {})
    if not meta:
        return

    st.caption(
        (
            f"Universe tickers found: {meta.get('universe_count', meta.get('requested_count', 0))} | "
            f"Tickers attempted: {meta.get('tickers_attempted', meta.get('requested_count', 0))} | "
            f"Download success: {meta.get('download_success_count', meta.get('valid_data_count', meta.get('scanned_count', 0)))} | "
            f"Failed/no data: {meta.get('missing_data_count', 0)} | "
            f"Invalid OHLCV: {meta.get('invalid_ohlcv_count', 0)} | "
            f"Not enough history: {meta.get('not_enough_history_count', 0)} | "
            f"Usable snapshot tickers: {meta.get('usable_snapshot_count', meta.get('snapshot_tickers_available', meta.get('scanned_count', 0)))} | "
            f"Final scanner results shown: {len(scan_results.get('all_ranked', []))} | "
            f"Runtime: {meta.get('duration_seconds', 0):.1f}s"
        )
    )
    if meta.get("raw_symbols_loaded") is not None:
        skipped_total = (
            int(meta.get("missing_data_count", 0) or 0)
            + int(meta.get("invalid_ohlcv_count", 0) or 0)
            + int(meta.get("not_enough_history_count", 0) or 0)
            + int(meta.get("snapshot_build_failed_count", 0) or 0)
        )
        st.caption(
            (
                f"Raw symbols loaded: {meta.get('raw_symbols_loaded', 0)} | "
                f"Non-common securities removed: {meta.get('non_common_removed', 0)} | "
                f"Final common stocks scanned: {meta.get('final_common_stocks', meta.get('universe_count', 0))} | "
                f"Skipped {skipped_total} tickers due to missing or invalid data."
            )
        )
        if meta.get("retry_missing_data_count"):
            st.caption(
                (
                    f"Missing-data retry attempted: {meta.get('retry_missing_data_count', 0)} | "
                    f"Recovered on retry: {meta.get('retry_recovered_count', 0)}"
                )
            )
        if meta.get("deep_scored_count"):
            st.caption(
                (
                    f"Deep-scored candidates: {meta.get('deep_scored_count', 0)} | "
                    f"Lightweight snapshot rows: {meta.get('lightweight_snapshot_count', 0)}"
                )
            )
    timings = meta.get("timings", {})
    if timings:
        if "filter_sort_seconds" in timings or "snapshot_load_seconds" in timings:
            st.caption(
                (
                    f"Fast scan runtime: {meta.get('duration_seconds', 0):.1f}s | "
                    f"Load snapshot: {timings.get('snapshot_load_seconds', 0):.1f}s | "
                    f"Filter/sort: {timings.get('filter_sort_seconds', timings.get('filter_display_seconds', 0)):.1f}s | "
                    f"Render prep: {timings.get('render_prep_seconds', 0):.1f}s"
                )
            )
        else:
            st.caption(
                (
                    f"Runtime: {meta.get('duration_seconds', 0):.1f}s | "
                    f"Universe load: {timings.get('universe_load_seconds', 0):.1f}s | "
                    f"Data fetch: {timings.get('data_fetch_seconds', 0):.1f}s | "
                    f"Snapshot prep: {timings.get('prefilter_seconds', 0):.1f}s | "
                    f"Trend prep: {timings.get('scoring_seconds', 0):.1f}s | "
                    f"Filtering/display prep: {timings.get('filter_display_seconds', 0):.1f}s"
                )
            )
    warning = meta.get("warning")
    if warning:
        st.warning(warning)
    skip_reasons = meta.get("prefilter_skip_reasons", {})
    if skip_reasons:
        skip_summary = ", ".join(f"{reason}: {count}" for reason, count in sorted(skip_reasons.items()))
        st.caption(f"Pre-score skips: {skip_summary}")


def render_price_filter_note(scanner_filters: dict | None) -> None:
    filters = scanner_filters or {}
    active_filters = []
    if filters.get("use_price_filter"):
        min_label = format_price(filters.get("min_price")) if filters.get("min_price") is not None else "No minimum"
        max_label = format_price(filters.get("max_price")) if filters.get("max_price") is not None else "No maximum"
        active_filters.append(f"Price: {min_label} to {max_label}")
    if filters.get("use_volume_filter"):
        active_filters.append(f"Avg volume >= {format_optional_number(filters.get('min_avg_volume'), decimals=0)}")
    if filters.get("use_relative_volume_filter"):
        active_filters.append(f"Rel volume >= {format_optional_number(filters.get('min_relative_volume'), decimals=2)}")
    if filters.get("above_ema21"):
        active_filters.append("Above EMA21")
    if filters.get("above_sma50"):
        active_filters.append("Above SMA50")
    if filters.get("above_sma200"):
        active_filters.append("Above SMA200")
    if active_filters:
        st.caption(f"Filters applied: {' | '.join(active_filters)}")
    else:
        st.caption("Filters applied: none")


def render_scan_not_run(message: str = "Choose filters, then run a scan.") -> None:
    st.info(message)


def render_scanner_description(universe_name: str) -> None:
    st.subheader(f"{get_universe_label(universe_name)} Scanner")
    st.info(
        "This scan looks for long-only swing setups with strong chart structure. It prioritizes stocks trading "
        "above key daily moving averages - EMA8, EMA21, SMA50, and SMA200 - with strong trend alignment and "
        "relative volume. The goal is to surface stocks already showing institutional-style strength, clean "
        "momentum, and potential follow-through."
    )
    st.caption(
        "Moving averages are calculated on daily candles. This is a technical scan only and does not replace chart review."
    )


def render_market_regime_status(scan_results: dict | None) -> None:
    st.markdown("### Market Regime Status")
    meta = (scan_results or {}).get("meta", {})
    regime = meta.get("market_regime") or {}
    if not regime:
        st.info("Market Regime will appear after refreshing the scanner snapshot.")
        return
    st.metric("Market Regime", regime.get("market_regime", "N/A"))
    st.caption(
        "Market Regime checks whether the broad market supports long trades. It uses SPY, QQQ, and IWM compared against EMA21, SMA50, and SMA200. Long breakouts usually work better when the overall market is Risk-On."
    )
    st.caption(regime.get("summary", ""))


def render_page(universe_name: str, scan_results: dict) -> None:
    render_scanner_description(universe_name)
    render_ranked_table("Main Trend Scanner", scan_results.get("all_ranked", []), key_prefix="ranked")


def render_main_trend_scanner(scan_results: dict, scanner_filters: dict | None = None) -> None:
    setup_view_labels = [
        "All Trend Setups",
        "Best A+ Setups",
        "Fresh Breakouts",
        "Breakout Retests",
        "Pre-Breakout Bases",
    ]
    setup_view_to_mode = {
        "All Trend Setups": "Main Trend Scanner",
        "Best A+ Setups": "A+ Setup Mode",
        "Fresh Breakouts": "Fresh Breakouts",
        "Breakout Retests": "Breakout Retests",
        "Pre-Breakout Bases": "Pre-Breakout Watchlist",
    }
    selected_setup_view = st.selectbox(
        "Trend setup view",
        setup_view_labels,
        index=0,
        key="main-trend-results-view",
        help="Filters the loaded trend-scan results by setup type.",
    )
    setup_mode = setup_view_to_mode[selected_setup_view]
    display_limit = int((scanner_filters or {}).get("display_limit") or 10)
    display_rows = filter_setup_mode(scan_results.get("all_ranked", []), setup_mode)[:display_limit]

    st.markdown("### Results")
    render_market_regime_status(scan_results)
    render_score_summary(scan_results)
    render_signal_score_methodology()
    render_column_guide()
    render_trend_status_guide()
    render_setup_type_guide()
    render_historical_edge_explanation()
    render_price_filter_note(scanner_filters)
    render_setup_quality_table(selected_setup_view, display_rows, key_prefix="ranked")


def render_score_summary(scan_results: dict | None) -> None:
    meta = (scan_results or {}).get("meta", {})
    st.info(
        "Signal Score is out of 100 points: Market Regime /15 + Trend Quality /15 + "
        "Base Quality /20 + Compression /15 + Trigger /15 + Risk/Reward /10 + "
        "Historical Edge /10, minus penalties."
    )
    if not bool(meta.get("calculate_historical_edge", False)):
        st.caption(
            "Historical Edge is currently not calculated. Scores may exclude the /10 Historical Edge component "
            "or show it as N/A depending on your settings."
        )


def render_column_guide() -> None:
    with st.expander("Column Guide", expanded=False):
        st.markdown(
            """
Setup Type: The current technical setup classification, such as Fresh Breakout, Breakout Retest, Pre-Breakout Base, EMA Reclaim, Stage-2 Leader, or Compression Squeeze.

Trend Status: A plain-English label that summarizes the stock's trend structure using price versus EMA8, EMA21, SMA50, SMA200, moving-average alignment, and moving-average slope.

Signal Score /100: Total setup score from 0 to 100. Higher means stronger overall swing setup quality.

Market Regime: Broad market condition based on SPY, QQQ, and IWM.

Trend Quality /15: Scores whether the stock is above key daily moving averages and whether the moving averages are aligned.

Base Quality /20: Scores whether the stock has a clean base, controlled pullbacks, higher lows, and a clear structure.

Compression /15: Scores whether volatility, range, and volume are tightening before a possible move.

Trigger /15: Scores whether the stock has a real breakout, reclaim, retest, or momentum trigger.

Risk/Reward /10: Scores whether the entry is efficient compared with the suggested stop and possible upside.

Historical Edge /10: Scores whether similar historical setups performed well in the stock's past data. Shows N/A if Historical Edge is turned off or there is not enough data.

Historical Hit Rate: Percentage of similar historical events that hit +5% before -3% within 30 trading days.

Historical Sample Size: Number of historical events used. Higher sample sizes are more reliable.

Suggested Pivot: Estimated resistance/breakout level based on recent highs.

Suggested Stop: Estimated invalidation area based on pivot, EMA21, recent swing low, or ATR. This is an estimate, not financial advice.

Risk %: Estimated downside from current price to suggested stop.

Reward/Risk: Estimated reward compared with estimated risk. Higher is better.

Relative Volume: Current volume divided by average 20-day volume.

ATR %: ATR14 divided by close price. Higher ATR% means the stock is more volatile.

Close Strength: Measures where the stock closed inside its daily range. A close near the high is stronger.
            """
        )


def render_trend_status_guide() -> None:
    with st.expander("Trend Status Guide", expanded=False):
        st.markdown(
            """
Strong Uptrend: Price is above EMA8, EMA21, SMA50, and SMA200 with moving averages aligned and rising. These are the cleanest trend structures.

Constructive Uptrend: Price is above major moving averages, but the trend may not be as perfectly aligned as Strong Uptrend.

Watchlist: The stock is improving or setting up, but still needs confirmation. This can include stocks reclaiming key moving averages or building near resistance.

Neutral / Mixed: Trend is unclear. Price may be between moving averages or the moving averages may be mixed.

Weak / Avoid: Price action is weak or below key short/intermediate moving averages. Usually not ideal for long swing setups.

Downtrend / Avoid: Price is below SMA200 or the long-term trend is falling. Avoid long setups unless you are specifically looking for reversal trades.

Insufficient Data: Not enough price history or moving-average data to classify the trend.

Setup Type answers what kind of setup this is. Trend Status answers how healthy the stock's overall trend is.
            """
        )


def render_setup_type_guide() -> None:
    with st.expander("Setup Type Guide", expanded=False):
        st.markdown(
            """
Pre-Breakout Base: Stock is near resistance, has a base of roughly 15+ trading days, volatility is contracting, and price is not extended.

Fresh Breakout: Close breaks above a recent 20-day or 50-day high, preferably with strong relative volume and a strong close location.

Breakout Retest: Stock recently broke above a pivot, then pulled back near that breakout level while holding trend support.

EMA Reclaim: Price closed back above EMA8, EMA21, SMA50, or SMA200 after previously closing below that moving average.

Stage-2 Leader: Price is above SMA50/SMA200, SMA50 is above SMA200, SMA200 is rising, and the stock is near 52-week highs.

Compression Squeeze: ATR%, recent range, and volume are tightening while price is near resistance.

No Clean Setup: The current structure does not match the clean long swing setup rules.
            """
        )


def render_historical_edge_explanation() -> None:
    with st.expander("Historical Edge Explanation", expanded=False):
        st.write(
            "Historical Hit Rate shows how often similar past events reached +5% before falling -3% within the next 30 trading days. This is not a guarantee. It is only a historical tendency based on available data."
        )
        st.caption("Historical Edge not calculated. Turn on Historical Edge to estimate past setup performance.")
        st.caption("If sample size is below 10, the app hides the hit rate and marks the evidence as low confidence.")


def render_setup_quality_table(title: str, rows: list[dict], key_prefix: str) -> None:
    st.markdown(f"### {title}")
    if not rows:
        st.info(f"No entries for {title.lower()}.")
        return

    display_rows = []
    for row in rows:
        sample_size = safe_numeric(row.get("historical_sample_size"))
        hit_rate = safe_numeric(row.get("historical_hit_rate"))
        display_rows.append(
            {
                "Ticker": row.get("ticker", "N/A"),
                "Company": row.get("company") or row.get("company_name") or "N/A",
                "Exchange": row.get("exchange") or "N/A",
                "Setup Type": row.get("setup_type", "N/A"),
                "Trend Status": row.get("trend_status", "Insufficient Data"),
                "A+ Status": row.get("a_plus_status", "N/A"),
                "Signal Score /100": format_score(row.get("setup_signal_score", row.get("signal_score")), 100),
                "Market Regime": row.get("market_regime", "N/A"),
                "Trend Quality /15": format_score(row.get("trend_quality"), 15),
                "Base Quality /20": format_score(row.get("base_quality_new"), 20),
                "Compression /15": format_score(row.get("compression_quality"), 15),
                "Trigger /15": format_score(row.get("trigger_quality_new"), 15),
                "Risk/Reward /10": format_score(row.get("risk_reward_quality"), 10),
                "Historical Edge /10": format_historical_edge_score(row),
                "Historical Hit Rate": f"{hit_rate:.0f}%" if hit_rate is not None and sample_size is not None and sample_size >= 10 else "N/A",
                "Historical Sample Size": int(sample_size) if sample_size is not None else "N/A",
                "Suggested Pivot": format_price(row.get("suggested_pivot")),
                "Suggested Stop": format_price(row.get("suggested_stop")),
                "Risk %": format_signed_percent(row.get("risk_pct")),
                "Reward/Risk": format_ratio(row.get("reward_risk")),
                "Rel Vol": format_ratio(row.get("volume_ratio")),
                "ATR %": format_percent_points(row.get("atr_pct")),
                "Close Strength": format_ratio(row.get("close_strength")),
                "Setup Note": truncate_text(row.get("a_plus_explanation"), max_length=120),
            }
        )

    st.dataframe(display_rows, use_container_width=True, hide_index=True)
    st.caption("Select a ticker below to open the existing chart/detail view.")
    ticker_cols = st.columns(6)
    for index, row in enumerate(rows[:24]):
        ticker = row.get("ticker", "N/A")
        ticker_cols[index % 6].button(ticker, key=f"select_{key_prefix}_{index}_{ticker}", on_click=open_ticker, args=(ticker,))


def render_trigger_scan_controls() -> tuple[list[str], str, int, int, int, bool, int, bool, bool, bool]:
    st.markdown("### Trigger Scan")
    st.caption("Finds stocks that recently reclaimed key moving averages or printed a major moving-average cross.")

    event_options = [
        "Crossed above EMA8",
        "Crossed above EMA21",
        "Crossed above SMA50",
        "Crossed above SMA100",
        "Crossed above SMA200",
        "Golden Cross: SMA50 crossed above SMA200",
        "Death Cross: SMA50 crossed below SMA200",
    ]
    event_display_labels = {
        "Crossed above EMA8": "EMA8 reclaim",
        "Crossed above EMA21": "EMA21 reclaim",
        "Crossed above SMA50": "SMA50 reclaim",
        "Crossed above SMA100": "SMA100 reclaim",
        "Crossed above SMA200": "SMA200 reclaim",
        "Golden Cross: SMA50 crossed above SMA200": "Golden cross",
        "Death Cross: SMA50 crossed below SMA200": "Death cross",
    }
    age_band_by_label = {
        "0-1 weeks ago": (0, 5),
        "1-2 weeks ago": (6, 10),
        "3-4 weeks ago": (11, 20),
        "5-6 weeks ago": (21, 30),
        "7-8 weeks ago": (31, 40),
        "8-9 weeks ago": (41, 45),
        "Custom full lookback": (0, 45),
    }
    col1, col2, col3 = st.columns([2.0, 1.1, 1.4])
    with col1:
        selected_events = st.multiselect(
            "Triggers to find",
            event_options,
            default=[
                "Crossed above EMA21",
                "Crossed above SMA50",
                "Crossed above SMA200",
                "Golden Cross: SMA50 crossed above SMA200",
            ],
            key="trigger-event-types",
            format_func=lambda option: event_display_labels.get(option, option),
        )
    with col2:
        age_band_label = st.selectbox("When it happened", list(age_band_by_label), index=1, key="trigger-age-band")
        max_trigger_results = st.slider(
            "Max results",
            min_value=5,
            max_value=100,
            value=25,
            step=5,
            key="trigger-max-results",
        )
    with col3:
        only_current = st.checkbox(
            "Still above EMA21 and SMA50",
            value=True,
            key="trigger-only-current-above",
        )
        one_row_per_ticker = st.checkbox("One row per ticker", value=True, key="trigger-one-row-per-ticker")
        show_all_trigger_events = st.checkbox("Show duplicate triggers", value=False, key="trigger-show-all-events")
        run_trigger_scan = st.button("Run Trigger Scan", type="secondary", key="trigger-run-scan")

    with st.expander("Trigger Scan Guide", expanded=False):
        st.markdown(
            """
Trigger Scan searches for recent moving-average reclaim events. A reclaim means price closed back above a selected moving average after previously closing below it.

When it happened filters results by the actual trigger date. This avoids showing the same recent stock in every longer lookback window.

One row per ticker keeps the list cleaner by showing the freshest and highest-priority trigger for each stock. Turn on duplicate triggers only when you want to inspect every signal.
            """
        )

    effective_one_row_per_ticker = one_row_per_ticker and not show_all_trigger_events
    min_days_ago, max_days_ago = age_band_by_label[age_band_label]

    return (
        selected_events,
        age_band_label,
        min_days_ago,
        max_days_ago,
        only_current,
        max_trigger_results,
        effective_one_row_per_ticker,
        show_all_trigger_events,
        run_trigger_scan,
    )


def render_trigger_scan_results(rows: list[dict] | None, scanner_filters: dict | None = None, age_band_label: str = "") -> None:
    render_price_filter_note(scanner_filters)
    if age_band_label:
        st.caption(f"Showing triggers from {age_band_label}")
    if rows is None:
        st.info("Choose trigger filters, then run Trigger Scan.")
        return
    if not rows:
        st.info("No trigger scan results matched the selected filters.")
        return

    widths = [0.75, 1.25, 0.85, 1.7, 1.0, 0.75, 0.85, 0.85, 0.85, 0.85, 0.85, 0.75, 0.85]
    header_cols = st.columns(widths)
    headers = [
        "Ticker",
        "Company",
        "Price",
        "Trigger",
        "Trigger Date",
        "Days Ago",
        "vs EMA8",
        "vs EMA21",
        "vs SMA50",
        "vs SMA100",
        "vs SMA200",
        "Rel Vol",
        "Action",
    ]
    for column, header in zip(header_cols, headers):
        column.markdown(f"**{header}**")

    for index, row in enumerate(rows):
        ticker = row.get("ticker", "N/A")
        cols = st.columns(widths)
        with cols[0]:
            if st.button(ticker, key=f"trigger_select_{index}_{ticker}_{row.get('trigger', '')}"):
                st.session_state["selected_ticker"] = ticker
        cols[1].caption(row.get("company") or row.get("company_name") or "N/A")
        cols[2].write(format_price(row.get("price")))
        cols[3].write(row.get("trigger", "N/A"))
        cols[4].write(row.get("trigger_date", "N/A"))
        cols[5].write(row.get("days_ago", "N/A"))
        cols[6].write(format_signed_percent(row.get("distance_from_ema8")))
        cols[7].write(format_signed_percent(row.get("distance_from_ema21")))
        cols[8].write(format_signed_percent(row.get("distance_from_sma50")))
        cols[9].write(format_signed_percent(row.get("distance_from_sma100")))
        cols[10].write(format_signed_percent(row.get("distance_from_sma200")))
        cols[11].write(format_ratio(row.get("volume_ratio")))
        if cols[12].button("Open", key=f"trigger_open_{index}_{ticker}_{row.get('trigger', '')}"):
            st.session_state["selected_ticker"] = ticker
            st.rerun()


def render_breakout_scan_controls() -> tuple[str, str, float, int, float, bool, float, int, bool]:
    st.markdown("### Breakout Search")
    st.caption("Finds stocks breaking out, setting up near resistance, or holding a recent breakout.")

    mode_options = {
        "Near Breakout": {
            "internal": "Pre-Breakout Setup",
            "description": "Finds stocks close to resistance before the breakout.",
            "max_ema21_extension": 12.0,
            "use_relative_volume": False,
        },
        "Breaking Out Now": {
            "internal": "Fresh Breakout",
            "description": "Finds stocks that recently broke above a 20D or 50D high.",
            "max_ema21_extension": 15.0,
            "use_relative_volume": True,
        },
        "Holding Breakout": {
            "internal": "Follow-Through Breakout",
            "description": "Finds stocks that broke out recently and are still holding strength.",
            "max_ema21_extension": 12.0,
            "use_relative_volume": False,
        },
    }
    fresh_window_options = {
        "Today only": 0,
        "Last 3 trading days": 3,
        "Last 5 trading days": 5,
        "Last 10 trading days": 10,
    }
    col1, col2, col3 = st.columns([1.7, 1.1, 1.0])
    with col1:
        display_mode = st.selectbox(
            "Setup to find",
            list(mode_options),
            index=0,
            key="breakout-mode",
        )
        st.caption(mode_options[display_mode]["description"])
        mode = mode_options[display_mode]["internal"]
        mode_key = display_mode.lower().replace(" ", "-")
        default_max_ema21_extension = mode_options[display_mode]["max_ema21_extension"]
        default_use_relative_volume = mode_options[display_mode]["use_relative_volume"]
    with col2:
        resistance_lookback = st.selectbox(
            "Resistance level",
            ["20D high", "50D high", "Both"],
            index=2,
            key="breakout-resistance-lookback",
        )
    with col3:
        max_results = st.slider(
            "Max results",
            min_value=5,
            max_value=100,
            value=25,
            step=5,
            key="breakout-max-results",
        )

    with st.expander("Breakout Search Guide", expanded=False):
        st.markdown(
            """
Breakout Search looks for long-only swing candidates showing breakout-style strength.

Near Breakout finds stocks close to resistance before the breakout. Breaking Out Now finds stocks that recently cleared a 20-day or 50-day high. Holding Breakout finds stocks that already broke out and are still holding strength.

Resistance level controls whether the breakout is judged against the recent 20-day high, 50-day high, or either level.
            """
        )

    max_distance_below_pct = 5.0
    fresh_window_label = "Last 5 trading days"
    max_distance_above_ema21_pct = default_max_ema21_extension
    use_relative_volume_filter = default_use_relative_volume
    min_relative_volume = 1.2
    with st.expander("Advanced Breakout Search Filters", expanded=False):
        st.caption("Advanced settings are optional. The default settings are designed to keep the scan clean.")
        advanced_cols = st.columns(3)
        with advanced_cols[0]:
            if mode == "Pre-Breakout Setup":
                max_distance_below_pct = st.number_input(
                    "How close to breakout level (%)",
                    min_value=0.0,
                    max_value=15.0,
                    value=5.0,
                    step=0.5,
                    format="%.1f",
                    key="breakout-max-below",
                )
            if mode == "Fresh Breakout":
                fresh_window_label = st.selectbox(
                    "Breakout must have happened within",
                    list(fresh_window_options),
                    index=2,
                    key="breakout-fresh-window",
                )
        with advanced_cols[1]:
            max_distance_above_ema21_pct = st.number_input(
                "Avoid if too far above EMA21 (%)",
                min_value=3.0,
                max_value=30.0,
                value=default_max_ema21_extension,
                step=0.5,
                format="%.1f",
                key=f"breakout-max-ema21-extension-{mode_key}",
            )
        with advanced_cols[2]:
            use_relative_volume_filter = st.checkbox(
                "Use relative volume filter",
                value=default_use_relative_volume,
                key=f"breakout-use-relvol-{mode_key}",
            )
            min_relative_volume = st.number_input(
                "Minimum volume strength",
                min_value=0.0,
                value=1.2,
                step=0.1,
                format="%.2f",
                key=f"breakout-min-relvol-{mode_key}",
                disabled=not use_relative_volume_filter,
            )

    action_col, note_col = st.columns([1.0, 3.0])
    with action_col:
        run_breakout_scan = st.button("Search Breakouts", type="secondary", key="breakout-run-scan")
    with note_col:
        st.caption("Snapshot data is already downloaded. This scan only filters the saved snapshot and does not redownload market data.")

    return (
        mode,
        resistance_lookback,
        float(max_distance_below_pct),
        fresh_window_options[fresh_window_label],
        float(max_distance_above_ema21_pct),
        bool(use_relative_volume_filter),
        float(min_relative_volume),
        int(max_results),
        run_breakout_scan,
    )


def render_breakout_scan_results(rows: list[dict] | None, scanner_filters: dict | None = None) -> None:
    render_price_filter_note(scanner_filters)
    if rows is None:
        st.info("Choose breakout search filters, then run Breakout Search.")
        return
    if not rows:
        st.info("No breakout search results matched the selected filters. If this snapshot was built before Breakout Search was added, refresh the snapshot once to populate breakout fields.")
        return

    with st.expander("Column Guide", expanded=False):
        st.markdown(
            """
Column guide:

Rel Vol = Current volume compared to normal volume. 1.0 means normal volume, 2.0 means about double normal volume. Higher relative volume can confirm a stronger breakout.

ATR % = Average True Range as a percentage of price. It shows how volatile the stock is. Lower ATR % usually means a tighter, cleaner stock. Higher ATR % means bigger daily swings and more risk.

Trend = The stock's current moving-average structure, based on daily EMA/SMA alignment. Strong Uptrend means price is above key moving averages with bullish alignment.

Quality = Breakout quality score from 0 to 100. Higher scores mean the setup has better trend alignment, breakout strength, volume confirmation, and risk/reward efficiency.
            """
        )

    widths = [0.7, 1.15, 0.75, 1.15, 0.8, 0.85, 0.85, 0.8, 0.75, 0.75, 0.75, 0.7, 0.65, 0.95, 0.7, 0.7]
    headers = [
        "Ticker",
        "Company",
        "Price",
        "Breakout Mode Match",
        "Level",
        "Type",
        "Dist",
        "Days",
        "vs EMA21",
        "vs SMA50",
        "vs SMA200",
        "Rel Vol",
        "ATR %",
        "Trend",
        "Quality",
        "Action",
    ]
    header_cols = st.columns(widths)
    for column, header in zip(header_cols, headers):
        column.markdown(f"**{header}**")

    for index, row in enumerate(rows):
        ticker = row.get("ticker", "N/A")
        display_mode = {
            "Pre-Breakout Setup": "Near Breakout",
            "Fresh Breakout": "Breaking Out Now",
            "Follow-Through Breakout": "Holding Breakout",
        }.get(row.get("breakout_mode_match"), row.get("breakout_mode_match", "N/A"))
        cols = st.columns(widths)
        cols[0].button(ticker, key=f"breakout_select_{index}_{ticker}", on_click=open_ticker, args=(ticker,))
        cols[1].caption(row.get("company") or row.get("company_name") or "N/A")
        cols[2].write(format_price(row.get("price")))
        cols[3].write(display_mode)
        cols[4].write(format_price(row.get("breakout_level")))
        cols[5].write(row.get("breakout_type", "N/A"))
        cols[6].write(format_signed_percent(row.get("distance_to_breakout")))
        cols[7].write("N/A" if row.get("days_since_breakout") is None else int(row.get("days_since_breakout")))
        cols[8].write(format_signed_percent(row.get("distance_from_ema21")))
        cols[9].write(format_signed_percent(row.get("distance_from_sma50")))
        cols[10].write(format_signed_percent(row.get("distance_from_sma200")))
        cols[11].write(format_ratio(row.get("relative_volume")))
        cols[12].write("N/A" if safe_numeric(row.get("atr_pct")) is None else f"{safe_numeric(row.get('atr_pct')):.1f}%")
        cols[13].write(row.get("trend_status", "N/A"))
        cols[14].write(row.get("breakout_quality", "N/A"))
        cols[15].button("Open", key=f"breakout_open_{index}_{ticker}", on_click=open_ticker, args=(ticker,))


def render_ma_event_results(rows: list[dict]) -> None:
    render_ma_scanner_explanations()

    if not rows:
        st.info("No MA break/reclaim events matched the current filters.")
        return

    header_cols = st.columns([1.0, 1.5, 1.1, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 2.2])
    headers = [
        "Ticker",
        "Event",
        "Event Date",
        "Days Ago",
        "Close",
        "MA Value",
        "% From MA",
        "Vol Ratio",
        "Signal",
        "Label",
        "Tags",
    ]
    for column, header in zip(header_cols, headers):
        column.markdown(f"**{header}**")

    for index, row in enumerate(rows):
        cols = st.columns([1.0, 1.5, 1.1, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 2.2])
        ticker = row.get("ticker", "N/A")
        event = row.get("event", "")
        with cols[0]:
            if st.button(ticker, key=f"ma-select-{index}-{ticker}-{event}"):
                st.session_state["selected_ticker"] = ticker
                enable_chart_marker_for_event(event)
        cols[1].write(event)
        cols[2].write(row.get("event_date", "N/A"))
        cols[3].write(row.get("days_since_event", "N/A"))
        cols[4].write(format_price(row.get("close")))
        cols[5].write(format_price(row.get("ma_value")))
        cols[6].write(format_signed_percent(row.get("pct_from_ma")))
        cols[7].write(format_ratio(row.get("volume_ratio")))
        cols[8].write(f"{row.get('signal_score', 0):.1f}")
        cols[9].write(row.get("label", "Pass"))
        cols[10].caption(format_tags(row.get("tags", [])))


def enable_chart_marker_for_event(event: str) -> None:
    st.session_state["detail-chart-show-ma-markers"] = True
    if "EMA8" in event:
        st.session_state["detail-chart-show-ema8"] = True
    elif "EMA21" in event:
        st.session_state["detail-chart-show-ema21"] = True
    elif "SMA50" in event:
        st.session_state["detail-chart-show-sma50"] = True
    elif "SMA200" in event:
        st.session_state["detail-chart-show-sma200"] = True


def render_section(title: str, rows: list[dict], key_prefix: str) -> None:
    st.markdown(f"### {title}")

    if not rows:
        st.info(f"No entries for {title.lower()}.")
        return

    for index, row in enumerate(rows):
        render_result_row(row, key_prefix=f"{key_prefix}-{index}")


def render_top_setups(rows: list[dict]) -> None:
    st.markdown("### Strong Long Setups")
    st.caption("Stocks above EMA21, SMA50, and SMA200 with rising EMA21/SMA50 structure and controlled extension.")

    if not rows:
        st.info("No Strong Long Setups found. Check Long Trend Results for watchlist or extended names.")
        return

    for row in rows:
        render_result_row(row, key_prefix=f"top_{row.get('ticker', 'N/A')}")


def render_result_row(row: dict, key_prefix: str, allow_save: bool = True) -> None:
    context = get_long_trend_context(row)
    summary_col, detail_col, action_col = st.columns([1, 7, 1.5])

    with summary_col:
        if st.button(row.get("ticker", "N/A"), key=f"select-{key_prefix}-{row.get('ticker', 'N/A')}"):
            st.session_state["selected_ticker"] = row.get("ticker")

    with detail_col:
        st.write(
            f"Price: {format_price(row.get('price'))} | "
            f"Change: {format_change_pct(row.get('change_pct'))} | "
            f"Long Status: {context.get('long_status')} | "
            f"EMA Stack: {context.get('ema_stack_status')}"
        )
        st.caption(
            f"EMA8 {context.get('ema8_status')} ({format_signed_percent(context.get('distance_ema8'))}) | "
            f"EMA21 {context.get('ema21_status')} ({format_signed_percent(context.get('distance_ema21'))}) | "
            f"vs SMA50 {format_signed_percent(context.get('distance_sma50'))} | "
            f"vs SMA200 {format_signed_percent(context.get('distance_sma200'))} | "
            f"Extension {context.get('extension_status')} | "
            f"Pullback {context.get('pullback_quality')} | "
            f"EMA21 Trend {context.get('ema21_trend')} | "
            f"SMA50 Trend {context.get('sma50_trend')} | "
            f"Vol Ratio {format_ratio(row.get('volume_ratio'))}"
        )

    with action_col:
        ticker = row.get("ticker", "N/A")
        if allow_save:
            already_saved = is_setup_saved(ticker)
            button_label = "Saved" if already_saved else "Save Setup"
            if st.button(button_label, key=f"save_setup_{key_prefix}_{ticker}", disabled=already_saved):
                if save_setup(row):
                    st.toast(f"{ticker} saved to Saved Setups")
                    st.rerun()
            if already_saved:
                st.caption("Already saved")


def render_ranked_table(title: str, rows: list[dict], key_prefix: str) -> None:
    st.markdown(f"### {title}")

    if not rows:
        st.info(f"No entries for {title.lower()}.")
        return

    widths = [0.75, 1.25, 0.9, 1.2, 0.85, 0.85, 0.85, 0.85, 0.9, 0.85]
    header_cols = st.columns(widths)
    headers = [
        "Ticker",
        "Company",
        "Price",
        "Trend Status",
        "vs EMA8",
        "vs EMA21",
        "vs SMA50",
        "vs SMA200",
        "Rel Vol",
        "Action",
    ]
    for column, header in zip(header_cols, headers):
        column.markdown(f"**{header}**")

    for row in rows:
        ticker = row.get("ticker", "N/A")
        context = get_long_trend_context(row)
        cols = st.columns(widths)
        with cols[0]:
            st.button(
                ticker,
                key=f"select_{key_prefix}_{ticker}",
                on_click=open_ticker,
                args=(ticker,),
            )
        cols[1].caption(row.get("company") or row.get("company_name") or "N/A")
        cols[2].write(format_price(row.get("price")))
        cols[3].write(classify_simple_trend_status(context))
        cols[4].write(format_signed_percent(context.get("distance_ema8")))
        cols[5].write(format_signed_percent(context.get("distance_ema21")))
        cols[6].write(format_signed_percent(context.get("distance_sma50")))
        cols[7].write(format_signed_percent(context.get("distance_sma200")))
        cols[8].write(format_ratio(row.get("volume_ratio")))
        cols[9].button(
            "Open",
            key=f"open_scanner_{key_prefix}_{ticker}",
            on_click=open_ticker,
            args=(ticker,),
        )


def render_saved_setups(scan_results: dict | None = None) -> None:
    st.markdown("### Saved Setups")
    st.caption("Manually saved trade ideas you want to track after a scan. Scanner results do not automatically appear here.")

    saved_setups = get_saved_setups()
    if not saved_setups:
        st.info("No saved setups yet. Run a scan and click Save Setup on names you want to track.")
        return

    header_cols = st.columns([1.0, 1.2, 1.0, 1.0, 1.8, 1.0, 1.0, 2.4, 2.0, 1.0])
    headers = [
        "Ticker",
        "Date Added",
        "Price at Add",
        "Current Price",
        "Long Status",
        "vs EMA21",
        "Rel Vol",
        "Tags",
        "Notes",
        "Remove",
    ]
    for column, header in zip(header_cols, headers):
        column.markdown(f"**{header}**")

    for index, setup in enumerate(list(saved_setups)):
        ticker = setup.get("ticker", "N/A")
        current_row = find_scan_row(ticker, scan_results)
        current_price = current_row.get("price") if current_row else None
        current_context = get_long_trend_context(current_row) if current_row else {}
        row_tags = format_tags(current_row.get("tags", [])) if current_row else setup.get("tags", "N/A")
        row_status = current_context.get("long_status") if current_row else setup.get("status", "N/A")

        cols = st.columns([1.0, 1.2, 1.0, 1.0, 1.8, 1.0, 1.0, 2.4, 2.0, 1.0])
        with cols[0]:
            if st.button(ticker, key=f"saved-select-{index}-{ticker}"):
                st.session_state["selected_ticker"] = ticker
        cols[1].write(setup.get("date_added", "N/A"))
        cols[2].write(format_price(setup.get("price_at_add")))
        cols[3].write(format_price(current_price))
        cols[4].write(row_status or "N/A")
        cols[5].write(format_signed_percent(current_context.get("distance_ema21")))
        cols[6].write(format_ratio(current_row.get("volume_ratio")) if current_row else "N/A")
        cols[7].caption(row_tags or "N/A")
        note_key = f"saved-note-{ticker}"
        setup["notes"] = cols[8].text_input(
            "Notes",
            value=setup.get("notes", ""),
            key=note_key,
            label_visibility="collapsed",
            placeholder="Add note",
        )
        if cols[9].button("Remove", key=f"remove-saved-{index}-{ticker}"):
            remove_saved_setup(ticker)
            st.rerun()


def get_latest_rsi(chart_history: object, length: int = 14) -> float | None:
    if chart_history is None or getattr(chart_history, "empty", True) or "Close" not in chart_history.columns:
        return None
    closes = chart_history["Close"].dropna()
    if len(closes.index) < length + 1:
        return None
    changes = closes.diff()
    gains = changes.clip(lower=0)
    losses = changes.clip(upper=0).abs()
    average_gain = gains.rolling(length, min_periods=length).mean()
    average_loss = losses.rolling(length, min_periods=length).mean()
    rs = average_gain / average_loss.mask(average_loss == 0)
    rsi = 100 - (100 / (1 + rs))
    latest = rsi.where(average_loss != 0, 100).dropna()
    return float(latest.iloc[-1]) if not latest.empty else None


def format_distance_from_ma(price: object, ma_value: object) -> str:
    price_value = safe_numeric(price)
    ma_numeric = safe_numeric(ma_value)
    if price_value is None or ma_numeric in (None, 0):
        return "N/A"
    return f"{((price_value / ma_numeric) - 1) * 100:+.1f}%"


def normalize_dashboard_value(value: object) -> str:
    if value is None:
        return "-"
    text_value = str(value).strip()
    if not text_value or text_value.upper() in {"N/A", "NONE", "NAN"}:
        return "-"
    return text_value


def get_dashboard_value_tone(label: str, value: object) -> str:
    text_value = normalize_dashboard_value(value)
    numeric_value = safe_numeric(text_value.replace("%", "").replace("$", "").replace(",", "").rstrip("x"))
    if text_value == "-":
        return "muted"
    if label in {"Short Change", "Penalty"} and numeric_value not in (None, 0):
        return "negative"
    if label in {"Day", "Distance", "Distance to Pivot", "% From High", "% Above Low"}:
        if text_value.startswith("+"):
            return "positive"
        if text_value.startswith("-"):
            return "negative"
    if text_value.startswith("+"):
        return "positive"
    if text_value.startswith("-"):
        return "negative"
    return "neutral"


def render_stock_detail_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1520px;
            padding-top: 3.25rem;
            padding-left: 1.15rem;
            padding-right: 1.15rem;
        }
        header[data-testid="stHeader"] {
            background: rgba(244, 247, 251, 0.96);
            backdrop-filter: blur(6px);
        }
        .stApp {
            background: #f4f7fb;
            color: #0f172a;
        }
        section[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #e5e7eb;
        }
        div[data-testid="stMarkdownContainer"],
        div[data-testid="stText"],
        label,
        p {
            color: #0f172a;
        }
        div[data-testid="stCaptionContainer"],
        small {
            color: #64748b;
        }
        div.stButton > button {
            background: #ffffff;
            color: #0f172a;
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        div.stButton > button:hover {
            border-color: #2563eb;
            color: #1d4ed8;
            background: #f8fafc;
        }
        div.stButton > button:disabled {
            color: #94a3b8;
            background: #f1f5f9;
            border-color: #e5e7eb;
        }
        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            border: 1px solid #dbe3ee;
            border-radius: 8px;
            padding: 0.72rem 0.82rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        div[data-testid="stMetricLabel"] {
            color: #64748b;
            font-size: 0.74rem;
            font-weight: 760;
        }
        div[data-testid="stMetricValue"] {
            color: #0f172a;
            font-size: 1.04rem;
            font-weight: 850;
        }
        div[data-testid="stTabs"] button {
            border-radius: 8px 8px 0 0;
            color: #475569;
            font-size: 0.9rem;
            font-weight: 680;
            padding: 0.5rem 0.84rem;
        }
        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: #0f172a;
            background: #ffffff;
            border-bottom-color: #2563eb;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid #dbe3ee;
            border-radius: 8px;
            overflow: hidden;
            background: #ffffff;
        }
        div[data-testid="stDataFrame"] [role="grid"] {
            font-size: 0.86rem;
        }
        div[data-testid="stExpander"] {
            border: 1px solid #dbe3ee;
            border-radius: 8px;
            background: #ffffff;
            overflow: hidden;
        }
        div[data-testid="stExpander"] details summary {
            font-weight: 760;
        }
        .detail-section {
            margin: 1rem 0 0.45rem;
        }
        .detail-section-title {
            color: #111827;
            font-size: 1rem;
            font-weight: 760;
            line-height: 1.2;
        }
        .detail-section-subtitle {
            color: #64748b;
            font-size: 0.86rem;
            line-height: 1.35;
            margin-top: 0.14rem;
        }
        .technical-readout {
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            background: #ffffff;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
            padding: 0.78rem 0.92rem;
            margin: 0.35rem 0 0.9rem;
        }
        .technical-readout-title {
            color: #111827;
            font-size: 1rem;
            font-weight: 760;
            line-height: 1.2;
            margin-bottom: 0.32rem;
        }
        .technical-readout-status {
            color: #334155;
            font-size: 0.9rem;
            font-weight: 680;
            margin-bottom: 0.38rem;
        }
        .technical-readout ul {
            margin: 0.25rem 0 0 1.15rem;
            padding: 0;
        }
        .technical-readout li {
            color: #1f2937;
            font-size: 0.92rem;
            line-height: 1.38;
            margin: 0.18rem 0;
        }
        .detail-stat-card,
        .trader-summary {
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            background: #ffffff;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
        }
        .detail-stat-card {
            min-height: 6rem;
            padding: 0.78rem 0.86rem;
            margin-bottom: 0.55rem;
        }
        .detail-stat-label {
            color: #64748b;
            font-size: 0.78rem;
            font-weight: 650;
            line-height: 1.15;
        }
        .detail-stat-value {
            color: #111827;
            font-size: 1.34rem;
            font-weight: 790;
            line-height: 1.14;
            margin-top: 0.34rem;
            overflow-wrap: anywhere;
        }
        .detail-stat-context {
            color: #64748b;
            font-size: 0.78rem;
            line-height: 1.25;
            margin-top: 0.34rem;
        }
        .trader-summary {
            color: #1f2937;
            font-size: 0.96rem;
            line-height: 1.48;
            padding: 0.82rem 0.95rem;
            margin: 0.45rem 0 0.85rem;
        }
        .detail-value-positive,
        .detail-value-neutral {
            color: #111827;
        }
        .detail-value-negative {
            color: #dc2626;
        }
        .detail-value-warning {
            color: #b45309;
        }
        .detail-value-muted {
            color: #94a3b8;
        }
        .fs-header,
        .fs-box {
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            background: #ffffff;
            margin: 0.52rem 0 0.78rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        .fs-header {
            display: grid;
            grid-template-columns: minmax(0, 1fr) repeat(5, auto);
            align-items: center;
            gap: 0.72rem;
            padding: 0.62rem 0.74rem;
        }
        .fs-title {
            color: #111827;
            font-size: 1.16rem;
            font-weight: 780;
            line-height: 1.18;
            overflow-wrap: anywhere;
        }
        .fs-subtitle {
            color: #64748b;
            font-size: 0.82rem;
            margin-top: 0.2rem;
        }
        .fs-chip {
            border: 1px solid #e5e7eb;
            border-radius: 4px;
            background: #f8fafc;
            padding: 0.34rem 0.5rem;
            min-width: 5.35rem;
            text-align: right;
        }
        .fs-chip-label {
            color: #64748b;
            font-size: 0.73rem;
            line-height: 1.1;
        }
        .fs-chip-value {
            color: #111827;
            font-size: 0.98rem;
            font-weight: 780;
            line-height: 1.18;
            margin-top: 0.11rem;
            white-space: nowrap;
        }
        .fs-box-title {
            color: #111827;
            font-size: 0.92rem;
            font-weight: 780;
            letter-spacing: 0;
            padding: 0.46rem 0.62rem;
            border-bottom: 1px solid #e5e7eb;
            background: #f1f5f9;
        }
        .fs-grid {
            display: grid;
            grid-template-columns: repeat(var(--fs-cols), minmax(0, 1fr));
        }
        .fs-cell {
            display: grid;
            grid-template-columns: minmax(5.3rem, 0.9fr) minmax(0, 1fr);
            min-height: 1.86rem;
            border-right: 1px solid #e5e7eb;
            border-bottom: 1px solid #e5e7eb;
        }
        .fs-label {
            color: #64748b;
            font-size: 0.82rem;
            line-height: 1.22;
            padding: 0.42rem 0.52rem;
            background: #f8fafc;
            overflow-wrap: anywhere;
        }
        .fs-value {
            color: #111827;
            font-size: 0.86rem;
            font-weight: 760;
            line-height: 1.22;
            padding: 0.42rem 0.52rem;
            text-align: right;
            overflow-wrap: anywhere;
        }
        .fs-positive {
            color: #047857;
        }
        .fs-negative {
            color: #dc2626;
        }
        .fs-warning {
            color: #b45309;
        }
        .fs-muted {
            color: #94a3b8;
        }
        .fund-quality {
            border: 1px solid #d1d5db;
            border-radius: 6px;
            background: #ffffff;
            padding: 0.7rem 0.82rem;
            margin: 0.55rem 0 0.75rem;
        }
        .fund-quality-title {
            color: #111827;
            font-size: 1rem;
            font-weight: 800;
            line-height: 1.2;
        }
        .fund-quality-note {
            color: #64748b;
            font-size: 0.82rem;
            line-height: 1.35;
            margin-top: 0.24rem;
        }
        .ma-summary-status {
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            background: #ffffff;
            padding: 0.58rem 0.72rem;
            margin: 0.35rem 0 0.55rem;
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            gap: 0.7rem;
            flex-wrap: wrap;
        }
        .ma-summary-status-label {
            color: #64748b;
            font-size: 0.8rem;
            font-weight: 680;
        }
        .ma-summary-status-value {
            color: #111827;
            font-size: 0.98rem;
            font-weight: 800;
        }
        .ma-card-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.55rem;
            margin-bottom: 0.7rem;
        }
        .ma-card {
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            background: #ffffff;
            overflow: hidden;
        }
        .ma-card-title {
            color: #111827;
            font-size: 0.88rem;
            font-weight: 800;
            padding: 0.48rem 0.58rem;
            background: #f8fafc;
            border-bottom: 1px solid #e5e7eb;
        }
        .ma-card-row {
            display: flex;
            justify-content: space-between;
            gap: 0.45rem;
            padding: 0.4rem 0.58rem;
            border-bottom: 1px solid #f1f5f9;
        }
        .ma-card-row:last-child {
            border-bottom: 0;
        }
        .ma-card-label {
            color: #64748b;
            font-size: 0.78rem;
            line-height: 1.2;
        }
        .ma-card-value {
            color: #111827;
            font-size: 0.8rem;
            font-weight: 760;
            line-height: 1.2;
            text-align: right;
        }
        .detail-explanation {
            border-bottom: 1px solid #e5e7eb;
            color: #1f2937;
            font-size: 0.86rem;
            line-height: 1.34;
            padding: 0.48rem 0.62rem;
        }
        .thesis-brief {
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            background: #ffffff;
            margin: 0.7rem 0 1rem;
            overflow: hidden;
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.06);
        }
        .thesis-hero {
            display: grid;
            grid-template-columns: minmax(0, 1fr) 15rem;
            gap: 1rem;
            padding: 0.95rem 1rem;
            border-bottom: 1px solid #e5e7eb;
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        }
        .thesis-eyebrow {
            color: #64748b;
            font-size: 0.72rem;
            font-weight: 820;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 0.28rem;
        }
        .thesis-headline {
            color: #0f172a;
            font-size: 1.25rem;
            font-weight: 860;
            line-height: 1.2;
            letter-spacing: 0;
            overflow-wrap: anywhere;
        }
        .thesis-lede {
            color: #334155;
            font-size: 0.93rem;
            line-height: 1.5;
            margin-top: 0.52rem;
            max-width: 78rem;
        }
        .thesis-verdict {
            align-self: stretch;
            border: 1px solid #dbe3ee;
            border-radius: 8px;
            background: #ffffff;
            padding: 0.72rem 0.78rem;
        }
        .thesis-verdict-label {
            color: #64748b;
            font-size: 0.72rem;
            font-weight: 800;
            text-transform: uppercase;
        }
        .thesis-verdict-value {
            color: #0f172a;
            font-size: 1.08rem;
            font-weight: 860;
            line-height: 1.2;
            margin-top: 0.2rem;
        }
        .thesis-verdict-note {
            color: #475569;
            font-size: 0.8rem;
            line-height: 1.34;
            margin-top: 0.35rem;
        }
        .thesis-strip {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            border-bottom: 1px solid #e5e7eb;
            background: #ffffff;
        }
        .thesis-stat {
            padding: 0.62rem 0.78rem;
            border-right: 1px solid #e5e7eb;
            min-width: 0;
        }
        .thesis-stat:last-child {
            border-right: 0;
        }
        .thesis-stat-label {
            color: #64748b;
            font-size: 0.72rem;
            font-weight: 760;
            line-height: 1.15;
        }
        .thesis-stat-value {
            color: #0f172a;
            font-size: 0.92rem;
            font-weight: 820;
            line-height: 1.22;
            margin-top: 0.2rem;
            overflow-wrap: anywhere;
        }
        .thesis-section-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0;
        }
        .thesis-section {
            padding: 0.86rem 1rem;
            border-right: 1px solid #e5e7eb;
            border-bottom: 1px solid #e5e7eb;
            min-width: 0;
        }
        .thesis-section:nth-child(2n) {
            border-right: 0;
        }
        .thesis-section-full {
            grid-column: 1 / -1;
            border-right: 0;
        }
        .thesis-section-header {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.44rem;
        }
        .thesis-pill {
            border-radius: 4px;
            color: #ffffff;
            background: #64748b;
            font-size: 0.68rem;
            font-weight: 840;
            line-height: 1;
            padding: 0.28rem 0.42rem;
            text-transform: uppercase;
        }
        .thesis-pill-positive {
            background: #047857;
        }
        .thesis-pill-warning {
            background: #b45309;
        }
        .thesis-pill-negative {
            background: #dc2626;
        }
        .thesis-section-title {
            color: #0f172a;
            font-size: 0.98rem;
            font-weight: 830;
            line-height: 1.25;
        }
        .thesis-section-text {
            color: #1f2937;
            font-size: 0.9rem;
            line-height: 1.48;
        }
        .thesis-evidence-row {
            display: flex;
            gap: 0.4rem;
            flex-wrap: wrap;
            margin-top: 0.58rem;
        }
        .thesis-evidence-chip {
            border: 1px solid #dbe3ee;
            border-radius: 999px;
            background: #f8fafc;
            color: #475569;
            font-size: 0.76rem;
            font-weight: 720;
            line-height: 1.2;
            padding: 0.26rem 0.48rem;
        }
        .thesis-lists {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0;
            border-bottom: 1px solid #e5e7eb;
        }
        .thesis-list-panel {
            padding: 0.86rem 1rem;
            border-right: 1px solid #e5e7eb;
        }
        .thesis-list-panel:last-child {
            border-right: 0;
        }
        .thesis-list-title,
        .thesis-source-title {
            color: #475569;
            font-size: 0.74rem;
            font-weight: 840;
            letter-spacing: 0.03em;
            text-transform: uppercase;
            margin-bottom: 0.36rem;
        }
        .thesis-list {
            margin: 0;
            padding-left: 1.05rem;
        }
        .thesis-list li {
            color: #1f2937;
            font-size: 0.88rem;
            line-height: 1.45;
            margin: 0.18rem 0;
        }
        .thesis-source-footer {
            color: #64748b;
            font-size: 0.78rem;
            line-height: 1.38;
            padding: 0.62rem 1rem 0.78rem;
            background: #fbfdff;
        }
        .thesis-brief {
            border: 1px solid #d1d5db;
            border-radius: 6px;
            background: #ffffff;
            box-shadow: none;
        }
        .thesis-hero {
            display: block;
            padding: 1.05rem 1.05rem 0.95rem;
            border-bottom: 0;
            background: #ffffff;
        }
        .thesis-eyebrow {
            color: #475569;
            font-size: 0.78rem;
            font-weight: 820;
            letter-spacing: 0;
            text-transform: none;
            margin-bottom: 0.45rem;
        }
        .thesis-headline {
            font-size: 1.42rem;
            line-height: 1.18;
            max-width: 78rem;
        }
        .thesis-lede {
            color: #1f2937;
            font-size: 0.98rem;
            line-height: 1.58;
            max-width: 86rem;
            margin-top: 0.56rem;
        }
        .thesis-verdict {
            display: none;
        }
        .thesis-strip {
            display: flex;
            flex-wrap: wrap;
            gap: 0.42rem;
            border: 0;
            padding: 0 1.05rem 0.3rem;
        }
        .thesis-stat {
            border: 1px solid #dbe3ee;
            border-radius: 999px;
            background: #f8fafc;
            padding: 0.28rem 0.55rem;
        }
        .thesis-stat-label,
        .thesis-stat-value {
            display: inline;
            font-size: 0.78rem;
            line-height: 1.2;
        }
        .thesis-stat-label {
            margin-right: 0.18rem;
        }
        .thesis-stat-value {
            color: #334155;
            font-weight: 760;
        }
        .thesis-report-label {
            color: #475569;
            font-size: 0.78rem;
            font-weight: 820;
            letter-spacing: 0.02em;
            text-transform: uppercase;
            padding: 0.85rem 1.05rem 0.1rem;
        }
        .thesis-section-grid {
            display: block;
            padding: 0 1.05rem 0.6rem;
        }
        .thesis-section {
            border: 0;
            border-left: 4px solid #64748b;
            border-radius: 6px;
            background: #f8fafc;
            margin: 0.55rem 0;
            padding: 0.82rem 0.95rem;
        }
        .thesis-section-positive {
            border-left-color: #059669;
        }
        .thesis-section-warning {
            border-left-color: #d97706;
        }
        .thesis-section-negative {
            border-left-color: #dc2626;
        }
        .thesis-section:nth-child(2n) {
            border-right: 0;
        }
        .thesis-section-header {
            margin-bottom: 0.5rem;
        }
        .thesis-pill {
            font-size: 0.72rem;
            padding: 0.24rem 0.45rem;
        }
        .thesis-section-title {
            font-size: 1.02rem;
            line-height: 1.25;
        }
        .thesis-section-text {
            font-size: 0.92rem;
            line-height: 1.55;
        }
        .thesis-evidence-row {
            border-top: 1px solid #e5e7eb;
            padding-top: 0.5rem;
        }
        .thesis-risk-report {
            border-left: 4px solid #d97706;
            border-radius: 6px;
            background: #fffbeb;
            margin: 0.75rem 1.05rem 0.9rem;
            padding: 0.78rem 0.95rem;
        }
        .thesis-confirm-report {
            border-left: 4px solid #2563eb;
            border-radius: 6px;
            background: #eff6ff;
            margin: 0.75rem 1.05rem 0;
            padding: 0.78rem 0.95rem;
        }
        .thesis-report-list-title {
            color: #475569;
            font-size: 0.78rem;
            font-weight: 820;
            letter-spacing: 0.02em;
            text-transform: uppercase;
            margin-bottom: 0.32rem;
        }
        .thesis-report-list {
            margin: 0;
            padding-left: 1.1rem;
        }
        .thesis-report-list li {
            color: #111827;
            font-size: 0.9rem;
            line-height: 1.5;
            margin: 0.16rem 0;
        }
        .thesis-source-footer {
            border-top: 1px solid #e5e7eb;
            padding: 0.7rem 1.05rem 0.85rem;
        }
        .guidance-brief {
            border: 1px solid #cbd5e1;
            border-radius: 6px;
            background: #ffffff;
            margin: 0.7rem 0 1rem;
            overflow: hidden;
        }
        .guidance-hero {
            display: grid;
            grid-template-columns: minmax(0, 1fr) 13rem;
            gap: 1rem;
            padding: 1rem 1.05rem 0.95rem;
            border-bottom: 1px solid #e5e7eb;
        }
        .guidance-eyebrow {
            color: #475569;
            font-size: 0.76rem;
            font-weight: 820;
            letter-spacing: 0.02em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }
        .guidance-headline {
            color: #0f172a;
            font-size: 1.22rem;
            font-weight: 860;
            line-height: 1.2;
            overflow-wrap: anywhere;
        }
        .guidance-lede {
            color: #334155;
            font-size: 0.92rem;
            line-height: 1.5;
            margin-top: 0.5rem;
        }
        .guidance-scorebox {
            border: 1px solid #dbe3ee;
            border-radius: 6px;
            background: #f8fafc;
            padding: 0.7rem 0.76rem;
            align-self: start;
        }
        .guidance-score-label {
            color: #64748b;
            font-size: 0.72rem;
            font-weight: 820;
            text-transform: uppercase;
        }
        .guidance-score-value {
            color: #0f172a;
            font-size: 1.05rem;
            font-weight: 860;
            line-height: 1.2;
            margin-top: 0.22rem;
        }
        .guidance-score-note {
            color: #475569;
            font-size: 0.78rem;
            line-height: 1.34;
            margin-top: 0.34rem;
        }
        .guidance-strip {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            border-bottom: 1px solid #e5e7eb;
        }
        .guidance-stat {
            padding: 0.62rem 0.78rem;
            border-right: 1px solid #e5e7eb;
            min-width: 0;
        }
        .guidance-stat:last-child {
            border-right: 0;
        }
        .guidance-stat-label {
            color: #64748b;
            font-size: 0.72rem;
            font-weight: 780;
            line-height: 1.15;
        }
        .guidance-stat-value {
            color: #0f172a;
            font-size: 0.9rem;
            font-weight: 820;
            line-height: 1.22;
            margin-top: 0.2rem;
            overflow-wrap: anywhere;
        }
        .guidance-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0;
        }
        .guidance-panel {
            padding: 0.86rem 1.05rem;
            border-right: 1px solid #e5e7eb;
        }
        .guidance-panel:last-child {
            border-right: 0;
        }
        .guidance-panel-title {
            color: #475569;
            font-size: 0.78rem;
            font-weight: 840;
            letter-spacing: 0.02em;
            text-transform: uppercase;
            margin-bottom: 0.45rem;
        }
        .guidance-row {
            display: grid;
            grid-template-columns: minmax(6.8rem, 0.46fr) minmax(0, 1fr);
            gap: 0.72rem;
            padding: 0.46rem 0;
            border-bottom: 1px solid #f1f5f9;
        }
        .guidance-row:last-child {
            border-bottom: 0;
        }
        .guidance-row-label {
            color: #64748b;
            font-size: 0.82rem;
            font-weight: 780;
            line-height: 1.3;
        }
        .guidance-row-value {
            color: #1f2937;
            font-size: 0.88rem;
            line-height: 1.42;
        }
        .guidance-pill {
            display: inline-block;
            border-radius: 4px;
            color: #ffffff;
            background: #64748b;
            font-size: 0.7rem;
            font-weight: 840;
            line-height: 1;
            padding: 0.25rem 0.42rem;
            margin-right: 0.38rem;
            text-transform: uppercase;
            vertical-align: 0.08rem;
        }
        .guidance-pill-positive {
            background: #047857;
        }
        .guidance-pill-warning {
            background: #b45309;
        }
        .guidance-pill-negative {
            background: #dc2626;
        }
        .guidance-note {
            color: #64748b;
            font-size: 0.78rem;
            line-height: 1.38;
            padding: 0.62rem 1.05rem 0.78rem;
            border-top: 1px solid #e5e7eb;
            background: #fbfdff;
        }
        @media (max-width: 720px) {
            .fs-header {
                grid-template-columns: 1fr 1fr;
            }
            .fs-title-wrap {
                grid-column: 1 / -1;
            }
            .fs-grid {
                grid-template-columns: repeat(1, minmax(0, 1fr));
            }
            .fs-chip {
                min-width: 0;
                text-align: left;
            }
            .ma-card-grid {
                grid-template-columns: repeat(1, minmax(0, 1fr));
            }
            .thesis-hero,
            .thesis-section-grid,
            .thesis-lists {
                grid-template-columns: 1fr;
            }
            .thesis-strip {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .guidance-hero,
            .guidance-grid,
            .guidance-strip {
                grid-template-columns: 1fr;
            }
            .thesis-section,
            .thesis-list-panel,
            .guidance-panel,
            .guidance-stat {
                border-right: 0;
            }
            .guidance-row {
                grid-template-columns: 1fr;
                gap: 0.18rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard_stat_card(
    label: str,
    value: object,
    context: str = "",
    tone: str | None = None,
    large: bool = False,
) -> None:
    display_value = normalize_dashboard_value(value)
    value_tone = tone or get_dashboard_value_tone(label, display_value)
    size_class = " detail-stat-card-large" if large else ""
    context_html = f"<div class='detail-stat-context'>{escape(context)}</div>" if context else ""
    st.markdown(
        f"<div class='detail-stat-card{size_class}'>"
        f"<div class='detail-stat-label'>{escape(label)}</div>"
        f"<div class='detail-stat-value detail-value-{value_tone}'>{escape(display_value)}</div>"
        f"{context_html}"
        "</div>",
        unsafe_allow_html=True,
    )


def render_dashboard_stat_grid(items: list[dict], columns: int = 4) -> None:
    for index in range(0, len(items), columns):
        cols = st.columns(columns)
        for col, item in zip(cols, items[index:index + columns]):
            with col:
                render_dashboard_stat_card(
                    label=item.get("label", ""),
                    value=item.get("value"),
                    context=item.get("context", ""),
                    tone=item.get("tone"),
                    large=bool(item.get("large")),
                )


def render_dashboard_section_header(title: str, subtitle: str = "") -> None:
    subtitle_html = f"<div class='detail-section-subtitle'>{escape(subtitle)}</div>" if subtitle else ""
    st.markdown(
        "<div class='detail-section'>"
        f"<div class='detail-section-title'>{escape(title)}</div>"
        f"{subtitle_html}"
        "</div>",
        unsafe_allow_html=True,
    )


def render_key_value_card(title: str, values: list[tuple[str, object]], class_prefix: str = "detail-card") -> None:
    rows = []
    for label, raw_value in values:
        value = normalize_dashboard_value(raw_value)
        tone = get_dashboard_value_tone(label, raw_value)
        rows.append(
            f"<div class='{class_prefix}-row'>"
            f"<span class='{class_prefix}-label'>{escape(label)}</span>"
            f"<span class='{class_prefix}-value {class_prefix}-value-{tone} detail-value-{tone}'>{escape(value)}</span>"
            "</div>"
        )

    st.markdown(
        f"<div class='{class_prefix}'>"
        f"<div class='{class_prefix}-title'>{escape(title)}</div>"
        f"{''.join(rows)}"
        "</div>",
        unsafe_allow_html=True,
    )


def normalize_fundamental_value(value: object) -> str:
    if value is None:
        return "—"
    text_value = str(value).strip()
    if not text_value or text_value.upper() in {"N/A", "NONE", "NAN"}:
        return "—"
    return text_value


def get_fundamental_value_tone(label: str, value: object) -> str:
    text_value = normalize_fundamental_value(value)
    if text_value == "—":
        return "muted"
    if label == "Short Change":
        if text_value.startswith("+"):
            return "negative"
        if text_value.startswith("-"):
            return "positive"
    if text_value.startswith("+"):
        return "positive"
    if text_value.startswith("-"):
        return "negative"
    return "neutral"


def render_fundamental_card(title: str, values: list[tuple[str, object]]) -> None:
    render_key_value_card(title, values, class_prefix="fund-card")


def format_metric_value(value: object) -> str:
    if value is None:
        return "—"
    text_value = str(value).strip()
    if not text_value or text_value.upper() in {"N/A", "NONE", "NAN"}:
        return "—"
    return text_value


def get_compact_value_class(label: str, value: object) -> str:
    text_value = format_metric_value(value)
    numeric_value = safe_numeric(text_value.replace("%", "").replace("$", "").replace(",", "").rstrip("x"))
    if text_value == "—":
        return "fs-muted"
    if label in {"Penalty", "Short Change"} and numeric_value not in (None, 0):
        return "fs-warning" if label == "Penalty" else "fs-negative"
    if text_value.startswith("+"):
        return "fs-positive"
    if text_value.startswith("-"):
        return "fs-negative"
    return ""


def render_compact_metric_grid(title: str, items: list[tuple[str, object]], columns: int = 4) -> None:
    if not items:
        st.info(f"No {title.lower()} data available.")
        return

    cells = []
    for label, raw_value in items:
        value = format_metric_value(raw_value)
        value_class = get_compact_value_class(label, raw_value)
        cells.append(
            "<div class='fs-cell'>"
            f"<div class='fs-label'>{escape(str(label))}</div>"
            f"<div class='fs-value {value_class}'>{escape(value)}</div>"
            "</div>"
        )

    title_html = f"<div class='fs-box-title'>{escape(title)}</div>" if title else ""
    st.markdown(
        "<div class='fs-box'>"
        f"{title_html}"
        f"<div class='fs-grid' style='--fs-cols:{max(1, columns)}'>{''.join(cells)}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_factsheet(title: str, sections: dict[str, list[tuple[str, object]]] | list[tuple[str, list[tuple[str, object]]]], columns: int = 4) -> None:
    section_items = sections.items() if isinstance(sections, dict) else sections
    for section_title, items in section_items:
        heading = f"{title} - {section_title}" if title else section_title
        render_compact_metric_grid(heading, items, columns=columns)


def render_compact_detail_header(
    heading: str,
    subtitle: str,
    stats: list[tuple[str, object]],
    link_label: str = "",
    link_url: str = "",
) -> None:
    chips = []
    for label, raw_value in stats:
        value = format_metric_value(raw_value)
        value_class = get_compact_value_class(label, raw_value)
        chips.append(
            "<div class='fs-chip'>"
            f"<div class='fs-chip-label'>{escape(label)}</div>"
            f"<div class='fs-chip-value {value_class}'>{escape(value)}</div>"
            "</div>"
        )
    if link_url:
        chips.append(
            "<a class='fs-chip' href='"
            f"{escape(link_url)}"
            "' target='_blank' rel='noopener noreferrer'>"
            "<div class='fs-chip-label'>Link</div>"
            f"<div class='fs-chip-value'>{escape(link_label or 'Open')}</div>"
            "</a>"
        )

    st.markdown(
        "<div class='fs-header'>"
        "<div class='fs-title-wrap'>"
        f"<div class='fs-title'>{escape(heading)}</div>"
        f"<div class='fs-subtitle'>{escape(subtitle)}</div>"
        "</div>"
        f"{''.join(chips)}"
        "</div>",
        unsafe_allow_html=True,
    )


def get_sma_from_history(chart_history: object, length: int) -> float | None:
    if chart_history is None or getattr(chart_history, "empty", True) or "Close" not in chart_history.columns:
        return None
    closes = chart_history["Close"].dropna()
    if len(closes.index) < length:
        return None
    return float(closes.rolling(length).mean().iloc[-1])


def get_ema_from_history(chart_history: object, length: int) -> float | None:
    if chart_history is None or getattr(chart_history, "empty", True) or "Close" not in chart_history.columns:
        return None
    closes = chart_history["Close"].dropna()
    if len(closes.index) < length:
        return None
    return float(closes.ewm(span=length, adjust=False, min_periods=length).mean().iloc[-1])


def detect_history_ma_trend(chart_history: object, ma_type: str, length: int, lookback: int = 5) -> str:
    if chart_history is None or getattr(chart_history, "empty", True) or "Close" not in chart_history.columns:
        return "Insufficient Data"
    closes = chart_history["Close"].dropna()
    if len(closes.index) < length + lookback:
        return "Insufficient Data"
    if ma_type == "ema":
        series = closes.ewm(span=length, adjust=False, min_periods=length).mean().dropna()
    else:
        series = closes.rolling(length).mean().dropna()
    if len(series.index) <= lookback:
        return "Insufficient Data"
    latest = safe_numeric(series.iloc[-1])
    prior = safe_numeric(series.iloc[-lookback - 1])
    if latest is None or prior in (None, 0):
        return "Insufficient Data"
    change_pct = ((latest / prior) - 1) * 100
    if change_pct > 0.15:
        return "Rising"
    if change_pct < -0.15:
        return "Falling"
    return "Flat"


def calculate_ma_distance(price: object, ma_value: object) -> float | None:
    price_value = safe_numeric(price)
    ma_numeric = safe_numeric(ma_value)
    if price_value is None or ma_numeric in (None, 0):
        return None
    return ((price_value / ma_numeric) - 1) * 100


def format_yes_no(value: bool | None) -> str:
    if value is None:
        return "N/A"
    return "Yes" if value else "No"


def get_latest_series_value(series: object) -> float | None:
    if series is None:
        return None
    values = series.dropna()
    if values.empty:
        return None
    return safe_numeric(values.iloc[-1])


def get_previous_series_value(series: object) -> float | None:
    if series is None:
        return None
    values = series.dropna()
    if len(values.index) < 2:
        return None
    return safe_numeric(values.iloc[-2])


def get_close_series(chart_history: object):
    if chart_history is None or getattr(chart_history, "empty", True) or "Close" not in chart_history.columns:
        return None
    return chart_history["Close"].dropna()


def get_low_series(chart_history: object):
    if chart_history is None or getattr(chart_history, "empty", True) or "Low" not in chart_history.columns:
        return None
    return chart_history["Low"].dropna()


def build_ma_series(chart_history: object, ma_type: str, length: int):
    closes = get_close_series(chart_history)
    if closes is None or len(closes.index) < length:
        return None
    if ma_type == "ema":
        return closes.ewm(span=length, adjust=False, min_periods=length).mean()
    return closes.rolling(length).mean()


def classify_extension_status(distance_from_ema21: object) -> str:
    distance = safe_numeric(distance_from_ema21)
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


def classify_ema_stack_status(price: object, ema8: object, ema21: object, sma50: object, sma200: object, ema21_trend: str) -> str:
    price_value = safe_numeric(price)
    ema8_value = safe_numeric(ema8)
    ema21_value = safe_numeric(ema21)
    sma50_value = safe_numeric(sma50)
    sma200_value = safe_numeric(sma200)
    if price_value is None or ema8_value is None or ema21_value is None or sma50_value is None:
        return "Insufficient Data"
    if (price_value < sma50_value and (sma200_value is None or price_value < sma200_value)) or ema21_trend == "Falling":
        return "Weak / Avoid"
    if price_value < ema21_value or price_value < sma50_value:
        return "Below Key MAs"
    if sma200_value is not None and price_value > ema8_value > ema21_value > sma50_value > sma200_value:
        return "Perfect Bullish Stack"
    if price_value > ema8_value and price_value > ema21_value:
        return "Bullish but Early"
    return "Mixed Trend"


def classify_ma_position_status(
    latest_close: object,
    prior_close: object,
    latest_ma: object,
    prior_ma: object,
    label: str,
    extended_threshold_pct: float,
) -> str:
    latest_close_value = safe_numeric(latest_close)
    prior_close_value = safe_numeric(prior_close)
    latest_ma_value = safe_numeric(latest_ma)
    prior_ma_value = safe_numeric(prior_ma)
    if latest_close_value is None or latest_ma_value is None:
        return "N/A"
    distance = calculate_ma_distance(latest_close_value, latest_ma_value)
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
    close_value = safe_numeric(latest_close)
    low_value = safe_numeric(latest_low)
    ema8_value = safe_numeric(ema8)
    ema21_value = safe_numeric(ema21)
    sma50_value = safe_numeric(sma50)
    sma200_value = safe_numeric(sma200)
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
    ema21_distance = calculate_ma_distance(close_value, ema21_value)
    if ema21_distance is not None and 0 <= ema21_distance <= 3 and ema21_trend == "Rising":
        return "Holding EMA21"
    return "N/A"


def classify_long_status_from_context(context: dict, volume_ratio: object = None) -> str:
    price = safe_numeric(context.get("price"))
    ema8 = safe_numeric(context.get("ema8"))
    ema21 = safe_numeric(context.get("ema21"))
    sma50 = safe_numeric(context.get("sma50"))
    sma200 = safe_numeric(context.get("sma200"))
    if price is None or ema21 is None or sma50 is None:
        return "Insufficient Data"
    above_ema8 = ema8 is not None and price > ema8
    above_ema21 = price > ema21
    above_sma50 = price > sma50
    above_sma200 = sma200 is not None and price > sma200
    ema21_trend = context.get("ema21_trend")
    sma50_trend = context.get("sma50_trend")
    extension_status = context.get("extension_status")
    pullback_quality = context.get("pullback_quality")
    volume_value = safe_numeric(volume_ratio)
    volume_ok = volume_value is None or volume_value >= 0.8
    if not above_sma50 and (sma200 is None or not above_sma200) and ema21_trend == "Falling":
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
        and context.get("ema_stack_status") == "Perfect Bullish Stack"
        and ema21_trend == "Rising"
        and sma50_trend == "Rising"
        and extension_status not in {"Extended", "Too Extended"}
        and volume_ok
    ):
        return "Strong Long Setup"
    return "Long Watchlist"


def classify_simple_trend_status(context: dict) -> str:
    price = safe_numeric(context.get("price"))
    ema8 = safe_numeric(context.get("ema8"))
    ema21 = safe_numeric(context.get("ema21"))
    sma50 = safe_numeric(context.get("sma50"))
    sma200 = safe_numeric(context.get("sma200"))
    if price is None or ema21 is None or sma50 is None:
        return "Weak / Avoid"
    if price < ema21 or price < sma50:
        return "Weak / Avoid"
    if (
        ema8 is not None
        and sma200 is not None
        and price > ema8
        and price > ema21
        and price > sma50
        and price > sma200
        and ema8 > ema21
        and context.get("ema21_trend") == "Rising"
    ):
        return "Strong Uptrend"
    return "Watchlist"


def get_long_trend_context(detail_row: dict, chart_history: object = None) -> dict:
    metrics = detail_row.get("metrics", {}) or {}
    price = detail_row.get("price")
    ema8_series = build_ma_series(chart_history, "ema", 8)
    ema21_series = build_ma_series(chart_history, "ema", 21)
    sma50_series = build_ma_series(chart_history, "sma", 50)
    sma200_series = build_ma_series(chart_history, "sma", 200)
    close_series = get_close_series(chart_history)
    low_series = get_low_series(chart_history)
    ema8 = detail_row.get("ema8") or metrics.get("ema_8") or get_latest_series_value(ema8_series)
    ema21 = detail_row.get("ema21") or metrics.get("ema_21") or get_latest_series_value(ema21_series)
    sma50 = detail_row.get("sma50") or metrics.get("sma50") or get_latest_series_value(sma50_series)
    sma200 = detail_row.get("sma200") or metrics.get("sma200") or get_latest_series_value(sma200_series)
    ema8_trend = detail_row.get("ema8_trend") or detect_history_ma_trend(chart_history, "ema", 8)
    ema21_trend = detail_row.get("ema21_trend") or detect_history_ma_trend(chart_history, "ema", 21)
    sma50_trend = detail_row.get("sma50_trend") or detect_history_ma_trend(chart_history, "sma", 50)
    price_value = safe_numeric(price)
    ema21_value = safe_numeric(ema21)
    sma50_value = safe_numeric(sma50)
    sma200_value = safe_numeric(sma200)
    above_ema8 = price_value > safe_numeric(ema8) if price_value is not None and safe_numeric(ema8) is not None else None
    above_ema21 = price_value > ema21_value if price_value is not None and ema21_value is not None else None
    above_sma50 = price_value > sma50_value if price_value is not None and sma50_value is not None else None
    above_sma200 = price_value > sma200_value if price_value is not None and sma200_value is not None else None
    distance_ema21 = detail_row.get("distance_from_ema21")
    distance_sma50 = detail_row.get("distance_from_sma50")
    distance_sma200 = detail_row.get("distance_from_sma200")
    distance_ema8 = detail_row.get("distance_from_ema8")
    if distance_ema8 is None:
        distance_ema8 = calculate_ma_distance(price, ema8)
    if distance_ema21 is None:
        distance_ema21 = calculate_ma_distance(price, ema21)
    if distance_sma50 is None:
        distance_sma50 = calculate_ma_distance(price, sma50)
    if distance_sma200 is None:
        distance_sma200 = calculate_ma_distance(price, sma200)

    prior_close = get_previous_series_value(close_series)
    latest_low = get_latest_series_value(low_series)
    ema_stack_status = detail_row.get("ema_stack_status") or classify_ema_stack_status(price, ema8, ema21, sma50, sma200, ema21_trend)
    ema8_status = detail_row.get("ema8_status") or classify_ma_position_status(
        price,
        prior_close,
        ema8,
        get_previous_series_value(ema8_series),
        "EMA8",
        6,
    )
    ema21_status = detail_row.get("ema21_status") or classify_ma_position_status(
        price,
        prior_close,
        ema21,
        get_previous_series_value(ema21_series),
        "EMA21",
        12,
    )
    extension_status = detail_row.get("extension_status") or classify_extension_status(distance_ema21)
    pullback_quality = detail_row.get("pullback_quality") or classify_pullback_quality(
        price,
        latest_low,
        ema8,
        ema21,
        sma50,
        sma200,
        ema21_trend,
        extension_status,
    )

    status = detail_row.get("long_status")
    if not status:
        status = classify_long_status_from_context(
            {
                "price": price,
                "ema8": ema8,
                "ema21": ema21,
                "sma50": sma50,
                "sma200": sma200,
                "ema_stack_status": ema_stack_status,
                "extension_status": extension_status,
                "pullback_quality": pullback_quality,
                "ema21_trend": ema21_trend,
                "sma50_trend": sma50_trend,
            },
            volume_ratio=detail_row.get("volume_ratio"),
        )

    return {
        "price": price,
        "ema8": ema8,
        "ema21": ema21,
        "sma50": sma50,
        "sma200": sma200,
        "distance_ema8": distance_ema8,
        "distance_ema21": distance_ema21,
        "distance_sma50": distance_sma50,
        "distance_sma200": distance_sma200,
        "ema8_trend": ema8_trend,
        "ema21_trend": ema21_trend,
        "sma50_trend": sma50_trend,
        "ema_stack_status": ema_stack_status,
        "ema8_status": ema8_status,
        "ema21_status": ema21_status,
        "extension_status": extension_status,
        "pullback_quality": pullback_quality,
        "above_ema8": above_ema8,
        "above_ema21": above_ema21,
        "above_sma50": above_sma50,
        "above_sma200": above_sma200,
        "long_status": status,
    }


def build_long_summary_from_context(context: dict) -> str:
    status = context.get("long_status")
    if status == "Strong Long Setup":
        return "Strong long candidate: price is above EMA21, SMA50, and SMA200 with rising trend structure."
    if status == "Long Watchlist":
        return "Watchlist only: price is above key short-term averages but not fully confirmed across all trend conditions."
    if status == "Pullback Entry Setup":
        return "Pullback setup: price is holding near EMA8 or EMA21 while the larger trend remains bullish."
    if status == "Too Extended":
        return "Too extended: trend is strong, but price is stretched above EMA21. Wait for consolidation or a pullback."
    if status == "Below Key MAs":
        return "Avoid for now: price is below EMA21 or SMA50, so it is not a clean long setup."
    if status == "Insufficient Data":
        return "Insufficient data: not enough moving-average history is available to classify this setup."
    return "Weak / avoid: price and moving-average structure do not support a long-only setup."


def render_selected_stock_header(
    detail_row: dict,
    memberships: list[str],
    fundamentals: dict,
    earnings_info: dict,
    latest_filing: dict | None = None,
    chart_history: object = None,
) -> None:
    metrics = detail_row.get("metrics", {})
    company_name = fundamentals.get("longName") or fundamentals.get("shortName")
    ticker = detail_row.get("ticker", "N/A")
    heading = f"{ticker} - {company_name}" if company_name else ticker
    long_context = get_long_trend_context(detail_row, chart_history=chart_history)
    subtitle = "Long-only EMA chart view"

    current_price = detail_row.get("price")
    extended_quote = detail_row.get("extended_quote") or {}
    header_stats = [
        ("Price", format_price(current_price)),
        ("Day", format_change_pct(detail_row.get("change_pct"))),
    ]
    if extended_quote.get("price") is not None:
        extended_value = format_price(extended_quote.get("price"))
        if extended_quote.get("change_pct") is not None:
            extended_value = f"{extended_value} {format_signed_percent(extended_quote.get('change_pct'))}"
        header_stats.append((extended_quote.get("label") or "Extended", extended_value))
    header_stats.extend(
        [
            ("Trend", classify_simple_trend_status(long_context)),
            ("vs EMA21", format_signed_percent(long_context.get("distance_ema21"))),
        ]
    )

    render_compact_detail_header(
        heading=heading,
        subtitle=subtitle,
        stats=header_stats,
    )


def get_trend_status(detail_row: dict, chart_history: object = None) -> str:
    metrics = detail_row.get("metrics", {})
    price = safe_numeric(detail_row.get("price"))
    if price is None:
        return "Unavailable"

    ema21 = safe_numeric(metrics.get("ema_21"))
    sma50 = safe_numeric(metrics.get("sma50") or get_sma_from_history(chart_history, 50))
    sma200 = safe_numeric(metrics.get("sma200") or get_sma_from_history(chart_history, 200))
    available_levels = [value for value in (ema21, sma50, sma200) if value not in (None, 0)]
    if not available_levels:
        return "Unavailable"

    above_count = sum(1 for level in available_levels if price > level)
    if above_count == len(available_levels):
        return "Above key MAs"
    if above_count == 0:
        return "Below key MAs"
    return f"Above {above_count}/{len(available_levels)} MAs"


def get_warning_status(detail_row: dict) -> str:
    penalty = safe_numeric(detail_row.get("penalty_score"))
    tags = detail_row.get("tags", []) or []
    warning_tags = [
        tag for tag in tags
        if any(keyword in str(tag).lower() for keyword in ("risk", "late", "earnings", "extended", "low liquidity", "wide"))
    ]
    if warning_tags:
        return ", ".join(str(tag) for tag in warning_tags[:2])
    if penalty is not None and penalty > 0:
        return f"Penalty {penalty:.1f}"
    return "No major warning"


def render_quick_decision_cards(detail_row: dict, chart_history: object = None) -> None:
    context = get_long_trend_context(detail_row, chart_history=chart_history)
    volume_ratio = safe_numeric(detail_row.get("volume_ratio"))
    volume_status = "N/A" if volume_ratio is None else ("Acceptable" if volume_ratio >= 0.8 else "Light")

    render_dashboard_section_header("Long Trend Decision", "Read this first: price location, moving-average confirmation, extension, and volume.")
    render_dashboard_stat_grid(
        [
            {
                "label": "Long Status",
                "value": context.get("long_status"),
                "context": "Long-only trend classification",
                "large": True,
            },
            {
                "label": "EMA Stack",
                "value": context.get("ema_stack_status"),
                "context": "Price / EMA8 / EMA21 / SMA50 / SMA200",
                "large": True,
            },
            {
                "label": "EMA8 Status",
                "value": context.get("ema8_status"),
                "context": f"vs EMA8: {format_signed_percent(context.get('distance_ema8'))}",
                "large": True,
            },
            {
                "label": "EMA21 Status",
                "value": context.get("ema21_status"),
                "context": f"vs EMA21: {format_signed_percent(context.get('distance_ema21'))}",
                "large": True,
            },
            {
                "label": "Extension",
                "value": context.get("extension_status"),
                "context": f"Pullback: {context.get('pullback_quality')}",
                "tone": "warning" if context.get("extension_status") in {"Extended", "Too Extended"} else "neutral",
                "large": True,
            },
            {
                "label": "Volume Status",
                "value": volume_status,
                "context": f"Rel Volume: {format_ratio(detail_row.get('volume_ratio'))}",
                "tone": "warning" if volume_status == "Light" else "neutral",
                "large": True,
            },
        ],
        columns=3,
    )


def build_trader_summary(detail_row: dict, chart_history: object = None) -> str:
    summary = detail_row.get("long_summary")
    if summary:
        return summary
    return build_long_summary_from_context(get_long_trend_context(detail_row, chart_history=chart_history))


def render_trader_summary(detail_row: dict, chart_history: object = None) -> None:
    render_dashboard_section_header("Trader Summary")
    st.markdown(
        f"<div class='trader-summary'>{escape(build_trader_summary(detail_row, chart_history=chart_history))}</div>",
        unsafe_allow_html=True,
    )


def render_key_facts(
    detail_row: dict,
    memberships: list[str],
    fundamentals: dict,
    earnings_info: dict,
    chart_history: object = None,
) -> None:
    context = get_long_trend_context(detail_row, chart_history=chart_history)
    metrics = detail_row.get("metrics", {})
    current_price = detail_row.get("price")
    membership_text = ", ".join(get_universe_label(name) for name in memberships) if memberships else "Not in selected universes"
    pct_from_high = calculate_percent_change(current_price, fundamentals.get("fiftyTwoWeekHigh"))
    pct_above_low = calculate_percent_change(current_price, fundamentals.get("fiftyTwoWeekLow"))

    render_dashboard_section_header("EMA Trend Summary", "Long-only trend context using EMA21, SMA50, and SMA200.")
    sections = [
        (
            "Trend Decision",
            [
                ("Price", format_price(current_price)),
                ("Long Status", context.get("long_status")),
                ("EMA Stack", context.get("ema_stack_status")),
                ("EMA8 Status", context.get("ema8_status")),
                ("EMA21 Status", context.get("ema21_status")),
                ("Extension", context.get("extension_status")),
                ("Pullback Quality", context.get("pullback_quality")),
                ("Above EMA8", format_yes_no(context.get("above_ema8"))),
            ],
        ),
        (
            "Moving Averages",
            [
                ("EMA8", format_price(context.get("ema8"))),
                ("EMA21", format_price(context.get("ema21"))),
                ("SMA50", format_price(context.get("sma50"))),
                ("SMA200", format_price(context.get("sma200"))),
                ("EMA8 Trend", context.get("ema8_trend")),
                ("EMA21 Trend", context.get("ema21_trend")),
                ("SMA50 Trend", context.get("sma50_trend")),
            ],
        ),
        (
            "Extension",
            [
                ("vs EMA8", format_signed_percent(context.get("distance_ema8"))),
                ("vs EMA21", format_signed_percent(context.get("distance_ema21"))),
                ("vs SMA50", format_signed_percent(context.get("distance_sma50"))),
                ("vs SMA200", format_signed_percent(context.get("distance_sma200"))),
                ("RSI", format_optional_number(get_latest_rsi(chart_history), decimals=1)),
                ("ATR", format_percent_points(metrics.get("atr_pct") or detail_row.get("atr_pct"))),
            ],
        ),
        (
            "Volume / Range",
            [
                ("Index / Universe", membership_text),
                ("Day %", format_change_pct(detail_row.get("change_pct"))),
                ("Volume", format_compact_number(detail_row.get("volume"))),
                ("Avg Volume", format_compact_number(detail_row.get("avg_volume") or fundamentals.get("averageVolume"))),
                ("Relative Volume", format_optional_number(detail_row.get("volume_ratio"), decimals=2)),
                ("52W High", format_price(fundamentals.get("fiftyTwoWeekHigh"))),
                ("% From High", format_signed_percent_optional(pct_from_high)),
            ],
        ),
        (
            "Fundamentals Snapshot",
            [
                ("Market Cap", format_compact_number(fundamentals.get("marketCap"))),
                ("Sector", format_text_value(fundamentals.get("sector"))),
                ("Industry", format_text_value(fundamentals.get("industry"))),
                ("Short Float", format_percent_ratio(fundamentals.get("shortPercentOfFloat"))),
                ("Revenue Growth", format_percent_optional(fundamentals.get("revenueGrowth"))),
                ("Forward P/E", format_multiple(fundamentals.get("forwardPE"))),
                ("Next Earnings", earnings_info.get("next_earnings_date") or "N/A"),
                ("52W Low", format_price(fundamentals.get("fiftyTwoWeekLow"))),
                ("% Above Low", format_signed_percent_optional(pct_above_low)),
            ],
        ),
    ]

    for index in range(0, len(sections), 2):
        cols = st.columns(2)
        for col, (section_title, rows) in zip(cols, sections[index:index + 2]):
            with col:
                render_compact_metric_grid(section_title, rows, columns=2)


def render_overview_tab(detail_row: dict, chart_history: object = None) -> None:
    context = get_long_trend_context(detail_row, chart_history=chart_history)
    render_compact_metric_grid(
        "Long Trend Checklist",
        [
            ("Ticker", detail_row.get("ticker")),
            ("Long Status", context.get("long_status")),
            ("Price", format_price(detail_row.get("price"))),
            ("Day %", format_change_pct(detail_row.get("change_pct"))),
            ("Volume", format_compact_number(detail_row.get("volume"))),
            ("Rel Volume", format_optional_number(detail_row.get("volume_ratio"), decimals=2)),
            ("EMA Stack", context.get("ema_stack_status")),
            ("EMA8 Status", context.get("ema8_status")),
            ("EMA21 Status", context.get("ema21_status")),
            ("Extension", context.get("extension_status")),
            ("Pullback Quality", context.get("pullback_quality")),
            ("EMA8 Trend", context.get("ema8_trend")),
            ("EMA21 Trend", context.get("ema21_trend")),
            ("SMA50 Trend", context.get("sma50_trend")),
        ],
        columns=4,
    )
    render_compact_metric_grid(
        "Moving Average Position",
        [
            ("EMA8", format_price(context.get("ema8"))),
            ("vs EMA8", format_signed_percent(context.get("distance_ema8"))),
            ("EMA21", format_price(context.get("ema21"))),
            ("vs EMA21", format_signed_percent(context.get("distance_ema21"))),
            ("SMA50", format_price(context.get("sma50"))),
            ("vs SMA50", format_signed_percent(context.get("distance_sma50"))),
            ("SMA200", format_price(context.get("sma200"))),
            ("vs SMA200", format_signed_percent(context.get("distance_sma200"))),
        ],
        columns=4,
    )
    with st.expander("Legacy score details", expanded=False):
        render_score_explanation(detail_row)


def render_simple_ma_summary(detail_row: dict, chart_history: object = None) -> None:
    context = get_long_trend_context(detail_row, chart_history=chart_history)
    trend_status = classify_simple_trend_status(context)
    ma_items = [
        ("EMA8", "ema8", "distance_ema8", "above_ema8"),
        ("EMA21", "ema21", "distance_ema21", "above_ema21"),
        ("SMA50", "sma50", "distance_sma50", "above_sma50"),
        ("SMA200", "sma200", "distance_sma200", "above_sma200"),
    ]
    cards = []
    for label, value_key, distance_key, above_key in ma_items:
        distance_label = f"vs {label}"
        above_label = f"Above {label}"
        cards.append(
            "<div class='ma-card'>"
            f"<div class='ma-card-title'>{escape(label)}</div>"
            "<div class='ma-card-row'>"
            f"<span class='ma-card-label'>{escape(label)}</span>"
            f"<span class='ma-card-value'>{escape(format_price(context.get(value_key)))}</span>"
            "</div>"
            "<div class='ma-card-row'>"
            f"<span class='ma-card-label'>{escape(distance_label)}</span>"
            f"<span class='ma-card-value'>{escape(format_signed_percent(context.get(distance_key)))}</span>"
            "</div>"
            "<div class='ma-card-row'>"
            f"<span class='ma-card-label'>{escape(above_label)}</span>"
            f"<span class='ma-card-value'>{escape(format_yes_no(context.get(above_key)))}</span>"
            "</div>"
            "</div>"
        )

    st.markdown(
        "<div class='detail-section'>"
        "<div class='detail-section-title'>Moving Average Summary</div>"
        "</div>"
        "<div class='ma-summary-status'>"
        "<span class='ma-summary-status-label'>Trend Status</span>"
        f"<span class='ma-summary-status-value'>{escape(format_metric_value(trend_status))}</span>"
        "</div>"
        f"<div class='ma-card-grid'>{''.join(cards)}</div>",
        unsafe_allow_html=True,
    )


def format_reclaim_percent(value: object) -> str:
    numeric_value = safe_numeric(value)
    if numeric_value is None:
        return "—"
    return f"{numeric_value * 100:+.1f}%"


def format_reclaim_win_rate(value: object) -> str:
    numeric_value = safe_numeric(value)
    if numeric_value is None:
        return "—"
    return f"{numeric_value * 100:.0f}%"


def get_reclaim_reliability_label(event_count: object) -> str:
    numeric_value = safe_numeric(event_count)
    if numeric_value is None:
        return "Low"
    if numeric_value < 8:
        return "Low"
    if numeric_value < 20:
        return "Medium"
    return "Strong"


def get_reclaim_edge_label(row: dict) -> str:
    median_60d = safe_numeric(row.get("median_return_60d"))
    win_rate_60d = safe_numeric(row.get("win_rate_60d"))
    if median_60d is None or win_rate_60d is None:
        return "Unknown"
    if median_60d >= 0.04 and win_rate_60d >= 0.60:
        return "Strong"
    if median_60d > 0 and win_rate_60d >= 0.50:
        return "Moderate"
    return "Weak"


def get_reclaim_edge_rank(edge_label: str) -> int:
    return {"Strong": 3, "Moderate": 2, "Weak": 1, "Unknown": 0}.get(edge_label, 0)


def format_reclaim_reward_risk(row: dict) -> str:
    runup = safe_numeric(row.get("median_max_runup_60d"))
    drawdown = safe_numeric(row.get("median_max_drawdown_60d"))
    if runup is None or drawdown in (None, 0):
        return "—"
    return f"{runup / abs(drawdown):.2f}R"


def get_best_reclaim_row(rows: list[dict]) -> dict | None:
    usable_rows = [
        row for row in rows
        if safe_numeric(row.get("event_count")) and safe_numeric(row.get("median_return_60d")) is not None
    ]
    if not usable_rows:
        return None
    return max(
        usable_rows,
        key=lambda row: (
            get_reclaim_edge_rank(get_reclaim_edge_label(row)),
            safe_numeric(row.get("median_return_60d")) or -999,
            safe_numeric(row.get("win_rate_60d")) or -999,
        ),
    )


def render_reclaim_summary(rows: list[dict], ticker: str | None = None) -> None:
    best_row = get_best_reclaim_row(rows)
    if not best_row:
        st.caption("Not enough completed reclaim history to summarize.")
        return

    ticker_label = ticker or "this stock"
    event_count = int(best_row.get("event_count") or 0)
    st.caption(
        f"Best historical reclaim for {ticker_label}: {best_row.get('signal')}, "
        f"with a {format_reclaim_percent(best_row.get('median_return_60d'))} 60D median return, "
        f"{format_reclaim_win_rate(best_row.get('win_rate_60d'))} 60D win rate, and "
        f"{format_reclaim_percent(best_row.get('median_max_runup_60d'))} median 60D run-up "
        f"across {event_count} completed events."
    )


def render_historical_ma_reclaim_returns(ma_reclaim_stats: dict | None, ticker: str | None = None) -> None:
    st.markdown("#### Historical MA Reclaim Returns")
    st.caption(
        "Shows how this stock historically performed after reclaiming key daily moving averages. "
        "Median return is usually more reliable than average because one large move can distort the average."
    )
    st.caption("Reliability is sample size only. Historical Edge is informational and does not affect the signal score.")

    if not ma_reclaim_stats or not ma_reclaim_stats.get("rows"):
        st.info("Historical moving-average reclaim data is unavailable for this stock.")
        return

    rows = ma_reclaim_stats.get("rows", [])
    render_reclaim_summary(rows, ticker=ticker)

    table_rows = []
    for row in rows:
        event_count = int(row.get("event_count") or 0)
        table_rows.append(
            {
                "Signal": row.get("signal") or "—",
                "Events": event_count,
                "Reliability": get_reclaim_reliability_label(event_count),
                "20D Median": format_reclaim_percent(row.get("median_return_20d")),
                "60D Median": format_reclaim_percent(row.get("median_return_60d")),
                "20D Win Rate": format_reclaim_win_rate(row.get("win_rate_20d")),
                "60D Win Rate": format_reclaim_win_rate(row.get("win_rate_60d")),
                "60D Median Run-Up": format_reclaim_percent(row.get("median_max_runup_60d")),
                "60D Median Drawdown": format_reclaim_percent(row.get("median_max_drawdown_60d")),
                "60D Reward/Risk": format_reclaim_reward_risk(row),
                "Historical Edge": get_reclaim_edge_label(row),
            }
        )

    st.dataframe(table_rows, use_container_width=True, hide_index=True)

    with st.expander("Show short-term / average return details", expanded=False):
        detail_rows = []
        for row in rows:
            detail_rows.append(
                {
                    "Signal": row.get("signal") or "—",
                    "5D Median": format_reclaim_percent(row.get("median_return_5d")),
                    "10D Median": format_reclaim_percent(row.get("median_return_10d")),
                    "20D Median": format_reclaim_percent(row.get("median_return_20d")),
                    "60D Median": format_reclaim_percent(row.get("median_return_60d")),
                    "5D Average": format_reclaim_percent(row.get("average_return_5d")),
                    "10D Average": format_reclaim_percent(row.get("average_return_10d")),
                    "20D Average": format_reclaim_percent(row.get("average_return_20d")),
                    "60D Average": format_reclaim_percent(row.get("average_return_60d")),
                }
            )
        st.dataframe(detail_rows, use_container_width=True, hide_index=True)

    active_reclaims = ma_reclaim_stats.get("active_recent_reclaims", [])
    if active_reclaims:
        st.markdown("##### Recent Active Reclaims")
        active_rows = []
        for reclaim in active_reclaims:
            active_rows.append(
                {
                    "Signal": reclaim.get("signal") or "—",
                    "Reclaim Date": reclaim.get("date") or "—",
                    "Trading Days Ago": reclaim.get("days_since") if reclaim.get("days_since") is not None else "—",
                    "Status": "Incomplete forward data",
                }
            )
        st.dataframe(active_rows, use_container_width=True, hide_index=True)


FUNDAMENTAL_SECTION_LABELS = {
    "Company Profile": "Overview",
    "Share Structure / Ownership": "Ownership",
}

FUNDAMENTAL_MONEY_LABELS = {
    "Market Cap",
    "Enterprise Value",
    "Revenue",
    "Revenue (TTM)",
    "Gross Profit (TTM)",
    "Cost of Revenue (TTM)",
    "R&D Expense (TTM)",
    "SG&A Expense (TTM)",
    "Operating Income",
    "Operating Income (TTM)",
    "EBITDA",
    "EBITDA (TTM)",
    "D&A (TTM)",
    "Interest Expense (TTM)",
    "Pretax Income (TTM)",
    "Income Tax Expense (TTM)",
    "Net Income",
    "Net Income (TTM)",
    "Total Cash",
    "Short-Term Debt",
    "Long-Term Debt",
    "Total Debt",
    "Total Assets",
    "Current Assets",
    "Total Liabilities",
    "Current Liabilities",
    "Stockholders Equity",
    "Retained Earnings",
    "Inventory",
    "Accounts Receivable",
    "Accounts Payable",
    "Operating Cash Flow",
    "Free Cash Flow",
    "Levered Free Cash Flow",
    "CapEx",
    "Target High Price",
    "Target Mean Price",
    "Target Low Price",
    "Latest Quarter Revenue",
    "Latest Quarter Gross Profit",
    "Latest Quarter Cost of Revenue",
    "Latest Quarter R&D Expense",
    "Latest Quarter SG&A Expense",
    "Latest Quarter Operating Income",
    "Latest Quarter Interest Expense",
    "Latest Quarter Pretax Income",
    "Latest Quarter Income Tax Expense",
    "Latest Quarter Net Income",
    "52 Week High",
    "52 Week Low",
    "Net Income to Common",
}
FUNDAMENTAL_PERCENT_LABELS = {
    "Revenue Growth",
    "Earnings Growth",
    "Quarterly Revenue Growth YoY",
    "Quarterly Earnings Growth YoY",
    "Gross Margin",
    "Gross Margin (TTM)",
    "Latest Quarter Gross Margin",
    "Operating Margin",
    "Operating Margin (TTM)",
    "Latest Quarter Operating Margin",
    "Profit Margin",
    "Net Margin (TTM)",
    "Latest Quarter Net Margin",
    "EBITDA Margin",
    "Return on Equity",
    "Return on Assets",
    "Effective Tax Rate (TTM)",
    "Free Cash Flow Margin (TTM)",
    "Return on Invested Capital",
    "Free Cash Flow Yield",
    "Short Percent of Float",
    "Percent Held by Insiders",
    "Percent Held by Institutions",
    "Dividend Yield",
    "Payout Ratio",
}
FUNDAMENTAL_SIGNED_PERCENT_LABELS = {
    "EPS Growth YoY",
    "Revenue Growth",
    "Earnings Growth",
    "Quarterly Revenue Growth YoY",
    "Quarterly Earnings Growth YoY",
    "Current Price vs Target Mean %",
    "Distance From 52W High %",
    "Distance From 52W Low %",
    "Performance 1W",
    "Performance 1M",
    "Performance 3M",
    "Performance 6M",
    "Performance YTD",
    "Performance 1Y",
}
FUNDAMENTAL_MULTIPLE_LABELS = {
    "Trailing P/E",
    "Forward P/E",
    "PEG Ratio",
    "Price / Sales",
    "Price / Book",
    "EV / Revenue",
    "EV / EBITDA",
    "Price / Cash Flow",
    "Debt / Equity",
    "Current Ratio",
    "Quick Ratio",
    "Short Ratio",
}
FUNDAMENTAL_SHARE_LABELS = {
    "Shares Outstanding",
    "Diluted Shares",
    "Float Shares",
    "Shares Short",
    "Implied Shares Outstanding",
    "Full-Time Employees",
    "Average Volume",
    "10-Day Average Volume",
}
FUNDAMENTAL_DATE_LABELS = {
    "Fiscal Year End",
    "Most Recent Quarter",
    "Ex-Dividend Date",
    "Earnings Date",
}
FUNDAMENTAL_EPS_LABELS = {
    "EPS (TTM)",
    "Forward EPS",
    "EPS Current Year",
    "EPS Next Year",
    "EPS Next Quarter",
    "EPS Estimate Current Quarter",
    "EPS Estimate Next Quarter",
    "EPS Trailing Twelve Months",
    "EPS Forward",
    "Basic EPS (TTM)",
    "Diluted EPS (TTM)",
    "Latest Quarter Basic EPS",
    "Latest Quarter Diluted EPS",
}
FUNDAMENTAL_POSITIVE_LABELS = FUNDAMENTAL_PERCENT_LABELS | FUNDAMENTAL_SIGNED_PERCENT_LABELS


def normalize_snapshot_value(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().upper() in {"", "N/A", "NONE", "NAN"}:
        return None
    return value


def format_epoch_date(value: object) -> str:
    if isinstance(value, (list, tuple)):
        value = next((item for item in value if item is not None), None)
        if value is None:
            return "N/A"
    if hasattr(value, "strftime"):
        try:
            return value.strftime("%Y-%m-%d")
        except (TypeError, ValueError):
            pass
    numeric_value = safe_numeric(value)
    if numeric_value is None:
        return format_text_value(value)
    try:
        return datetime.fromtimestamp(numeric_value).strftime("%Y-%m-%d")
    except (OverflowError, OSError, ValueError):
        return format_text_value(value)


def format_snapshot_percent(value: object, signed: bool = False) -> str:
    numeric_value = safe_numeric(value)
    if numeric_value is None:
        return "N/A"
    display_value = numeric_value * 100 if abs(numeric_value) <= 1 else numeric_value
    prefix = "+" if signed and display_value > 0 else ""
    return f"{prefix}{display_value:.1f}%"


def format_eps(value: object) -> str:
    numeric_value = safe_numeric(value)
    if numeric_value is None:
        return "N/A"
    absolute_value = abs(numeric_value)
    sign = "-" if numeric_value < 0 else ""
    return f"{sign}${absolute_value:,.2f}"


def format_fundamental_snapshot_value(section: str, label: str, value: object) -> str:
    value = normalize_snapshot_value(value)
    if value is None:
        return ""
    if label in FUNDAMENTAL_MONEY_LABELS:
        if label in {"Target High Price", "Target Mean Price", "Target Low Price", "52 Week High", "52 Week Low"}:
            return format_price(safe_numeric(value))
        return format_compact_money(value)
    if label in FUNDAMENTAL_SIGNED_PERCENT_LABELS:
        return format_snapshot_percent(value, signed=True)
    if label in FUNDAMENTAL_PERCENT_LABELS:
        return format_snapshot_percent(value)
    if label in FUNDAMENTAL_MULTIPLE_LABELS:
        return format_debt_to_equity(value) if label == "Debt / Equity" else format_multiple(value)
    if label in FUNDAMENTAL_SHARE_LABELS:
        return format_compact_number(value)
    if label in FUNDAMENTAL_DATE_LABELS:
        return format_epoch_date(value)
    if label in FUNDAMENTAL_EPS_LABELS:
        return format_eps(value)
    if label in {"Cash Per Share", "Book Value Per Share", "Revenue Per Share", "Dividend Rate"}:
        return format_price(safe_numeric(value))
    if label in {"Beta", "RSI 14", "Relative Volume", "Recommendation Mean", "Number of Analyst Opinions"}:
        return format_optional_number(value)
    return format_text_value(value)


def get_snapshot_value_class(label: str, display_value: str) -> str:
    if display_value in {"", "N/A"}:
        return "fs-muted"
    numeric_value = safe_numeric(display_value.replace("%", "").replace("$", "").replace(",", "").rstrip("x"))
    if numeric_value is None:
        return ""
    if label in FUNDAMENTAL_POSITIVE_LABELS:
        if numeric_value > 0:
            return "fs-positive"
        if numeric_value < 0:
            return "fs-negative"
    return ""


def render_snapshot_metric_grid(title: str, section: dict, columns: int = 4) -> None:
    cells = []
    for label, raw_value in section.items():
        display_value = format_fundamental_snapshot_value(title, label, raw_value)
        value_class = get_snapshot_value_class(label, display_value)
        cells.append(
            "<div class='fs-cell'>"
            f"<div class='fs-label'>{escape(str(label))}</div>"
            f"<div class='fs-value {value_class}'>{escape(display_value)}</div>"
            "</div>"
        )
    st.markdown(
        "<div class='fs-box'>"
        f"<div class='fs-box-title'>{escape(title)}</div>"
        f"<div class='fs-grid' style='--fs-cols:{max(1, columns)}'>{''.join(cells)}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_fundamental_quality_summary(snapshot: dict) -> None:
    quality = snapshot.get("__quality__") or {}
    score = quality.get("score")
    label = quality.get("label") or "Insufficient Data"
    score_text = "N/A" if score is None else f"{int(score)} / 100"
    st.markdown(
        "<div class='fund-quality'>"
        f"<div class='fund-quality-title'>Fundamental Quality: {escape(score_text)} - {escape(label)}</div>"
        "<div class='fund-quality-note'>Fundamental Quality is separate from the technical Signal Score. "
        "A stock can have a strong chart but weak fundamentals, or strong fundamentals but a weak chart.</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_fundamentals_guide() -> None:
    guide_rows = [
        ("Market Cap", "Total equity value of the company."),
        ("Enterprise Value", "Market cap plus debt minus cash. Useful when comparing companies with different debt levels."),
        ("P/E", "Price divided by earnings per share. Lower can be cheaper, but only if earnings quality is strong."),
        ("Forward P/E", "Price divided by expected future earnings."),
        ("PEG", "P/E adjusted for expected growth."),
        ("Price / Sales", "Market cap divided by revenue."),
        ("Price / Book", "Price compared with book value."),
        ("EV / EBITDA", "Enterprise value compared with EBITDA. Often used to compare companies across capital structures."),
        ("EPS (TTM)", "Earnings per share over the trailing twelve months."),
        ("Forward EPS", "Expected earnings per share based on analyst estimates."),
        ("EPS Current Year", "Estimated earnings per share for the current fiscal year."),
        ("EPS Next Year", "Estimated earnings per share for the next fiscal year."),
        ("Earnings Growth", "Growth rate of earnings, usually year over year when available."),
        ("Quarterly Earnings Growth YoY", "Most recent quarter's earnings growth compared with the same quarter last year."),
        ("Revenue Per Share", "Revenue divided by shares outstanding."),
        ("Net Income to Common", "Net income available to common shareholders."),
        ("Gross Margin", "Revenue left after cost of goods sold. SEC view labels whether it is trailing twelve months or latest quarter."),
        ("Operating Margin", "Profitability from core operations. SEC view labels whether it is trailing twelve months or latest quarter."),
        ("Profit Margin", "Net income as a percentage of revenue."),
        ("ROE", "Return on shareholder equity."),
        ("Debt / Equity", "Debt compared with shareholder equity."),
        ("Current Ratio", "Current assets divided by current liabilities."),
        ("Free Cash Flow", "Cash left after operating cash flow and capital expenditures."),
        ("Short Percent of Float", "Percentage of tradable shares sold short."),
        ("Institutional Ownership", "Percentage of shares held by institutions."),
        ("Relative Volume", "Current volume divided by normal volume."),
        ("ATR %", "ATR divided by price. Higher means more volatility."),
        ("Performance 1M / 3M / 6M / 1Y", "Price change over those periods."),
    ]
    with st.expander("Fundamentals Guide", expanded=False):
        for label, description in guide_rows:
            st.caption(f"{label}: {description}")


def get_yahoo_fallback_section(fundamentals_snapshot: dict | None, section_name: str) -> dict:
    section = (fundamentals_snapshot or {}).get(section_name)
    return section.copy() if isinstance(section, dict) else {}


def build_sec_fundamentals_snapshot(sec_inputs: dict, fundamentals_snapshot: dict | None = None) -> dict:
    snapshot = {
        "Company Profile": {
            "Company Name": sec_inputs.get("entity_name"),
            "Financial Data Source": "SEC companyfacts XBRL",
            "Latest SEC Period": sec_inputs.get("latest_period"),
            "Latest SEC Filed": sec_inputs.get("latest_filed"),
        },
        "Valuation": get_yahoo_fallback_section(fundamentals_snapshot, "Valuation"),
        "Income Statement - TTM": {
            "Revenue (TTM)": sec_inputs.get("totalRevenue"),
            "Gross Profit (TTM)": sec_inputs.get("grossProfit"),
            "Cost of Revenue (TTM)": sec_inputs.get("costOfRevenue"),
            "R&D Expense (TTM)": sec_inputs.get("researchAndDevelopment"),
            "SG&A Expense (TTM)": sec_inputs.get("sellingGeneralAdministrative"),
            "Operating Income (TTM)": sec_inputs.get("operatingIncome"),
            "EBITDA (TTM)": sec_inputs.get("ebitda"),
            "D&A (TTM)": sec_inputs.get("depreciationAmortization"),
            "Interest Expense (TTM)": sec_inputs.get("interestExpense"),
            "Pretax Income (TTM)": sec_inputs.get("pretaxIncome"),
            "Income Tax Expense (TTM)": sec_inputs.get("incomeTaxExpense"),
            "Net Income (TTM)": sec_inputs.get("netIncome"),
            "Basic EPS (TTM)": sec_inputs.get("basicEps"),
            "Diluted EPS (TTM)": sec_inputs.get("dilutedEps"),
        },
        "Income Statement - Latest Q": {
            "Latest Quarter Revenue": sec_inputs.get("latestQuarterRevenue"),
            "Latest Quarter Gross Profit": sec_inputs.get("latestQuarterGrossProfit"),
            "Latest Quarter Cost of Revenue": sec_inputs.get("latestQuarterCostOfRevenue"),
            "Latest Quarter R&D Expense": sec_inputs.get("latestQuarterResearchAndDevelopment"),
            "Latest Quarter SG&A Expense": sec_inputs.get("latestQuarterSellingGeneralAdministrative"),
            "Latest Quarter Operating Income": sec_inputs.get("latestQuarterOperatingIncome"),
            "Latest Quarter Interest Expense": sec_inputs.get("latestQuarterInterestExpense"),
            "Latest Quarter Pretax Income": sec_inputs.get("latestQuarterPretaxIncome"),
            "Latest Quarter Income Tax Expense": sec_inputs.get("latestQuarterIncomeTaxExpense"),
            "Latest Quarter Net Income": sec_inputs.get("latestQuarterNetIncome"),
            "Latest Quarter Basic EPS": sec_inputs.get("latestQuarterBasicEps"),
            "Latest Quarter Diluted EPS": sec_inputs.get("latestQuarterDilutedEps"),
        },
        "Margins / Returns": {
            "Gross Margin (TTM)": sec_inputs.get("grossMargins"),
            "Operating Margin (TTM)": sec_inputs.get("operatingMargins"),
            "Net Margin (TTM)": sec_inputs.get("profitMargins"),
            "Latest Quarter Gross Margin": sec_inputs.get("latestQuarterGrossMargins"),
            "Latest Quarter Operating Margin": sec_inputs.get("latestQuarterOperatingMargins"),
            "Latest Quarter Net Margin": sec_inputs.get("latestQuarterProfitMargins"),
            "Effective Tax Rate (TTM)": sec_inputs.get("effectiveTaxRate"),
            "Return on Assets": sec_inputs.get("returnOnAssets"),
            "Return on Equity": sec_inputs.get("returnOnEquity"),
            "Free Cash Flow Margin (TTM)": sec_inputs.get("freeCashFlowMargin"),
        },
        "Balance Sheet": {
            "Total Cash": sec_inputs.get("totalCash"),
            "Short-Term Debt": sec_inputs.get("shortTermDebt"),
            "Long-Term Debt": sec_inputs.get("longTermDebt"),
            "Total Debt": sec_inputs.get("totalDebt"),
            "Total Assets": sec_inputs.get("totalAssets"),
            "Current Assets": sec_inputs.get("currentAssets"),
            "Total Liabilities": sec_inputs.get("totalLiabilities"),
            "Current Liabilities": sec_inputs.get("currentLiabilities"),
            "Stockholders Equity": sec_inputs.get("stockholdersEquity"),
            "Retained Earnings": sec_inputs.get("retainedEarnings"),
            "Inventory": sec_inputs.get("inventory"),
            "Accounts Receivable": sec_inputs.get("accountsReceivable"),
            "Accounts Payable": sec_inputs.get("accountsPayable"),
            "Current Ratio": sec_inputs.get("currentRatio"),
            "Debt / Equity": sec_inputs.get("debtToEquity"),
        },
        "Cash Flow": {
            "Operating Cash Flow": sec_inputs.get("operatingCashflow"),
            "CapEx": sec_inputs.get("capitalExpenditures"),
            "Free Cash Flow": sec_inputs.get("freeCashflow"),
        },
        "Share Structure / Ownership": get_yahoo_fallback_section(fundamentals_snapshot, "Share Structure / Ownership"),
        "Dividends": get_yahoo_fallback_section(fundamentals_snapshot, "Dividends"),
        "Analyst / Target Info": get_yahoo_fallback_section(fundamentals_snapshot, "Analyst / Target Info"),
        "Performance / Risk Snapshot": get_yahoo_fallback_section(fundamentals_snapshot, "Performance / Risk Snapshot"),
    }
    snapshot["__quality__"] = build_fundamental_quality_summary(snapshot)
    return snapshot


def build_yahoo_fundamentals_snapshot(fundamentals_snapshot: dict | None, fundamentals: dict) -> dict:
    snapshot = {
        section: values.copy() if isinstance(values, dict) else values
        for section, values in (fundamentals_snapshot or {}).items()
    }
    if snapshot:
        return snapshot

    if not fundamentals:
        return {}

    snapshot = {
        "Company Profile": {
            "Company Name": fundamentals.get("longName") or fundamentals.get("shortName"),
            "Sector": fundamentals.get("sector"),
            "Industry": fundamentals.get("industry"),
            "Country": fundamentals.get("country"),
            "Exchange": fundamentals.get("exchange") or fundamentals.get("fullExchangeName"),
            "Currency": fundamentals.get("currency") or fundamentals.get("financialCurrency"),
        },
        "Valuation": {
            "Market Cap": fundamentals.get("marketCap"),
            "Enterprise Value": fundamentals.get("enterpriseValue"),
            "Forward P/E": fundamentals.get("forwardPE"),
            "Price / Sales": fundamentals.get("priceToSalesTrailing12Months"),
            "Price / Book": fundamentals.get("priceToBook"),
        },
        "Growth": {
            "Revenue Growth": fundamentals.get("revenueGrowth"),
            "Earnings Growth": fundamentals.get("earningsGrowth"),
        },
        "Profitability": {
            "Gross Margin": fundamentals.get("grossMargins"),
            "Operating Margin": fundamentals.get("operatingMargins"),
            "Profit Margin": fundamentals.get("profitMargins"),
            "Return on Equity": fundamentals.get("returnOnEquity"),
        },
        "Balance Sheet": {
            "Total Cash": fundamentals.get("totalCash"),
            "Total Debt": fundamentals.get("totalDebt"),
            "Debt / Equity": fundamentals.get("debtToEquity"),
            "Current Ratio": fundamentals.get("currentRatio"),
        },
        "Cash Flow": {
            "Operating Cash Flow": fundamentals.get("operatingCashflow"),
            "Free Cash Flow": fundamentals.get("freeCashflow"),
        },
        "Share Structure / Ownership": {
            "Shares Outstanding": fundamentals.get("sharesOutstanding"),
            "Float Shares": fundamentals.get("floatShares"),
            "Short Percent of Float": fundamentals.get("shortPercentOfFloat"),
        },
    }
    snapshot["__quality__"] = build_fundamental_quality_summary(snapshot)
    return snapshot


def apply_current_price_performance_overrides(snapshot: dict, detail_row: dict | None) -> dict:
    current_price = safe_numeric((detail_row or {}).get("price"))
    performance_snapshot = snapshot.get("Performance / Risk Snapshot")
    if current_price is None or not isinstance(performance_snapshot, dict):
        return snapshot

    high_52w = safe_numeric(performance_snapshot.get("52 Week High"))
    low_52w = safe_numeric(performance_snapshot.get("52 Week Low"))
    if high_52w not in (None, 0):
        performance_snapshot["Distance From 52W High %"] = (current_price / high_52w) - 1
    if low_52w not in (None, 0):
        performance_snapshot["Distance From 52W Low %"] = (current_price / low_52w) - 1
    return snapshot


def render_simple_fundamentals_tab(fundamentals: dict, fundamentals_snapshot: dict | None = None, detail_row: dict | None = None) -> None:
    st.markdown("#### Fundamentals Snapshot")
    ticker = str((detail_row or {}).get("ticker") or "").strip().upper()
    sec_inputs = get_sec_dcf_inputs(ticker) if ticker else {}
    sec_available = bool(sec_inputs.get("available"))
    yahoo_available = bool(fundamentals_snapshot or fundamentals)
    if not sec_available and not yahoo_available:
        st.info("No fundamentals available.")
        return

    yahoo_snapshot = build_yahoo_fundamentals_snapshot(fundamentals_snapshot, fundamentals)
    snapshot = build_sec_fundamentals_snapshot(sec_inputs, yahoo_snapshot) if sec_available else yahoo_snapshot
    snapshot = apply_current_price_performance_overrides(snapshot, detail_row)
    if sec_available:
        st.caption(
            (
                f"SEC source: {sec_inputs.get('entity_name') or ticker} | "
                f"Latest period: {sec_inputs.get('latest_period') or 'N/A'} | "
                f"Filed: {sec_inputs.get('latest_filed') or 'N/A'} | "
                f"TTM source periods: {', '.join(sec_inputs.get('source_periods') or []) or 'N/A'}"
            )
        )
        st.caption(
            "SEC statement values are filing-based. Yahoo only fills the market-style sections: Valuation, Share Structure / Ownership, Dividends, Analyst / Target Info, and Performance / Risk Snapshot."
        )
        st.caption("SEC values labeled TTM use current fiscal YTD plus prior fiscal year minus prior-year same YTD. Latest Q values use the latest standalone quarterly filing fact when available.")
    else:
        st.caption("Yahoo fills the available non-filing fields because SEC companyfacts were not available for this ticker.")

    if not snapshot:
        st.info("No fundamentals available.")
        return

    st.caption(
        "Fundamentals Snapshot summarizes valuation, growth, profitability, balance sheet strength, ownership, "
        "dividends, analyst targets, and performance data. Blank cells mean the selected source did not provide that field."
    )

    render_fundamental_quality_summary(snapshot)
    render_fundamentals_guide()

    section_order = [
        "Company Profile",
        "Valuation",
        "Earnings / EPS",
        "Growth",
        "Income Statement - TTM",
        "Income Statement - Latest Q",
        "Margins / Returns",
        "Profitability",
        "Balance Sheet",
        "Cash Flow",
        "Share Structure / Ownership",
        "Dividends",
        "Analyst / Target Info",
        "Performance / Risk Snapshot",
    ]
    for section_name in section_order:
        section = snapshot.get(section_name)
        if isinstance(section, dict):
            if not any(normalize_snapshot_value(value) is not None for value in section.values()):
                continue
            render_snapshot_metric_grid(section_name, section, columns=4)


def format_dcf_table_value(value: object, value_type: str) -> str:
    numeric_value = safe_numeric(value)
    if numeric_value is None:
        return "N/A"
    if value_type == "money":
        return format_compact_money(numeric_value)
    if value_type == "price":
        return format_price(numeric_value)
    if value_type == "percent":
        return f"{numeric_value * 100:.1f}%"
    if value_type == "signed_percent":
        return f"{numeric_value:+.1f}%"
    if value_type == "multiple":
        return f"{numeric_value:.2f}x"
    return f"{numeric_value:,.2f}"


def format_dcf_rows(rows: list[dict], column_types: dict[str, str]) -> list[dict]:
    formatted_rows = []
    for row in rows:
        formatted_row = {}
        for column, value in row.items():
            formatted_row[column] = format_dcf_table_value(value, column_types.get(column, "number"))
        formatted_rows.append(formatted_row)
    return formatted_rows


def get_fundamental_number(fundamentals: dict, *keys: str, default: float | None = None) -> float | None:
    for key in keys:
        value = safe_float((fundamentals or {}).get(key))
        if value is not None:
            return value
    return default


def clamp_slider_default(value: float, minimum: float, maximum: float) -> float:
    return min(max(float(value), minimum), maximum)


def format_dcf_percent(value: object) -> str:
    numeric_value = safe_numeric(value)
    return "N/A" if numeric_value is None else f"{numeric_value * 100:.2f}%"


def build_dcf_summary(result: dict, assumptions: DcfAssumptions) -> str:
    fair_value = result.get("fair_value_per_share")
    current_price = assumptions.current_price
    terminal_share = result.get("terminal_value_share")
    margin = assumptions.current_operating_margin
    if fair_value is None or current_price in (None, 0):
        return "DCF output is available, but the per-share comparison is limited because price or share count data is missing."
    if margin < 0:
        return "DCF may be unreliable because current operating margin is negative; the story depends heavily on margin recovery."
    if terminal_share is not None and terminal_share > 0.80:
        return "Most of the value is coming from terminal value, so the model is sensitive to WACC and terminal growth."
    if fair_value > current_price:
        return "Base case suggests the stock is undervalued relative to the current price, assuming the revenue, margin, and reinvestment story holds."
    return "Base case suggests the stock is overvalued because fair value is below the current price."


def render_dcf_result_table(title: str, rows: list[dict], column_types: dict[str, str]) -> None:
    st.markdown(f"**{title}**")
    if not rows:
        st.info(f"No {title.lower()} data available.")
        return
    st.dataframe(format_dcf_rows(rows, column_types), use_container_width=True, hide_index=True)


def render_damodaran_dcf_tab(fundamentals: dict, detail_row: dict | None = None) -> None:
    fundamentals = fundamentals or {}
    detail_row = detail_row or {}
    ticker = str(detail_row.get("ticker") or "").strip().upper()
    sec_dcf_inputs = get_sec_dcf_inputs(ticker) if ticker else {}
    sec_available = bool(sec_dcf_inputs.get("available"))
    combined_dcf_values = fundamentals.copy()
    if sec_available:
        combined_dcf_values.update(
            {
                "totalRevenue": sec_dcf_inputs.get("totalRevenue"),
                "grossProfit": sec_dcf_inputs.get("grossProfit"),
                "costOfRevenue": sec_dcf_inputs.get("costOfRevenue"),
                "grossMargins": sec_dcf_inputs.get("grossMargins"),
                "researchAndDevelopment": sec_dcf_inputs.get("researchAndDevelopment"),
                "sellingGeneralAdministrative": sec_dcf_inputs.get("sellingGeneralAdministrative"),
                "operatingIncome": sec_dcf_inputs.get("operatingIncome"),
                "operatingMargins": sec_dcf_inputs.get("operatingMargins"),
                "ebitda": sec_dcf_inputs.get("ebitda"),
                "depreciationAmortization": sec_dcf_inputs.get("depreciationAmortization"),
                "interestExpense": sec_dcf_inputs.get("interestExpense"),
                "pretaxIncome": sec_dcf_inputs.get("pretaxIncome"),
                "incomeTaxExpense": sec_dcf_inputs.get("incomeTaxExpense"),
                "netIncome": sec_dcf_inputs.get("netIncome"),
                "profitMargins": sec_dcf_inputs.get("profitMargins"),
                "totalCash": sec_dcf_inputs.get("totalCash"),
                "totalDebt": sec_dcf_inputs.get("totalDebt"),
                "sharesOutstanding": sec_dcf_inputs.get("sharesOutstanding"),
                "minorityInterest": sec_dcf_inputs.get("minorityInterest"),
                "totalInvestments": sec_dcf_inputs.get("totalInvestments"),
                "freeCashflow": sec_dcf_inputs.get("freeCashflow"),
            }
        )
    revenue_default = get_fundamental_number(combined_dcf_values, "totalRevenue", default=0.0) or 0.0
    current_price = safe_numeric(detail_row.get("price"))
    current_margin_default = get_fundamental_number(combined_dcf_values, "operatingMargins", default=0.10) or 0.10
    if current_margin_default < -0.50 or current_margin_default > 0.70:
        current_margin_default = 0.10

    ebit_default = get_fundamental_number(combined_dcf_values, "operatingIncome", "ebit")
    if ebit_default is None and revenue_default:
        ebit_default = revenue_default * current_margin_default

    beta_default = get_fundamental_number(fundamentals, "beta", default=1.0) or 1.0
    cash_default = get_fundamental_number(combined_dcf_values, "totalCash", default=0.0) or 0.0
    debt_default = get_fundamental_number(combined_dcf_values, "totalDebt", default=0.0) or 0.0
    shares_default = get_fundamental_number(combined_dcf_values, "sharesOutstanding", default=0.0) or 0.0
    minority_default = get_fundamental_number(combined_dcf_values, "minorityInterest", default=0.0) or 0.0
    investments_default = get_fundamental_number(combined_dcf_values, "totalInvestments", "longTermInvestments", default=0.0) or 0.0
    free_cash_flow = get_fundamental_number(combined_dcf_values, "freeCashflow")
    try:
        wacc_inputs = get_default_wacc_inputs(ticker)
    except Exception:
        wacc_inputs = {"wacc": 0.10, "source_note": "Using default WACC fallback."}
    auto_wacc_default = clamp_slider_default((safe_numeric(wacc_inputs.get("wacc")) or 0.10) * 100, 4.0, 20.0)

    st.caption("This is a story-driven FCFF model: revenue growth, margins, reinvestment efficiency, cost of capital, and terminal excess returns drive value.")
    with st.expander("DCF Model Method", expanded=False):
        st.caption("Operating value is built from revenue, operating margin, taxes, reinvestment, FCFF, and a cost-of-capital path.")
        st.caption("Explicit reinvestment is tied to growth through sales-to-capital: reinvestment = change in revenue / sales-to-capital.")
        st.caption("Terminal reinvestment is tied to stable growth and mature ROIC: terminal reinvestment rate = terminal growth / terminal ROIC.")
        st.caption("Enterprise value is bridged to equity value by adding cash/investments and subtracting debt and minority interest.")
    if sec_available:
        st.caption(
            (
                f"SEC source: {sec_dcf_inputs.get('entity_name') or ticker} | "
                f"Latest period: {sec_dcf_inputs.get('latest_period') or 'N/A'} | "
                f"Filed: {sec_dcf_inputs.get('latest_filed') or 'N/A'} | "
                f"Periods used for TTM: {', '.join(sec_dcf_inputs.get('source_periods') or []) or 'N/A'}"
            )
        )
        st.caption("DCF uses SEC filings for financial statement inputs and Yahoo for market-linked inputs such as beta and cost-of-capital defaults.")
    else:
        st.caption("DCF uses Yahoo defaults where SEC companyfacts are unavailable.")
    dcf_defaults_key = f"{ticker}-combined"
    if st.session_state.get("dcf-defaults-source-key") != dcf_defaults_key:
        st.session_state["dcf-revenue-b"] = clamp_slider_default(revenue_default / 1_000_000_000, 0.0, 500.0)
        st.session_state["dcf-current-op-margin"] = clamp_slider_default(current_margin_default * 100, -50.0, 50.0)
        st.session_state["dcf-target-margin"] = clamp_slider_default(max(current_margin_default * 100, 12.0), -20.0, 60.0)
        st.session_state["dcf-terminal-margin"] = clamp_slider_default(max(current_margin_default * 100, 12.0), -10.0, 40.0)
        st.session_state["dcf-cash-b"] = max(0.0, cash_default / 1_000_000_000)
        st.session_state["dcf-debt-b"] = max(0.0, debt_default / 1_000_000_000)
        st.session_state["dcf-shares-m"] = max(0.0, shares_default / 1_000_000)
        st.session_state["dcf-minority-b"] = max(0.0, minority_default / 1_000_000_000)
        st.session_state["dcf-investments-b"] = max(0.0, investments_default / 1_000_000_000)
        st.session_state["dcf-terminal-sales-capital"] = 2.0
        st.session_state["dcf-terminal-wacc"] = clamp_slider_default(max(auto_wacc_default - 0.5, 4.0), 4.0, 20.0)
        st.session_state["dcf-defaults-source-key"] = dcf_defaults_key

    with st.expander("Story / Assumptions", expanded=True):
        st.caption("Model structure: value operating assets with FCFF, discount at a cost-of-capital path, then bridge enterprise value to equity value.")
        auto_wacc = st.checkbox("Auto-calculate WACC", value=True, key="dcf-auto-wacc")
        if auto_wacc:
            last_auto_ticker = st.session_state.get("dcf-auto-wacc-ticker")
            if last_auto_ticker != ticker and "dcf-wacc" in st.session_state:
                st.session_state["dcf-wacc"] = auto_wacc_default
            st.session_state["dcf-auto-wacc-ticker"] = ticker
        if st.button("Reset to auto WACC", key="dcf-reset-auto-wacc", disabled=not auto_wacc):
            st.session_state["dcf-wacc"] = auto_wacc_default
        st.caption(f"Auto WACC: {auto_wacc_default:.2f}% {wacc_inputs.get('source_note', 'Using default WACC fallback.')}")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            revenue_billions = st.slider(
                "Starting revenue ($B)",
                min_value=0.0,
                max_value=500.0,
                value=clamp_slider_default(revenue_default / 1_000_000_000, 0.0, 500.0),
                step=0.01,
                key="dcf-revenue-b",
            )
            current_margin = st.slider(
                "Current operating margin (%)",
                min_value=-50.0,
                max_value=50.0,
                value=clamp_slider_default(current_margin_default * 100, -50.0, 50.0),
                step=0.5,
                key="dcf-current-op-margin",
            ) / 100
            tax_rate = st.slider("Marginal tax rate (%)", min_value=0.0, max_value=40.0, value=21.0, step=0.5, key="dcf-tax-rate") / 100
        with col2:
            high_growth = st.slider("Revenue growth years 1-5 (%)", min_value=-20.0, max_value=50.0, value=6.0, step=0.5, key="dcf-high-growth") / 100
            fade_growth = st.slider("Revenue growth year 6 (%)", min_value=-10.0, max_value=30.0, value=4.0, step=0.5, key="dcf-fade-growth") / 100
            projection_years = int(st.slider("Projection years", min_value=5, max_value=15, value=10, step=1, key="dcf-years"))
            high_growth_years = int(st.slider("High-growth years", min_value=1, max_value=max(projection_years - 1, 1), value=min(5, max(projection_years - 1, 1)), step=1, key="dcf-high-growth-years"))
        with col3:
            target_margin = st.slider("Target operating margin (%)", min_value=-20.0, max_value=60.0, value=clamp_slider_default(max(current_margin_default * 100, 12.0), -20.0, 60.0), step=0.5, key="dcf-target-margin") / 100
            years_to_margin = int(st.slider("Years to target margin", min_value=1, max_value=10, value=5, step=1, key="dcf-years-to-margin"))
            terminal_margin = st.slider("Terminal operating margin (%)", min_value=-10.0, max_value=40.0, value=clamp_slider_default(max(current_margin_default * 100, 12.0), -10.0, 40.0), step=0.5, key="dcf-terminal-margin") / 100
        with col4:
            sales_to_capital = st.slider("Sales-to-capital years 1-5", min_value=0.1, max_value=10.0, value=2.0, step=0.1, key="dcf-sales-capital")
            terminal_sales_to_capital = st.slider("Mature sales-to-capital", min_value=0.1, max_value=10.0, value=2.0, step=0.1, key="dcf-terminal-sales-capital")
            wacc_default = auto_wacc_default if auto_wacc else 10.0
            if "dcf-wacc" not in st.session_state:
                st.session_state["dcf-wacc"] = wacc_default
            wacc = st.slider("WACC / cost of capital (%)", min_value=4.0, max_value=20.0, step=0.25, key="dcf-wacc") / 100
            terminal_wacc_default = clamp_slider_default(max((wacc * 100) - 0.5, 4.0), 4.0, 20.0)
            terminal_wacc = st.slider("Mature WACC (%)", min_value=4.0, max_value=20.0, value=terminal_wacc_default, step=0.25, key="dcf-terminal-wacc") / 100
            terminal_growth = st.slider("Terminal growth (%)", min_value=0.0, max_value=5.0, value=2.5, step=0.25, key="dcf-terminal-growth") / 100

        with st.expander("Advanced Cost of Capital"):
            render_compact_metric_grid(
                "Auto WACC Inputs",
                [
                    ("Sector", format_text_value(wacc_inputs.get("sector"))),
                    ("Industry", format_text_value(wacc_inputs.get("industry"))),
                    ("Beta", f"{safe_numeric(wacc_inputs.get('beta')):.2f}" if safe_numeric(wacc_inputs.get("beta")) is not None else "N/A"),
                    ("Risk-Free Rate", format_dcf_percent(wacc_inputs.get("risk_free_rate"))),
                    ("Equity Risk Premium", format_dcf_percent(wacc_inputs.get("equity_risk_premium"))),
                    ("Cost of Equity", format_dcf_percent(wacc_inputs.get("cost_of_equity"))),
                    ("Pre-Tax Cost of Debt", format_dcf_percent(wacc_inputs.get("pre_tax_cost_of_debt"))),
                    ("Tax Rate", format_dcf_percent(wacc_inputs.get("tax_rate"))),
                    ("Debt Weight", format_dcf_percent(wacc_inputs.get("debt_weight"))),
                    ("Equity Weight", format_dcf_percent(wacc_inputs.get("equity_weight"))),
                    ("Estimated WACC", format_dcf_percent(wacc_inputs.get("wacc"))),
                    ("Source", format_text_value(wacc_inputs.get("source_note"))),
                ],
                columns=4,
            )

        with st.expander("Equity Bridge Inputs"):
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                cash_billions = st.number_input("Cash ($B)", min_value=0.0, value=max(0.0, cash_default / 1_000_000_000), step=0.5, format="%.2f", key="dcf-cash-b")
            with col2:
                debt_billions = st.number_input("Debt ($B)", min_value=0.0, value=max(0.0, debt_default / 1_000_000_000), step=0.5, format="%.2f", key="dcf-debt-b")
            with col3:
                shares_millions = st.number_input("Diluted shares (M)", min_value=0.0, value=max(0.0, shares_default / 1_000_000), step=10.0, format="%.1f", key="dcf-shares-m")
            with col4:
                minority_billions = st.number_input("Minority interest ($B)", min_value=0.0, value=max(0.0, minority_default / 1_000_000_000), step=0.1, format="%.2f", key="dcf-minority-b")
            with col5:
                investments_billions = st.number_input("Non-operating investments ($B)", min_value=0.0, value=max(0.0, investments_default / 1_000_000_000), step=0.1, format="%.2f", key="dcf-investments-b")

    terminal_roic_default = max(terminal_wacc + 0.01, terminal_growth + 0.01)
    col1, col2, col3 = st.columns(3)
    with col1:
        terminal_roic = st.slider(
            "Terminal ROIC (%)",
            min_value=2.0,
            max_value=30.0,
            value=clamp_slider_default(terminal_roic_default * 100, 2.0, 30.0),
            step=0.5,
            key="dcf-terminal-roic",
        ) / 100
    with col2:
        terminal_reinvestment_rate = terminal_growth / terminal_roic if terminal_roic else None
        st.metric("Terminal reinvestment rate", format_percent_ratio(terminal_reinvestment_rate))
    with col3:
        st.metric("Reference FCF", format_compact_money(free_cash_flow))

    assumptions = DcfAssumptions(
        starting_revenue=revenue_billions * 1_000_000_000,
        current_operating_margin=current_margin,
        tax_rate=tax_rate,
        cash=cash_billions * 1_000_000_000,
        debt=debt_billions * 1_000_000_000,
        diluted_shares=shares_millions * 1_000_000,
        minority_interest=minority_billions * 1_000_000_000,
        investments=investments_billions * 1_000_000_000,
        high_growth_rate=high_growth,
        fade_growth_rate=fade_growth,
        high_growth_years=high_growth_years,
        target_operating_margin=target_margin,
        years_to_target_margin=years_to_margin,
        projection_years=projection_years,
        sales_to_capital_ratio=sales_to_capital,
        terminal_sales_to_capital_ratio=terminal_sales_to_capital,
        wacc=wacc,
        terminal_cost_of_capital=terminal_wacc,
        terminal_growth_rate=terminal_growth,
        terminal_operating_margin=terminal_margin,
        terminal_roic=terminal_roic,
        current_price=current_price,
    )
    result = calculate_damodaran_dcf(assumptions)

    warnings = result.get("warnings", [])
    if not result.get("ok"):
        for warning in warnings:
            st.warning(warning)
        st.info("DCF unavailable with the current assumptions. Fix the warnings above to calculate intrinsic value.")
        return

    render_compact_metric_grid(
        "Base Case Intrinsic Value",
        [
            ("Fair Value / Share", format_price(result.get("fair_value_per_share"))),
            ("Current Price", format_price(result.get("current_price"))),
            ("Upside / Downside", format_signed_percent_optional(result.get("upside_downside"))),
            ("Margin of Safety Price", format_price(result.get("margin_of_safety_price"))),
            ("Enterprise Value", format_compact_money(result.get("enterprise_value"))),
            ("Equity Value", format_compact_money(result.get("equity_value"))),
        ],
        columns=3,
    )
    st.info(build_dcf_summary(result, assumptions))

    if ebit_default is not None:
        st.caption(f"Starting EBIT reference: {format_compact_money(ebit_default)}. The model forecasts EBIT from revenue and operating margin.")

    render_compact_metric_grid(
        "Cost of Capital",
        [
            ("Starting WACC", f"{assumptions.wacc * 100:.2f}%"),
            ("Mature WACC", f"{assumptions.terminal_cost_of_capital * 100:.2f}%"),
            ("Risk-Free Rate", format_dcf_percent(wacc_inputs.get("risk_free_rate"))),
            ("Equity Risk Premium", format_dcf_percent(wacc_inputs.get("equity_risk_premium"))),
            ("Beta", f"{safe_numeric(wacc_inputs.get('beta')):.2f}" if safe_numeric(wacc_inputs.get("beta")) is not None else "N/A"),
            ("Pre-Tax Cost of Debt", format_dcf_percent(wacc_inputs.get("pre_tax_cost_of_debt"))),
            ("Debt-to-Capital", format_dcf_percent(wacc_inputs.get("debt_weight"))),
        ],
        columns=4,
    )
    render_compact_metric_grid(
        "Reinvestment",
        [
            ("Starting Sales-to-Capital", f"{assumptions.sales_to_capital_ratio:.2f}x"),
            ("Mature Sales-to-Capital", f"{assumptions.terminal_sales_to_capital_ratio:.2f}x"),
            ("Terminal ROIC", f"{assumptions.terminal_roic * 100:.1f}%"),
            ("Terminal Excess Return", f"{(assumptions.terminal_roic - assumptions.terminal_cost_of_capital) * 100:.1f}%"),
            ("Terminal Reinvestment Rate", format_percent_ratio(terminal_reinvestment_rate)),
            ("Reference FCF", format_compact_money(free_cash_flow)),
        ],
        columns=3,
    )

    forecast_rows = result.get("forecast", [])
    operating_rows = [
        {
            "Year": row["Year"],
            "Revenue": row["Revenue"],
            "Revenue Growth": row["Revenue Growth"],
            "Operating Margin": row["Operating Margin"],
            "EBIT": row["EBIT"],
            "NOPAT": row["NOPAT"],
            "Cost of Capital": row["Cost of Capital"],
        }
        for row in forecast_rows
    ]
    fcff_rows = [
        {
            "Year": row["Year"],
            "Reinvestment": row["Reinvestment"],
            "Reinvestment Rate": row["Reinvestment Rate"],
            "FCFF": row["FCFF"],
            "Sales-to-Capital": row["Sales-to-Capital"],
            "ROIC": row["ROIC"],
            "Discount Factor": row["Discount Factor"],
            "PV FCFF": row["PV FCFF"],
        }
        for row in forecast_rows
    ]

    render_dcf_result_table(
        "Operating Forecast",
        operating_rows,
        {
            "Revenue": "money",
            "Revenue Growth": "percent",
            "Operating Margin": "percent",
            "EBIT": "money",
            "NOPAT": "money",
            "Cost of Capital": "percent",
        },
    )
    render_dcf_result_table(
        "Reinvestment / FCFF Forecast",
        fcff_rows,
        {
            "Reinvestment": "money",
            "Reinvestment Rate": "percent",
            "FCFF": "money",
            "Sales-to-Capital": "multiple",
            "ROIC": "percent",
            "Discount Factor": "number",
            "PV FCFF": "money",
        },
    )

    terminal = result.get("terminal", {})
    render_compact_metric_grid(
        "Terminal Value",
        [
            ("Terminal Revenue", format_compact_money(terminal.get("terminal_revenue"))),
            ("Terminal Margin", format_percent_ratio(assumptions.terminal_operating_margin)),
            ("Terminal NOPAT", format_compact_money(terminal.get("terminal_nopat"))),
            ("Terminal Reinvestment", format_compact_money(terminal.get("terminal_reinvestment"))),
            ("Terminal FCFF", format_compact_money(terminal.get("terminal_fcff"))),
            ("PV Terminal Value", format_compact_money(result.get("pv_terminal_value"))),
        ],
        columns=3,
    )
    render_compact_metric_grid(
        "Equity Bridge",
        [
            ("Enterprise Value", format_compact_money(result.get("enterprise_value"))),
            ("Cash", format_compact_money(assumptions.cash)),
            ("Investments", format_compact_money(assumptions.investments)),
            ("Debt", format_compact_money(assumptions.debt)),
            ("Minority Interest", format_compact_money(assumptions.minority_interest)),
            ("Equity Value", format_compact_money(result.get("equity_value"))),
        ],
        columns=3,
    )

    render_dcf_result_table(
        "Bear / Base / Bull Scenarios",
        calculate_scenarios(assumptions),
        {
            "Revenue CAGR": "percent",
            "Target Operating Margin": "percent",
            "WACC": "percent",
            "Terminal WACC": "percent",
            "Terminal Growth": "percent",
            "Sales-to-Capital": "multiple",
            "Mature Sales-to-Capital": "multiple",
            "Fair Value / Share": "price",
            "Current Price": "price",
            "Upside / Downside": "signed_percent",
            "MOS Buy Price": "price",
            "Enterprise Value": "money",
            "Equity Value": "money",
        },
    )

    sensitivity_rows = build_sensitivity_table(assumptions)
    render_dcf_result_table(
        "Sensitivity: Fair Value / Share",
        sensitivity_rows,
        {column: "price" for row in sensitivity_rows for column in row if column != "WACC"} | {"WACC": "percent"},
    )

    if warnings:
        st.markdown("**Key Warnings**")
        for warning in warnings:
            st.warning(warning)


def render_sec_filing_tab(latest_filing: dict) -> None:
    if not latest_filing:
        st.info("No recent 10-Q or 10-K filing found.")
        return

    render_compact_metric_grid(
        "Latest SEC Filing",
        [
            ("Form", latest_filing.get("form") or "N/A"),
            ("Filing Date", latest_filing.get("filing_date") or latest_filing.get("date") or "N/A"),
            ("Report Date", latest_filing.get("report_date") or "N/A"),
            ("Accession", latest_filing.get("accession_number") or "N/A"),
        ],
        columns=2,
    )
    filing_url = latest_filing.get("url")
    if filing_url:
        st.markdown(f'<a href="{escape(filing_url)}" target="_blank" rel="noopener noreferrer">Open SEC filing</a>', unsafe_allow_html=True)
    else:
        st.info("SEC filing link unavailable.")


def render_technical_readout(chart_history, ticker: str | None = None) -> None:
    readout = build_technical_readout(chart_history, ticker=ticker)
    bullets = [
        readout.get("trend_summary"),
        readout.get("ma_summary"),
        readout.get("momentum_summary"),
        readout.get("volume_summary"),
        readout.get("support_resistance_summary"),
        readout.get("swing_read"),
    ]
    bullet_html = "".join(f"<li>{escape(str(item))}</li>" for item in bullets if item)
    st.markdown(
        (
            "<div class='technical-readout'>"
            "<div class='technical-readout-title'>1-Month Technical Readout</div>"
            f"<div class='technical-readout-status'>Status: {escape(str(readout.get('status') or 'N/A'))}</div>"
            f"<ul>{bullet_html}</ul>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_company_snapshot(company_profile: dict | None, fundamentals: dict | None = None) -> None:
    profile = company_profile or {}
    fallback = fundamentals or {}
    summary = profile.get("longBusinessSummary") or "Company description unavailable."
    sector = profile.get("sector") or fallback.get("sector") or "N/A"
    industry = profile.get("industry") or fallback.get("industry") or "N/A"
    exchange = profile.get("exchange") or "N/A"
    market_cap = profile.get("marketCap") or fallback.get("marketCap")

    render_dashboard_section_header("Company Snapshot")
    st.markdown(
        f"<div class='trader-summary'>{escape(str(summary))}</div>",
        unsafe_allow_html=True,
    )
    render_compact_metric_grid(
        "",
        [
            ("Sector", sector),
            ("Industry", industry),
            ("Exchange", exchange),
            ("Market Cap", format_market_cap(market_cap)),
        ],
        columns=4,
    )


def get_thesis_tone_from_context(context: dict) -> str:
    long_status = str(context.get("long_status") or "")
    extension = str(context.get("extension_status") or "")
    if long_status in {"Strong Long Setup", "Pullback Entry Setup"} and extension != "Too Extended":
        return "positive"
    if long_status in {"Below Key MAs", "Weak / Avoid"}:
        return "negative"
    if extension in {"Extended", "Too Extended"}:
        return "warning"
    return "neutral"


def get_thesis_verdict(context: dict) -> dict:
    long_status = str(context.get("long_status") or "Setup unavailable")
    extension = str(context.get("extension_status") or "")
    pullback = str(context.get("pullback_quality") or "N/A")
    if long_status == "Strong Long Setup" and extension != "Too Extended":
        return {
            "label": "Constructive",
            "tone": "positive",
            "note": "Trend is aligned and price is holding above the core moving averages.",
        }
    if long_status == "Pullback Entry Setup":
        return {
            "label": "Constructive Pullback",
            "tone": "positive",
            "note": f"Entry quality depends on whether the pullback stays orderly: {pullback}.",
        }
    if long_status in {"Too Extended", "Long Watchlist"}:
        return {
            "label": "Wait For Setup",
            "tone": "warning",
            "note": "Structure is useful, but risk/reward needs either a reset or a stronger trigger.",
        }
    if long_status in {"Below Key MAs", "Weak / Avoid"}:
        return {
            "label": "Needs Repair",
            "tone": "negative",
            "note": "Price is not in a clean long-only structure yet.",
        }
    return {
        "label": "Watchlist",
        "tone": "neutral",
        "note": "Data is mixed or incomplete, so treat the thesis as context only.",
    }


def get_thesis_title(ticker: str, context: dict) -> str:
    long_status = str(context.get("long_status") or "Setup unavailable")
    if long_status == "Strong Long Setup":
        return f"{ticker} trend structure remains intact above key moving averages"
    if long_status == "Pullback Entry Setup":
        return f"{ticker} constructive pullback setup with moving-average support"
    if long_status in {"Too Extended", "Long Watchlist"}:
        return f"{ticker} watchlist setup needs cleaner risk/reward before entry"
    if long_status in {"Below Key MAs", "Weak / Avoid"}:
        return f"{ticker} technical repair needed before this is a clean long"
    return f"{ticker} scanner thesis based on trend, fundamentals, and risk"


def get_thesis_summary(
    ticker: str,
    detail_row: dict,
    fundamentals: dict,
    context: dict,
) -> str:
    parts = [
        f"{ticker} trades at {format_price(detail_row.get('price'))}, {format_signed_percent(context.get('distance_ema21'))} versus EMA21",
        f"the scanner classifies it as {context.get('long_status', 'N/A')} with {context.get('ema_stack_status', 'N/A')} structure",
    ]
    sector = fundamentals.get("sector")
    industry = fundamentals.get("industry")
    if sector or industry:
        parts.append(f"{format_text_value(sector)} / {format_text_value(industry)} context")
    return ". ".join(parts) + "."


def build_thesis_stat_html(label: str, value: object) -> str:
    return (
        "<div class='thesis-stat'>"
        f"<div class='thesis-stat-label'>{escape(label)}</div>"
        f"<div class='thesis-stat-value'>{escape(str(value))}</div>"
        "</div>"
    )


def build_thesis_chip_html(label: str, value: object) -> str:
    return f"<span class='thesis-evidence-chip'>{escape(label)} {escape(str(value))}</span>"


def build_thesis_section_html(
    label: str,
    title: str,
    text: str,
    evidence: list[tuple[str, object]],
    tone: str = "neutral",
    full: bool = False,
) -> str:
    evidence_html = "".join(build_thesis_chip_html(chip_label, chip_value) for chip_label, chip_value in evidence)
    full_class = " thesis-section-full" if full else ""
    return (
        f"<section class='thesis-section thesis-section-{escape(tone)}{full_class}'>"
        "<div class='thesis-section-header'>"
        f"<span class='thesis-pill thesis-pill-{escape(tone)}'>{escape(label)}</span>"
        f"<div class='thesis-section-title'>{escape(title)}</div>"
        "</div>"
        f"<div class='thesis-section-text'>{escape(text)}</div>"
        f"<div class='thesis-evidence-row'>{evidence_html}</div>"
        "</section>"
    )


def build_technical_thesis_section(detail_row: dict, chart_history: object, context: dict) -> dict:
    technical_readout = build_technical_readout(chart_history, ticker=detail_row.get("ticker")) if chart_history is not None else {}
    text = technical_readout.get("ma_summary") or build_trader_summary(detail_row, chart_history=chart_history)
    momentum = technical_readout.get("momentum_summary")
    support = technical_readout.get("support_resistance_summary")
    if momentum:
        text = f"{text} {momentum}"
    if support:
        text = f"{text} {support}"
    return {
        "label": "technical",
        "title": f"5. Technical base intact - {context.get('pullback_quality', 'N/A')}",
        "text": text,
        "evidence": [
            ("Price", format_price(detail_row.get("price"))),
            ("EMA8", format_price(context.get("ema8"))),
            ("EMA21", format_price(context.get("ema21"))),
            ("SMA50", format_price(context.get("sma50"))),
            ("SMA200", format_price(context.get("sma200"))),
            ("vs EMA21", format_signed_percent(context.get("distance_ema21"))),
            ("Rel Vol", format_optional_number(detail_row.get("volume_ratio"))),
        ],
        "tone": get_thesis_tone_from_context(context),
    }


def build_earnings_thesis_section(fundamentals: dict, earnings_info: dict) -> dict:
    earnings_growth = safe_numeric(fundamentals.get("earningsGrowth"))
    quarterly_growth = safe_numeric(fundamentals.get("earningsQuarterlyGrowth"))
    revenue_growth = safe_numeric(fundamentals.get("revenueGrowth"))
    operating_margin = safe_numeric(fundamentals.get("operatingMargins"))
    profit_margin = safe_numeric(fundamentals.get("profitMargins"))
    next_earnings = (earnings_info or {}).get("next_earnings_date")
    last_earnings = (earnings_info or {}).get("last_earnings_date")
    days_until = safe_numeric((earnings_info or {}).get("days_until_earnings"))

    positives = []
    negatives = []
    if earnings_growth is not None:
        (positives if earnings_growth >= 0.08 else negatives if earnings_growth < 0 else positives).append(
            f"earnings growth {format_percent_optional(earnings_growth)}"
        )
    if quarterly_growth is not None:
        (positives if quarterly_growth >= 0 else negatives).append(
            f"quarterly earnings growth {format_percent_optional(quarterly_growth)}"
        )
    if revenue_growth is not None:
        (positives if revenue_growth >= 0.05 else negatives if revenue_growth < 0 else positives).append(
            f"revenue growth {format_percent_optional(revenue_growth)}"
        )
    if operating_margin is not None:
        (positives if operating_margin >= 0.10 else negatives if operating_margin < 0 else positives).append(
            f"operating margin {format_percent_ratio(operating_margin)}"
        )

    if negatives and not positives:
        headline = "earnings pressure"
        tone = "negative"
        text = f"Earnings read is cautious because {', '.join(negatives[:3])}. The scanner wants proof that revenue, EPS, and margins are stabilizing before giving the setup full fundamental support."
    elif negatives:
        headline = "mixed earnings quality"
        tone = "warning"
        text = f"Earnings read is mixed: positives include {', '.join(positives[:3])}, while risks include {', '.join(negatives[:3])}. Watch the next report for whether growth and margins move in the same direction."
    elif positives:
        headline = "earnings confirming"
        tone = "positive"
        text = f"Earnings read is constructive with {', '.join(positives[:4])}. That supports the technical setup as long as the next report does not show deceleration."
    else:
        headline = "earnings data limited"
        tone = "neutral"
        text = "Earnings read is limited because growth and margin fields are not available from the current feed. Treat the catalyst as incomplete until fresh reported results are reviewed."

    if days_until is not None and days_until <= 21:
        text = f"{text} Next earnings are close, so position risk should account for the report window."

    return {
        "label": "earnings",
        "title": f"1. Earnings read - {headline}",
        "text": text,
        "evidence": [
            ("EPS Growth", format_percent_optional(earnings_growth)),
            ("Q EPS", format_percent_optional(quarterly_growth)),
            ("Revenue Growth", format_percent_optional(revenue_growth)),
            ("Op Margin", format_percent_ratio(operating_margin)),
            ("Profit Margin", format_percent_ratio(profit_margin)),
            ("Last Report", last_earnings or "N/A"),
            ("Next Report", next_earnings or "N/A"),
        ],
        "tone": tone,
    }


def get_sector_driver_text(sector: object, industry: object) -> str:
    sector_text = str(sector or "").strip()
    industry_text = str(industry or "").strip()
    combined = f"{sector_text} {industry_text}".lower()
    if "semiconductor" in combined or "technology" in combined or "software" in combined:
        return "Big sector driver is AI infrastructure spending, cloud budgets, chip supply, and enterprise software demand."
    if "communication" in combined or "internet content" in combined or "entertainment" in combined:
        return "Big sector driver is digital advertising demand, streaming economics, AI search disruption, and subscriber growth."
    if "consumer cyclical" in combined or "retail" in combined or "auto" in combined:
        return "Big sector driver is consumer spending, pricing power, ad/commerce demand, rates, and margin pressure from fulfillment or inventory costs."
    if "financial" in combined or "bank" in combined:
        return "Big sector driver is interest-rate path, credit quality, loan demand, deposit costs, and capital-market activity."
    if "healthcare" in combined or "biotech" in combined or "pharma" in combined:
        return "Big sector driver is trial data, regulatory approvals, reimbursement risk, and procedure or prescription demand."
    if "energy" in combined or "oil" in combined or "gas" in combined:
        return "Big sector driver is commodity pricing, production discipline, service costs, and cash returns to shareholders."
    if "industrial" in combined or "aerospace" in combined:
        return "Big sector driver is order backlog, capex cycles, reshoring, defense demand, and input-cost pressure."
    if "real estate" in combined or "reit" in combined:
        return "Big sector driver is rates, occupancy, refinancing costs, cap rates, and property-level cash flow."
    if "utilities" in combined:
        return "Big sector driver is power demand, rate-case outcomes, grid capex, and financing costs."
    if "materials" in combined:
        return "Big sector driver is commodity demand, China/global manufacturing activity, pricing, and input costs."
    if "consumer defensive" in combined or "staples" in combined:
        return "Big sector driver is volume growth, pricing power, freight/input costs, and consumer trade-down behavior."
    return "Big sector driver is the current industry cycle, demand trend, margin pressure, and whether market leadership is rotating into this group."


def build_sector_thesis_section(fundamentals: dict, company_profile: dict | None) -> dict:
    profile = company_profile or {}
    sector = fundamentals.get("sector") or profile.get("sector")
    industry = fundamentals.get("industry") or profile.get("industry")
    summary = profile.get("longBusinessSummary") or ""
    sector_driver = get_sector_driver_text(sector, industry)
    summary_sentence = ""
    if summary:
        summary_sentence = str(summary).split(". ")[0].strip()
        if summary_sentence and not summary_sentence.endswith("."):
            summary_sentence = f"{summary_sentence}."
    text = (
        f"{sector_driver} {summary_sentence} "
        "This section is sector-context from company profile data, not a live headline feed yet."
    ).strip()
    tone = "positive" if sector or industry else "neutral"
    return {
        "label": "sector",
        "title": f"2. Sector context - {format_text_value(sector)} / {format_text_value(industry)}",
        "text": text,
        "evidence": [
            ("Sector", format_text_value(sector)),
            ("Industry", format_text_value(industry)),
            ("Market Cap", format_market_cap(fundamentals.get("marketCap"))),
            ("Beta", format_optional_number(fundamentals.get("beta"))),
            ("Inst Own", format_percent_ratio(fundamentals.get("heldPercentInstitutions"))),
        ],
        "tone": tone,
    }


def build_future_guidance_thesis_section(fundamentals: dict, earnings_info: dict, forward_profile: dict | None = None) -> dict:
    forward = forward_profile or {}
    current_year = safe_numeric(forward.get("eps_current_year_growth"))
    if current_year is None:
        current_year = safe_numeric(fundamentals.get("earningsGrowthEstimateCurrentYear"))
    next_year = safe_numeric(forward.get("eps_next_year_growth"))
    if next_year is None:
        next_year = safe_numeric(fundamentals.get("earningsGrowthEstimateNextYear"))
    next_five_years = safe_numeric(fundamentals.get("earningsGrowthEstimateNext5Years"))
    revenue_growth = safe_numeric(forward.get("revenue_next_year_growth"))
    if revenue_growth is None:
        revenue_growth = safe_numeric(fundamentals.get("revenueGrowth"))
    forward_pe = safe_numeric(fundamentals.get("forwardPE"))
    next_earnings = (earnings_info or {}).get("next_earnings_date")
    eps_next_year = safe_numeric(forward.get("eps_next_year_avg"))
    revenue_next_year = safe_numeric(forward.get("revenue_next_year_avg"))
    next_year_analysts = safe_numeric(forward.get("eps_next_year_analysts"))

    estimate_values = [value for value in (current_year, next_year, next_five_years) if value is not None]
    if estimate_values and max(estimate_values) >= 0.10:
        tone = "positive"
        headline = "future estimates improving"
        text = (
            f"Forward guidance read is constructive: current-year EPS estimate is {format_percent_optional(current_year)}, "
            f"next-year estimate is {format_percent_optional(next_year)}, and long-term estimate is {format_percent_optional(next_five_years)}. "
            "The scanner wants these estimates to stay positive after the next report."
        )
        if eps_next_year is not None or revenue_next_year is not None:
            text = (
                f"{text} Analysts model next-year EPS near {format_optional_number(eps_next_year)} "
                f"and revenue near {format_compact_money(revenue_next_year)}."
            )
    elif estimate_values and min(estimate_values) < 0:
        tone = "negative"
        headline = "future estimates under pressure"
        text = (
            f"Forward guidance read is cautious: current-year EPS estimate is {format_percent_optional(current_year)} and "
            f"next-year estimate is {format_percent_optional(next_year)}. Negative estimate revisions would weaken the fundamental case."
        )
    elif estimate_values:
        tone = "warning"
        headline = "future estimates mixed"
        text = (
            f"Forward guidance read is mixed with current-year EPS estimate {format_percent_optional(current_year)}, "
            f"next-year estimate {format_percent_optional(next_year)}, and revenue growth {format_percent_optional(revenue_growth)}. "
            "The stock needs clearer acceleration to support a higher multiple."
        )
    else:
        tone = "neutral"
        headline = "future guidance unavailable"
        text = (
            "Forward earnings estimate fields are not available from the current feed. "
            "Use the next earnings date and reported company outlook before relying on this catalyst."
        )

    if next_earnings:
        text = f"{text} Next earnings checkpoint: {next_earnings}."

    return {
        "label": "guidance",
        "title": f"3. Future earnings guidance - {headline}",
        "text": text,
        "evidence": [
            ("Curr Yr EPS", format_percent_optional(current_year)),
            ("Next Yr EPS", format_percent_optional(next_year)),
            ("5Y EPS", format_percent_optional(next_five_years)),
            ("Next Yr Rev", format_percent_optional(revenue_growth)),
            ("Next Yr EPS $", format_optional_number(eps_next_year)),
            ("Next Yr Rev $", format_compact_money(revenue_next_year)),
            ("Analysts", format_optional_number(next_year_analysts, decimals=0)),
            ("Forward P/E", format_multiple(forward_pe)),
            ("Next Report", next_earnings or "N/A"),
        ],
        "tone": tone,
    }


def format_recommendation_label(value: object) -> str:
    text_value = str(value or "").strip().replace("_", " ").replace("-", " ")
    if not text_value:
        return "N/A"
    return " ".join(part.capitalize() for part in text_value.split())


def get_analyst_recommendation_tone(recommendation_key: object, recommendation_mean: object) -> str:
    key = str(recommendation_key or "").lower()
    mean = safe_numeric(recommendation_mean)
    if "sell" in key or (mean is not None and mean >= 3.6):
        return "negative"
    if "hold" in key or (mean is not None and 2.6 <= mean < 3.6):
        return "warning"
    if "buy" in key or (mean is not None and mean < 2.6):
        return "positive"
    return "neutral"


def build_analyst_thesis_section(detail_row: dict, fundamentals: dict) -> dict:
    recommendation_key = fundamentals.get("recommendationKey")
    recommendation_mean = safe_numeric(fundamentals.get("recommendationMean"))
    target_mean = safe_numeric(fundamentals.get("targetMeanPrice"))
    target_median = safe_numeric(fundamentals.get("targetMedianPrice"))
    price = safe_numeric(detail_row.get("price"))
    upside = calculate_percent_change(target_mean, price) if target_mean is not None else None
    recommendation = format_recommendation_label(recommendation_key)
    tone = get_analyst_recommendation_tone(recommendation_key, recommendation_mean)

    if recommendation != "N/A" and upside is not None:
        text = (
            f"Analyst view is {recommendation} with mean target {format_price(target_mean)}, "
            f"or {format_signed_percent(upside)} versus current price. "
            "Treat this as consensus context, not a trade signal by itself."
        )
    elif recommendation != "N/A":
        text = f"Analyst view is {recommendation}, but target-price upside is unavailable from the current feed."
    elif target_mean is not None and upside is not None:
        text = (
            f"Analyst target context shows mean target {format_price(target_mean)}, "
            f"or {format_signed_percent(upside)} versus current price, but recommendation label is unavailable."
        )
    else:
        text = "Analyst recommendation and target data are unavailable from the current feed."

    return {
        "label": "guidance",
        "title": f"4. Analyst guidance - {recommendation}",
        "text": text,
        "evidence": [
            ("Rating", recommendation),
            ("Rating Mean", format_optional_number(recommendation_mean)),
            ("Mean Target", format_price(target_mean)),
            ("Median Target", format_price(target_median)),
            ("Upside", format_signed_percent(upside)),
        ],
        "tone": tone,
    }


def build_fundamental_thesis_section(fundamentals: dict, company_profile: dict | None) -> dict:
    profile = company_profile or {}
    revenue_growth = safe_numeric(fundamentals.get("revenueGrowth"))
    earnings_growth = safe_numeric(fundamentals.get("earningsGrowth"))
    operating_margin = safe_numeric(fundamentals.get("operatingMargins"))
    forward_pe = fundamentals.get("forwardPE")
    sector = fundamentals.get("sector") or profile.get("sector")
    industry = fundamentals.get("industry") or profile.get("industry")
    quality_parts = []
    if revenue_growth is not None:
        quality_parts.append(f"revenue growth {format_percent_optional(revenue_growth)}")
    if earnings_growth is not None:
        quality_parts.append(f"earnings growth {format_percent_optional(earnings_growth)}")
    if operating_margin is not None:
        quality_parts.append(f"operating margin {format_percent_ratio(operating_margin)}")
    if forward_pe is not None:
        quality_parts.append(f"forward P/E {format_multiple(forward_pe)}")
    if not quality_parts:
        quality_parts.append("fundamental data is limited")
    text = (
        f"{format_text_value(sector)} / {format_text_value(industry)} profile with "
        f"{', '.join(quality_parts)}."
    )
    tone = "positive" if any(value is not None and value > 0 for value in (revenue_growth, earnings_growth, operating_margin)) else "neutral"
    return {
        "label": "fundamentals",
        "title": "2. Business quality and valuation context",
        "text": text,
        "evidence": [
            ("Market Cap", format_market_cap(fundamentals.get("marketCap"))),
            ("Revenue", format_percent_optional(revenue_growth)),
            ("Earnings", format_percent_optional(earnings_growth)),
            ("Op Margin", format_percent_ratio(operating_margin)),
            ("Forward P/E", format_multiple(forward_pe)),
            ("52W", f"{format_price(fundamentals.get('fiftyTwoWeekLow'))} - {format_price(fundamentals.get('fiftyTwoWeekHigh'))}"),
        ],
        "tone": tone,
    }


def build_growth_thesis_section(fundamentals: dict) -> dict:
    revenue_growth = safe_numeric(fundamentals.get("revenueGrowth"))
    revenue_quarterly_growth = safe_numeric(fundamentals.get("revenueQuarterlyGrowth"))
    earnings_growth = safe_numeric(fundamentals.get("earningsGrowth"))
    earnings_quarterly_growth = safe_numeric(fundamentals.get("earningsQuarterlyGrowth"))
    current_year_estimate = safe_numeric(fundamentals.get("earningsGrowthEstimateCurrentYear"))
    next_year_estimate = safe_numeric(fundamentals.get("earningsGrowthEstimateNextYear"))

    growth_points = []
    if revenue_growth is not None:
        growth_points.append(f"sales growth {format_percent_optional(revenue_growth)}")
    if revenue_quarterly_growth is not None:
        growth_points.append(f"quarterly sales growth {format_percent_optional(revenue_quarterly_growth)}")
    if earnings_growth is not None:
        growth_points.append(f"earnings growth {format_percent_optional(earnings_growth)}")
    if next_year_estimate is not None:
        growth_points.append(f"next-year earnings estimate {format_percent_optional(next_year_estimate)}")

    if growth_points:
        text = (
            f"Growth checkpoint shows {', '.join(growth_points)}. "
            "The setup gains quality when revenue and earnings are both expanding; it weakens if sales slow while valuation remains elevated."
        )
    else:
        text = (
            "Growth data is limited from the current Yahoo Finance feed. "
            "Treat the technical setup as less confirmed until revenue and earnings trend data is available."
        )

    positive_growth = any(value is not None and value >= 0.08 for value in (revenue_growth, revenue_quarterly_growth, earnings_growth, earnings_quarterly_growth))
    negative_growth = any(value is not None and value < 0 for value in (revenue_growth, revenue_quarterly_growth, earnings_growth, earnings_quarterly_growth))
    tone = "negative" if negative_growth else ("positive" if positive_growth else "warning")

    return {
        "label": "growth",
        "title": "3. Sales and earnings growth checkpoint",
        "text": text,
        "evidence": [
            ("Revenue", format_compact_money(fundamentals.get("totalRevenue"))),
            ("Sales Growth", format_percent_optional(revenue_growth)),
            ("Q Sales", format_percent_optional(revenue_quarterly_growth)),
            ("Earnings", format_percent_optional(earnings_growth)),
            ("Q Earnings", format_percent_optional(earnings_quarterly_growth)),
            ("Next Yr Est", format_percent_optional(next_year_estimate)),
            ("Curr Yr Est", format_percent_optional(current_year_estimate)),
        ],
        "tone": tone,
    }


def build_profitability_thesis_section(fundamentals: dict) -> dict:
    gross_margin = safe_numeric(fundamentals.get("grossMargins"))
    operating_margin = safe_numeric(fundamentals.get("operatingMargins"))
    profit_margin = safe_numeric(fundamentals.get("profitMargins"))
    return_on_equity = safe_numeric(fundamentals.get("returnOnEquity"))
    free_cashflow = safe_numeric(fundamentals.get("freeCashflow"))
    operating_cashflow = safe_numeric(fundamentals.get("operatingCashflow"))
    debt_to_equity = normalize_debt_to_equity(fundamentals.get("debtToEquity"))

    quality_points = []
    if operating_margin is not None:
        quality_points.append(f"operating margin {format_percent_ratio(operating_margin)}")
    if profit_margin is not None:
        quality_points.append(f"profit margin {format_percent_ratio(profit_margin)}")
    if free_cashflow is not None:
        quality_points.append(f"free cash flow {format_compact_money(free_cashflow, signed=True)}")
    if debt_to_equity is not None:
        quality_points.append(f"debt/equity {debt_to_equity:.2f}x")

    if quality_points:
        text = (
            f"Profit-quality checkpoint shows {', '.join(quality_points)}. "
            "The best scanner setups pair price strength with durable margins, positive cash generation, and manageable leverage."
        )
    else:
        text = (
            "Profitability and cash-flow data is limited from the current feed. "
            "Keep the thesis technical until margin, cash-flow, and balance-sheet data can confirm it."
        )

    margin_negative = any(value is not None and value < 0 for value in (operating_margin, profit_margin, free_cashflow))
    quality_confirmed = (
        (operating_margin is not None and operating_margin >= 0.10)
        and (profit_margin is None or profit_margin >= 0)
        and (free_cashflow is None or free_cashflow > 0)
        and (debt_to_equity is None or debt_to_equity <= 1.5)
    )
    tone = "negative" if margin_negative else ("positive" if quality_confirmed else "warning")

    return {
        "label": "quality",
        "title": "4. Profit and cash-flow quality",
        "text": text,
        "evidence": [
            ("Gross Margin", format_percent_ratio(gross_margin)),
            ("Op Margin", format_percent_ratio(operating_margin)),
            ("Profit Margin", format_percent_ratio(profit_margin)),
            ("ROE", format_percent_ratio(return_on_equity)),
            ("FCF", format_compact_money(free_cashflow, signed=True)),
            ("Op Cash", format_compact_money(operating_cashflow, signed=True)),
            ("Debt/Eq", format_debt_to_equity(fundamentals.get("debtToEquity"))),
        ],
        "tone": tone,
    }


def build_filing_thesis_section(latest_filing: dict) -> dict:
    if not latest_filing:
        return {
            "label": "filing",
            "title": "SEC filing context",
            "text": "No recent 10-Q or 10-K filing metadata is available from the SEC helper.",
            "evidence": [("Status", "Unavailable")],
            "tone": "neutral",
        }
    form = latest_filing.get("form", "Filing")
    filing_date = latest_filing.get("filing_date") or latest_filing.get("date") or "N/A"
    report_date = latest_filing.get("report_date") or "N/A"
    return {
        "label": "filing",
        "title": "5. Latest filing checkpoint",
        "text": f"Latest SEC context is {form} filed {filing_date}. Use it as background before relying on the technical thesis.",
        "evidence": [
            ("Form", form),
            ("Filed", filing_date),
            ("Report", report_date),
            ("Accession", latest_filing.get("accession_number") or "N/A"),
        ],
        "tone": "neutral",
    }


def build_risk_factors(
    detail_row: dict,
    fundamentals: dict,
    earnings_info: dict,
    context: dict,
) -> list[str]:
    risks = []
    if context.get("extension_status") in {"Extended", "Too Extended"}:
        risks.append(f"Price is {context.get('extension_status')} versus EMA21 at {format_signed_percent(context.get('distance_ema21'))}.")
    if context.get("long_status") in {"Below Key MAs", "Weak / Avoid"}:
        risks.append("Price structure is below key moving averages, so the long setup needs repair.")
    volume_ratio = safe_numeric(detail_row.get("volume_ratio"))
    if volume_ratio is not None and volume_ratio < 0.8:
        risks.append(f"Relative volume is light at {format_optional_number(volume_ratio)}x.")
    short_float = safe_numeric(fundamentals.get("shortPercentOfFloat"))
    if short_float is not None and short_float >= 0.10:
        risks.append(f"Short interest is elevated at {format_percent_ratio(short_float)} of float.")
    debt_to_equity = normalize_debt_to_equity(fundamentals.get("debtToEquity"))
    if debt_to_equity is not None and debt_to_equity >= 1.5:
        risks.append(f"Leverage is high with debt/equity near {format_debt_to_equity(fundamentals.get('debtToEquity'))}.")
    days_until_earnings = safe_numeric((earnings_info or {}).get("days_until_earnings"))
    if days_until_earnings is not None and days_until_earnings <= 10:
        risks.append(f"Earnings are close: {int(days_until_earnings)} days until the next reported date.")
    if not risks:
        risks.append("No major scanner-level risk flag is active, but fresh news, earnings, and market regime still need review.")
    return risks[:6]


def build_confirmation_items(detail_row: dict, context: dict) -> list[str]:
    items = []
    if context.get("above_ema21"):
        items.append(f"Hold EMA21 near {format_price(context.get('ema21'))}; losing it weakens the pullback setup.")
    else:
        items.append(f"Reclaim EMA21 near {format_price(context.get('ema21'))} before treating this as a clean long setup.")
    if context.get("above_sma50"):
        items.append(f"Keep price above SMA50 near {format_price(context.get('sma50'))} to preserve intermediate support.")
    else:
        items.append(f"Reclaim SMA50 near {format_price(context.get('sma50'))} to repair the intermediate trend.")
    volume_ratio = safe_numeric(detail_row.get("volume_ratio"))
    if volume_ratio is None or volume_ratio < 1:
        items.append("Look for volume expansion on the next push higher; current confirmation is not volume-led.")
    else:
        items.append(f"Maintain participation near {format_optional_number(volume_ratio)}x relative volume on follow-through.")
    if context.get("extension_status") in {"Extended", "Too Extended"}:
        items.append("Avoid chasing while extended; wait for either a tight base or a reset toward short-term averages.")
    return items[:4]


def build_guidance_stat_html(label: str, value: object) -> str:
    return (
        "<div class='guidance-stat'>"
        f"<div class='guidance-stat-label'>{escape(str(label))}</div>"
        f"<div class='guidance-stat-value'>{escape(str(value))}</div>"
        "</div>"
    )


def build_guidance_row_html(label: str, text: str, status: str = "", tone: str = "neutral") -> str:
    pill_html = ""
    if status:
        pill_html = f"<span class='guidance-pill guidance-pill-{escape(tone)}'>{escape(status)}</span>"
    return (
        "<div class='guidance-row'>"
        f"<div class='guidance-row-label'>{escape(label)}</div>"
        f"<div class='guidance-row-value'>{pill_html}{escape(text)}</div>"
        "</div>"
    )


def classify_growth_guidance(value: object, strong_threshold: float = 0.15, good_threshold: float = 0.05) -> tuple[str, str]:
    numeric_value = safe_numeric(value)
    if numeric_value is None:
        return "Unavailable", "neutral"
    if numeric_value >= strong_threshold:
        return "Strong Increase", "positive"
    if numeric_value >= good_threshold:
        return "Increasing", "positive"
    if numeric_value >= 0:
        return "Slow Growth", "warning"
    return "Declining", "negative"


def classify_margin_guidance(operating_margin: object, profit_margin: object) -> tuple[str, str]:
    operating_value = safe_numeric(operating_margin)
    profit_value = safe_numeric(profit_margin)
    available_values = [value for value in (operating_value, profit_value) if value is not None]
    if not available_values:
        return "Unavailable", "neutral"
    if operating_value is not None and operating_value >= 0.15 and (profit_value is None or profit_value >= 0.08):
        return "High Quality", "positive"
    if all(value >= 0 for value in available_values):
        return "Profitable", "positive"
    if any(value < 0 for value in available_values):
        return "Margin Risk", "negative"
    return "Mixed", "warning"


def classify_cashflow_guidance(free_cashflow: object, operating_cashflow: object) -> tuple[str, str]:
    free_value = safe_numeric(free_cashflow)
    operating_value = safe_numeric(operating_cashflow)
    if free_value is None and operating_value is None:
        return "Unavailable", "neutral"
    if free_value is not None and free_value > 0:
        return "Cash Generative", "positive"
    if operating_value is not None and operating_value > 0:
        return "Operating Cash Positive", "positive"
    if free_value is not None and free_value < 0:
        return "Cash Burn", "negative"
    return "Mixed", "warning"


def classify_debt_guidance(value: object) -> tuple[str, str]:
    normalized_value = normalize_debt_to_equity(value)
    if normalized_value is None:
        return "Unavailable", "neutral"
    if normalized_value <= 0.6:
        return "Low Debt", "positive"
    if normalized_value <= 1.5:
        return "Moderate Debt", "warning"
    return "High Debt", "negative"


def classify_valuation_guidance(forward_pe: object, revenue_growth: object, earnings_growth: object) -> tuple[str, str]:
    pe_value = safe_numeric(forward_pe)
    growth_values = [safe_numeric(revenue_growth), safe_numeric(earnings_growth)]
    growth_values = [value for value in growth_values if value is not None]
    growth_floor = max(growth_values) if growth_values else None
    if pe_value is None:
        return "Unavailable", "neutral"
    if pe_value <= 0:
        return "Not Meaningful", "warning"
    if pe_value <= 25:
        return "Reasonable", "positive"
    if growth_floor is not None and growth_floor >= 0.15 and pe_value <= 45:
        return "Growth Premium", "positive"
    if pe_value >= 60:
        return "Expensive", "warning"
    return "Needs Growth", "warning"


def get_guidance_action_bias(context: dict) -> dict:
    status = context.get("long_status")
    if status == "Strong Long Setup":
        return {
            "label": "Constructive Buy Setup",
            "tone": "positive",
            "note": "Trend and moving-average structure support a long-side setup while risk levels hold.",
        }
    if status == "Pullback Entry Setup":
        return {
            "label": "Pullback Buy Setup",
            "tone": "positive",
            "note": "Entry quality is tied to the pullback staying orderly near EMA8 or EMA21.",
        }
    if status == "Too Extended":
        return {
            "label": "Hold / Wait For Reset",
            "tone": "warning",
            "note": "Trend is strong, but the scanner does not favor chasing far above EMA21.",
        }
    if status == "Long Watchlist":
        return {
            "label": "Watchlist - Needs Trigger",
            "tone": "warning",
            "note": "Structure is workable, but it still needs better confirmation before a cleaner entry.",
        }
    if status == "Below Key MAs":
        return {
            "label": "No Buy Setup",
            "tone": "negative",
            "note": "Price is below a key average, so the setup needs repair before long entries improve.",
        }
    if status == "Insufficient Data":
        return {
            "label": "Data Limited",
            "tone": "warning",
            "note": "There is not enough moving-average history to score the setup cleanly.",
        }
    return {
        "label": "Risk-Off / Avoid",
        "tone": "negative",
        "note": "The current scanner read does not support a clean long-side setup.",
    }


def format_guidance_zone(*levels: object) -> str:
    numeric_levels = sorted({round(level, 2) for level in (safe_numeric(value) for value in levels) if level is not None})
    if not numeric_levels:
        return "N/A"
    if len(numeric_levels) == 1:
        return format_price(numeric_levels[0])
    return f"{format_price(numeric_levels[0])} - {format_price(numeric_levels[-1])}"


def build_entry_guidance(context: dict) -> tuple[str, str, str]:
    status = context.get("long_status")
    extension_status = context.get("extension_status")
    ema8 = context.get("ema8")
    ema21 = context.get("ema21")
    sma50 = context.get("sma50")
    if status in {"Strong Long Setup", "Pullback Entry Setup"}:
        zone = format_guidance_zone(ema8, ema21)
        return "Entry Area", f"Best scanner entry area is the EMA8 to EMA21 zone near {zone}; avoid adding if extension stretches beyond healthy range.", "positive"
    if status == "Too Extended" or extension_status in {"Extended", "Too Extended"}:
        zone = format_guidance_zone(ema8, ema21)
        return "Reset Area", f"Wait for a base or pullback toward {zone} before the entry quality improves.", "warning"
    if status in {"Below Key MAs", "Weak / Avoid"}:
        reclaim = format_guidance_zone(context.get("ema21"), sma50)
        return "Repair Area", f"Needs a reclaim and hold above {reclaim} before the scanner treats it as a cleaner long setup.", "negative"
    return "Trigger Area", f"Watch for a close above EMA8/EMA21 with stronger participation; current setup is {format_text_value(status)}.", "warning"


def build_invalidation_guidance(context: dict) -> tuple[str, str, str]:
    ema21 = context.get("ema21")
    sma50 = context.get("sma50")
    if context.get("above_ema21") and context.get("above_sma50"):
        return "Invalidation", f"A close below EMA21 near {format_price(ema21)} weakens the setup; losing SMA50 near {format_price(sma50)} is the bigger sell-risk line.", "warning"
    if context.get("above_sma50"):
        return "Invalidation", f"Failure to reclaim EMA21 near {format_price(ema21)} keeps the setup tactical; losing SMA50 near {format_price(sma50)} turns the scanner defensive.", "warning"
    return "Invalidation", f"Below SMA50 near {format_price(sma50)}, the scanner remains defensive until price repairs the trend.", "negative"


def build_followthrough_guidance(detail_row: dict, context: dict) -> tuple[str, str, str]:
    volume_ratio = safe_numeric(detail_row.get("volume_ratio"))
    if volume_ratio is None:
        return "Follow-through", "Volume confirmation is unavailable; use price reclaim and close quality as the main confirmation checks.", "warning"
    if volume_ratio >= 1.2:
        return "Follow-through", f"Relative volume is {format_optional_number(volume_ratio)}x, which supports follow-through if price holds above EMA21.", "positive"
    return "Follow-through", f"Relative volume is only {format_optional_number(volume_ratio)}x; look for expansion above 1.2x before trusting a breakout attempt.", "warning"


def build_profit_risk_guidance(detail_row: dict, fundamentals: dict, context: dict) -> tuple[str, str, str]:
    price = safe_numeric(detail_row.get("price"))
    high_52w = safe_numeric(fundamentals.get("fiftyTwoWeekHigh"))
    if context.get("extension_status") in {"Extended", "Too Extended"}:
        return "Profit / Risk", f"Price is {context.get('extension_status')} versus EMA21 at {format_signed_percent(context.get('distance_ema21'))}; manage adds tightly and watch for a reset.", "warning"
    if price is not None and high_52w is not None and high_52w > price:
        upside = calculate_percent_change(high_52w, price)
        return "Profit / Risk", f"First upside reference is the 52-week high near {format_price(high_52w)} ({format_signed_percent(upside)} from current price); compare that against EMA21/SMA50 risk.", "positive"
    if price is not None and high_52w is not None and price >= high_52w:
        return "Profit / Risk", "Price is near or above the 52-week high; use extension versus EMA21 as the main profit-protection check.", "warning"
    return "Profit / Risk", "Use the next resistance area on the chart against EMA21/SMA50 invalidation to judge risk/reward.", "neutral"


def build_trading_guidance(detail_row: dict, fundamentals: dict, context: dict) -> dict:
    action = get_guidance_action_bias(context)
    entry_label, entry_text, entry_tone = build_entry_guidance(context)
    invalidation_label, invalidation_text, invalidation_tone = build_invalidation_guidance(context)
    follow_label, follow_text, follow_tone = build_followthrough_guidance(detail_row, context)
    risk_label, risk_text, risk_tone = build_profit_risk_guidance(detail_row, fundamentals, context)
    rows = [
        ("Action Bias", action["note"], action["label"], action["tone"]),
        (entry_label, entry_text, "Entry", entry_tone),
        (invalidation_label, invalidation_text, "Sell Risk", invalidation_tone),
        (follow_label, follow_text, "Confirm", follow_tone),
        (risk_label, risk_text, "Manage", risk_tone),
    ]
    return {
        "action": action,
        "entry_label": entry_label,
        "entry_text": entry_text,
        "rows": rows,
    }


def build_fundamental_guidance(fundamentals: dict) -> dict:
    revenue_growth = safe_numeric(fundamentals.get("revenueGrowth"))
    earnings_growth = safe_numeric(fundamentals.get("earningsGrowth"))
    operating_margin = safe_numeric(fundamentals.get("operatingMargins"))
    profit_margin = safe_numeric(fundamentals.get("profitMargins"))
    forward_pe = fundamentals.get("forwardPE")
    free_cashflow = fundamentals.get("freeCashflow")
    operating_cashflow = fundamentals.get("operatingCashflow")
    debt_to_equity = fundamentals.get("debtToEquity")

    revenue_status, revenue_tone = classify_growth_guidance(revenue_growth)
    earnings_status, earnings_tone = classify_growth_guidance(earnings_growth, strong_threshold=0.20, good_threshold=0.08)
    margin_status, margin_tone = classify_margin_guidance(operating_margin, profit_margin)
    cashflow_status, cashflow_tone = classify_cashflow_guidance(free_cashflow, operating_cashflow)
    debt_status, debt_tone = classify_debt_guidance(debt_to_equity)
    valuation_status, valuation_tone = classify_valuation_guidance(forward_pe, revenue_growth, earnings_growth)

    score = 0
    for tone in (revenue_tone, earnings_tone, margin_tone, cashflow_tone, debt_tone, valuation_tone):
        if tone == "positive":
            score += 1
        elif tone == "negative":
            score -= 1
    available_count = sum(
        value is not None
        for value in (
            revenue_growth,
            earnings_growth,
            operating_margin,
            profit_margin,
            safe_numeric(forward_pe),
            safe_numeric(free_cashflow),
            safe_numeric(operating_cashflow),
            normalize_debt_to_equity(debt_to_equity),
        )
    )
    if available_count < 3:
        bias = {"label": "Limited Data", "tone": "warning", "note": "Fundamental data is too thin for a firm scanner read."}
    elif score >= 3:
        bias = {"label": "Improving Fundamentals", "tone": "positive", "note": "Growth, profitability, cash flow, or balance-sheet checks lean constructive."}
    elif score >= 0:
        bias = {"label": "Mixed Fundamentals", "tone": "warning", "note": "Business quality is usable, but not all growth and profitability checks confirm."}
    else:
        bias = {"label": "Weak Fundamentals", "tone": "negative", "note": "Growth, margin, cash flow, or leverage checks are not supporting the setup."}

    watch_items = []
    if revenue_growth is None or revenue_growth < 0.08:
        watch_items.append("revenue/sales growth")
    if earnings_growth is None or earnings_growth < 0.08:
        watch_items.append("earnings growth")
    if operating_margin is None or operating_margin < 0.10:
        watch_items.append("operating margin")
    if safe_numeric(free_cashflow) is None or safe_numeric(free_cashflow) <= 0:
        watch_items.append("free cash flow")
    if not watch_items:
        watch_text = "Revenue, earnings, margins, and cash flow already screen positively; watch that they stay positive in the next report."
    else:
        watch_text = f"Most important improvements to watch: {', '.join(watch_items[:4])}."

    rows = [
        ("Revenue / Sales", f"Revenue growth is {format_percent_optional(revenue_growth)}; the scanner wants sales growth to stay positive and preferably accelerate.", revenue_status, revenue_tone),
        ("Earnings", f"Earnings growth is {format_percent_optional(earnings_growth)}; stronger EPS growth improves conviction behind technical setups.", earnings_status, earnings_tone),
        ("Profit Quality", f"Operating margin is {format_percent_ratio(operating_margin)} and profit margin is {format_percent_ratio(profit_margin)}.", margin_status, margin_tone),
        ("Cash Flow", f"Free cash flow is {format_compact_money(free_cashflow, signed=True)} and operating cash flow is {format_compact_money(operating_cashflow, signed=True)}.", cashflow_status, cashflow_tone),
        ("Balance Sheet", f"Debt/equity is {format_debt_to_equity(debt_to_equity)}; lower leverage gives the setup more room for error.", debt_status, debt_tone),
        ("Valuation", f"Forward P/E is {format_multiple(forward_pe)}; valuation needs to be supported by revenue and earnings growth.", valuation_status, valuation_tone),
        ("Watch Next", watch_text, "Next Report", "warning"),
    ]
    return {
        "bias": bias,
        "rows": rows,
    }


def render_stock_guidance(
    detail_row: dict,
    fundamentals: dict,
    fundamentals_snapshot: dict | None,
    chart_history=None,
) -> None:
    ticker = str(detail_row.get("ticker") or "Ticker").upper()
    context = get_long_trend_context(detail_row, chart_history=chart_history)
    trading = build_trading_guidance(detail_row, fundamentals, context)
    fundamental = build_fundamental_guidance(fundamentals)
    action = trading["action"]
    fundamental_bias = fundamental["bias"]
    entry_zone = format_guidance_zone(context.get("ema8"), context.get("ema21"))
    invalidation_zone = format_guidance_zone(context.get("ema21"), context.get("sma50"))
    snapshot_note = " Local fundamentals snapshot is available." if fundamentals_snapshot else ""

    stat_html = "".join(
        [
            build_guidance_stat_html("Price", format_price(detail_row.get("price"))),
            build_guidance_stat_html("Action Bias", action["label"]),
            build_guidance_stat_html("Entry Area", entry_zone),
            build_guidance_stat_html("Sell Risk Line", invalidation_zone),
            build_guidance_stat_html("Fundamental Bias", fundamental_bias["label"]),
        ]
    )
    trading_rows = "".join(build_guidance_row_html(*row) for row in trading["rows"])
    fundamental_rows = "".join(build_guidance_row_html(*row) for row in fundamental["rows"])

    st.markdown(
        (
            "<div class='guidance-brief'>"
            "<div class='guidance-hero'>"
            "<div>"
            "<div class='guidance-eyebrow'>Trade and fundamentals guidance</div>"
            f"<div class='guidance-headline'>{escape(ticker)} scanner decision guide</div>"
            f"<div class='guidance-lede'>{escape(action['note'])} Fundamental read: {escape(fundamental_bias['note'])}</div>"
            "</div>"
            "<div class='guidance-scorebox'>"
            "<div class='guidance-score-label'>Current Read</div>"
            f"<div class='guidance-score-value'>{escape(action['label'])}</div>"
            f"<div class='guidance-score-note'>{escape(format_text_value(context.get('long_status')))} | {escape(format_text_value(context.get('extension_status')))}</div>"
            "</div>"
            "</div>"
            f"<div class='guidance-strip'>{stat_html}</div>"
            "<div class='guidance-grid'>"
            "<section class='guidance-panel'>"
            "<div class='guidance-panel-title'>Trading Guide</div>"
            f"{trading_rows}"
            "</section>"
            "<section class='guidance-panel'>"
            "<div class='guidance-panel-title'>Fundamental Guide</div>"
            f"{fundamental_rows}"
            "</section>"
            "</div>"
            "<div class='guidance-note'>"
            f"Uses scanner moving averages, Yahoo Finance fundamentals, and local price history.{escape(snapshot_note)} "
            "This is decision support, not financial advice."
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def extract_prices_from_summary(summary: str | None) -> tuple[float | None, float | None]:
    if not summary:
        return None, None
    matches = [safe_numeric(value.replace("$", "").replace(",", "")) for value in re.findall(r"\$([0-9]+(?:\.[0-9]+)?)", str(summary))]
    if len(matches) >= 2:
        return matches[0], matches[1]
    if len(matches) == 1:
        return matches[0], None
    return None, None


def build_local_trade_ideas(
    detail_row: dict,
    fundamentals: dict,
    fundamentals_snapshot: dict | None,
    chart_history=None,
) -> list[dict]:
    context = get_long_trend_context(detail_row, chart_history=chart_history)
    technical = build_technical_readout(chart_history, ticker=detail_row.get("ticker")) if chart_history is not None else {}
    fundamental_bias = build_fundamental_guidance(fundamentals)
    price = safe_numeric(detail_row.get("price"))
    ema8 = safe_numeric(context.get("ema8"))
    ema21 = safe_numeric(context.get("ema21"))
    sma50 = safe_numeric(context.get("sma50"))
    sma200 = safe_numeric(context.get("sma200"))
    high_52w = safe_numeric(fundamentals.get("fiftyTwoWeekHigh"))
    low_52w = safe_numeric(fundamentals.get("fiftyTwoWeekLow"))
    atr_pct = safe_numeric(detail_row.get("atr_pct"))
    atr_dollars = price * atr_pct / 100 if price is not None and atr_pct is not None else None
    support, resistance = extract_prices_from_summary(technical.get("support_resistance_summary"))
    trend_status = str(context.get("long_status") or "")
    extension_status = str(context.get("extension_status") or "")
    pullback_quality = str(context.get("pullback_quality") or "")
    volume_ratio = safe_numeric(detail_row.get("volume_ratio"))
    fundamental_label = str((fundamental_bias.get("bias") or {}).get("label") or "Mixed Fundamentals")
    momentum_summary = str(technical.get("momentum_summary") or "").strip()
    support_summary = str(technical.get("support_resistance_summary") or "").strip()
    swing_read = str(technical.get("swing_read") or "").strip()

    def fmt_price(value: object) -> str:
        return format_price(value)

    def fmt_pct(value: object) -> str:
        return format_signed_percent_optional(value, decimals=1)

    ideas: list[dict] = []

    if trend_status in {"Strong Long Setup", "Pullback Entry Setup"} and extension_status != "Too Extended":
        entry_anchor = resistance or ema21 or high_52w or price
        stop_anchor = min([value for value in (support, sma50, ema21, low_52w) if value is not None], default=None)
        if stop_anchor is None and price is not None and atr_dollars is not None:
            stop_anchor = price - atr_dollars * 1.5
        target_anchor = high_52w or (resistance * 1.06 if resistance is not None else None)
        if entry_anchor is not None and stop_anchor is not None:
            ideas.append(
                {
                    "title": "Primary: momentum continuation long",
                    "summary": "Use this when the trend is already working and price can keep confirming strength.",
                    "entry": f"Above {fmt_price(entry_anchor)}",
                    "stop": f"Below {fmt_price(stop_anchor)}",
                    "target": fmt_price(target_anchor),
                    "confidence": "High" if trend_status == "Strong Long Setup" and volume_ratio is not None and volume_ratio >= 1.2 else "Medium",
                    "why": f"Trend status is {trend_status.lower() or 'constructive'}, volume is {format_optional_number(volume_ratio)}x, and the fundamental read is {fundamental_label.lower()}. {momentum_summary}",
                }
            )

    if trend_status in {"Strong Long Setup", "Pullback Entry Setup", "Long Watchlist"} and extension_status != "Too Extended":
        pullback_entry = ema21 or ema8 or sma50 or price
        pullback_stop = min([value for value in (sma50, sma200, support, low_52w) if value is not None], default=None)
        if pullback_stop is None and pullback_entry is not None and atr_dollars is not None:
            pullback_stop = pullback_entry - atr_dollars * 1.25
        pullback_target = high_52w or resistance or price
        if pullback_entry is not None and pullback_stop is not None:
            ideas.append(
                {
                    "title": "Alternate: pullback buy on reclaim",
                    "summary": "A lower-stress entry that waits for the stock to hold a nearby moving-average zone.",
                    "entry": f"Buy the zone near {fmt_price(pullback_entry)}",
                    "stop": f"Below {fmt_price(pullback_stop)}",
                    "target": fmt_price(pullback_target),
                    "confidence": "Medium" if trend_status != "Long Watchlist" else "Low",
                    "why": f"Pullback quality is {pullback_quality or 'not fully defined'}, support/resistance reads {support_summary or 'as nearby moving averages and recent range'}, and the setup stays cleaner when volume is not collapsing.",
                }
            )

    if extension_status in {"Extended", "Too Extended"} or trend_status in {"Weak / Avoid", "Below Key MAs"}:
        watch_trigger = resistance or ema21 or sma50 or high_52w
        invalidation = sma200 or low_52w or sma50
        ideas.append(
            {
                "title": "Defensive: wait for repair",
                "summary": "This is the local no-chase idea when price is extended or trend quality is not clean.",
                "entry": f"Wait for a close back above {fmt_price(watch_trigger)}" if watch_trigger is not None else "Wait for a cleaner reclaim",
                "stop": f"Invalidate below {fmt_price(invalidation)}" if invalidation is not None else "Use the next major moving average as invalidation",
                "target": fmt_price(high_52w or resistance),
                "confidence": "High",
                "why": f"Trend status is {trend_status.lower() or 'unavailable'} and extension status is {extension_status.lower() or 'unknown'}. The app should not force a long idea when the structure still needs repair.",
            }
        )

    if not ideas:
        ideas.append(
            {
                "title": "No clean local idea",
                "summary": "The local engine does not see a clean swing setup right now.",
                "entry": "N/A",
                "stop": "N/A",
                "target": "N/A",
                "confidence": "Low",
                "why": f"{swing_read or 'The chart read is incomplete.'} {support_summary or ''}".strip(),
            }
        )

    return ideas[:3]


def render_local_trade_ideas(
    detail_row: dict,
    fundamentals: dict,
    fundamentals_snapshot: dict | None,
    chart_history=None,
) -> None:
    ideas = build_local_trade_ideas(detail_row, fundamentals, fundamentals_snapshot, chart_history=chart_history)
    with st.expander("AI Trade Ideas", expanded=False):
        st.caption("Generated locally from the scanner, chart context, and fundamentals. No API call.")
        for idea in ideas:
            render_compact_metric_grid(
                idea["title"],
                [
                    ("Entry", idea["entry"]),
                    ("Stop", idea["stop"]),
                    ("Target", idea["target"]),
                    ("Confidence", idea["confidence"]),
                ],
                columns=4,
            )
            st.caption(idea["summary"])
            st.write(idea["why"])


def render_stock_thesis(
    detail_row: dict,
    fundamentals: dict,
    fundamentals_snapshot: dict | None,
    company_profile: dict | None,
    latest_filing: dict,
    earnings_info: dict,
    forward_profile: dict | None = None,
    chart_history=None,
) -> None:
    context = get_long_trend_context(detail_row, chart_history=chart_history)
    verdict = get_thesis_verdict(context)
    technical_section = build_technical_thesis_section(detail_row, chart_history, context)
    confirmation_items = build_confirmation_items(detail_row, context)[:4]
    risk_items = build_risk_factors(detail_row, fundamentals, earnings_info, context)[:4]

    st.markdown("#### Setup Notes")
    render_compact_metric_grid(
        "Current Read",
        [
            ("Verdict", verdict["label"]),
            ("Setup", context.get("long_status", "N/A")),
            ("EMA Stack", context.get("ema_stack_status", "N/A")),
            ("vs EMA21", format_signed_percent(context.get("distance_ema21"))),
            ("Rel Volume", format_optional_number(detail_row.get("volume_ratio"))),
        ],
        columns=5,
    )
    st.caption(technical_section["text"])

    note_col, risk_col = st.columns(2)
    with note_col:
        st.markdown("**Confirmation**")
        for item in confirmation_items:
            st.markdown(f"- {item}")
    with risk_col:
        st.markdown("**Risks**")
        for item in risk_items:
            st.markdown(f"- {item}")


def render_stock_detail(
    detail_row: dict | None,
    memberships: list[str],
    fundamentals: dict,
    fundamentals_snapshot: dict | None,
    company_profile: dict | None,
    latest_filing: dict,
    earnings_info: dict,
    forward_profile: dict,
    chart_available: bool,
    chart_history=None,
    chart_renderer=None,
) -> None:
    render_stock_detail_styles()

    if not detail_row:
        st.info("No stock selected.")
        return

    render_selected_stock_header(
        detail_row=detail_row,
        memberships=memberships,
        fundamentals=fundamentals,
        earnings_info=earnings_info,
        latest_filing=latest_filing,
        chart_history=chart_history,
    )

    render_company_snapshot(company_profile, fundamentals=fundamentals)
    render_stock_thesis(
        detail_row=detail_row,
        fundamentals=fundamentals,
        fundamentals_snapshot=fundamentals_snapshot,
        company_profile=company_profile,
        latest_filing=latest_filing,
        earnings_info=earnings_info,
        forward_profile=forward_profile,
        chart_history=chart_history,
    )
    render_stock_guidance(
        detail_row=detail_row,
        fundamentals=fundamentals,
        fundamentals_snapshot=fundamentals_snapshot,
        chart_history=chart_history,
    )
    render_simple_ma_summary(detail_row, chart_history=chart_history)
    render_historical_ma_reclaim_returns(detail_row.get("ma_reclaim_stats"), ticker=detail_row.get("ticker"))

    chart_tab, fundamentals_tab, dcf_tab, sec_tab = st.tabs(["Chart", "Fundamentals", "DCF", "SEC Filing"])
    with chart_tab:
        if chart_available and chart_renderer:
            chart_renderer()
            render_technical_readout(chart_history, ticker=detail_row.get("ticker"))
        else:
            st.info("No chart data available for the selected stock.")
    with fundamentals_tab:
        render_simple_fundamentals_tab(fundamentals, fundamentals_snapshot=fundamentals_snapshot, detail_row=detail_row)
    with dcf_tab:
        render_damodaran_dcf_tab(fundamentals, detail_row=detail_row)
    with sec_tab:
        render_sec_filing_tab(latest_filing)


def render_fundamentals(fundamentals: dict, detail_row: dict | None = None, earnings_info: dict | None = None) -> None:
    st.markdown("#### Fundamentals")

    if not fundamentals:
        st.info("No fundamentals available.")
        return

    current_price = (detail_row or {}).get("price")
    average_volume = fundamentals.get("averageVolume")
    dollar_volume = None
    if safe_numeric(current_price) is not None and safe_numeric(average_volume) is not None:
        dollar_volume = safe_numeric(current_price) * safe_numeric(average_volume)
    pct_from_high = calculate_percent_change(current_price, fundamentals.get("fiftyTwoWeekHigh"))
    pct_above_low = calculate_percent_change(current_price, fundamentals.get("fiftyTwoWeekLow"))
    next_earnings_date = (earnings_info or {}).get("next_earnings_date")

    with st.expander("How to read these fundamentals"):
        st.caption("Market Cap: company equity value based on share price and shares outstanding.")
        st.caption("Average Volume: typical daily shares traded; higher volume usually improves tradeability.")
        st.caption("Dollar Volume: current price multiplied by average volume; a liquidity context check.")
        st.caption("Revenue Growth and Earnings Growth: recent business growth rates from Yahoo Finance.")
        st.caption("Gross Margins and Operating Margins: profitability before and after operating costs.")
        st.caption("Return on Equity: profit generated relative to shareholder equity.")
        st.caption("Debt to Equity: leverage relative to equity; higher values can increase risk.")
        st.caption("52W High / Low: current price context versus the past year's range.")
        st.caption("Short % of Float: portion of tradable shares sold short.")
        st.caption("Days to Cover: estimated days for shorts to cover based on average volume.")
        st.caption("Float Shares: shares generally available for public trading.")
        st.caption("Institutional Ownership and Insider Ownership: ownership context from reported holders.")

    st.markdown("##### Core Trading Context")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Market Cap", format_compact_number(fundamentals.get("marketCap")))
        st.metric("Average Volume", format_compact_number(average_volume))
    with col2:
        render_metric_with_context("Dollar Volume", format_compact_number(dollar_volume), get_context_label("dollar_volume", dollar_volume))
        st.metric("Sector", format_text_value(fundamentals.get("sector")))
    with col3:
        st.metric("Industry", format_text_value(fundamentals.get("industry")))
        st.metric("Beta", format_optional_number(fundamentals.get("beta")))

    st.markdown("##### Valuation / Business Quality")
    val_col1, val_col2, val_col3, val_col4 = st.columns(4)
    with val_col1:
        st.metric("Trailing P/E", format_multiple(fundamentals.get("trailingPE")))
        st.metric("Forward P/E", format_multiple(fundamentals.get("forwardPE")))
    with val_col2:
        st.metric("Revenue Growth", format_percent_optional(fundamentals.get("revenueGrowth")))
        st.metric("Earnings Growth", format_percent_optional(fundamentals.get("earningsGrowth")))
    with val_col3:
        st.metric("Gross Margins", format_percent_ratio(fundamentals.get("grossMargins")))
        st.metric("Operating Margins", format_percent_ratio(fundamentals.get("operatingMargins")))
    with val_col4:
        st.metric("Return on Equity", format_percent_optional(fundamentals.get("returnOnEquity")))
        render_metric_with_context(
            "Debt to Equity",
            format_debt_to_equity(fundamentals.get("debtToEquity")),
            get_context_label("debt_to_equity", fundamentals.get("debtToEquity")),
        )

    st.markdown("##### Price Context")
    price_col1, price_col2, price_col3 = st.columns(3)
    with price_col1:
        st.metric("52W High", format_price(fundamentals.get("fiftyTwoWeekHigh")))
        render_metric_with_context(
            "% From 52W High",
            format_signed_percent_optional(pct_from_high),
            get_context_label("from_52w_high", pct_from_high),
        )
    with price_col2:
        st.metric("52W Low", format_price(fundamentals.get("fiftyTwoWeekLow")))
        st.metric("% Above 52W Low", format_signed_percent_optional(pct_above_low))
    with price_col3:
        st.metric("Next Earnings Date", next_earnings_date or "N/A")


def render_short_interest(fundamentals: dict) -> None:
    st.markdown("#### Short Interest")

    if not fundamentals:
        st.info("No short interest data available.")
        return

    short_change = calculate_short_interest_change(fundamentals)

    col1, col2, col3 = st.columns(3)
    with col1:
        render_metric_with_context(
            "Short % of Float",
            format_percent_ratio(fundamentals.get("shortPercentOfFloat")),
            get_context_label("short_percent_float", fundamentals.get("shortPercentOfFloat")),
        )
        st.metric("Short % of Shares Outstanding", format_percent_ratio(fundamentals.get("shortPercentOfSharesOutstanding")))
        st.metric("Shares Short", format_compact_number(fundamentals.get("sharesShort")))
    with col2:
        st.metric("Days to Cover / Short Ratio", format_days(fundamentals.get("shortRatio")))
        st.metric("Shares Short Prior Month", format_compact_number(fundamentals.get("sharesShortPriorMonth")))
        st.metric("Short Interest Change %", format_signed_percent_optional(short_change))
    with col3:
        st.metric("Float Shares", format_compact_number(fundamentals.get("floatShares")))
        st.metric("Institutional Ownership", format_percent_ratio(fundamentals.get("heldPercentInstitutions")))
        st.metric("Insider Ownership", format_percent_ratio(fundamentals.get("heldPercentInsiders")))

    with st.expander("Short Interest"):
        st.write(
            "Short interest shows how much of a company's float has been sold short. High short interest can "
            "create squeeze potential, but it can also signal real bearish pressure."
        )
        st.caption(
            "Calculation: Short % of Float = shares sold short divided by float shares. Short Ratio / Days to "
            "Cover estimates how many trading days it may take short sellers to cover based on average volume."
        )
        st.caption(
            "Why it matters: For swing trading, high short interest can add fuel to breakouts if price reclaims "
            "key levels with strong volume. It should be used as context, not as a standalone buy signal."
        )
    st.caption("Short interest data may be delayed or unavailable depending on Yahoo Finance coverage.")


def render_earnings_info(earnings_info: dict) -> None:
    if not earnings_info or earnings_info.get("status") != "available":
        render_compact_metric_grid("Earnings", [("Status", "Earnings date unavailable")], columns=1)
        return

    next_earnings_date = earnings_info.get("next_earnings_date")
    days_until_earnings = earnings_info.get("days_until_earnings")
    last_earnings_date = earnings_info.get("last_earnings_date")

    render_compact_metric_grid(
        "Earnings",
        [
            ("Next Earnings", next_earnings_date or "N/A"),
            ("Days Until Earnings", days_until_earnings if days_until_earnings is not None else "N/A"),
            ("Last Earnings", last_earnings_date or "N/A"),
        ],
        columns=3,
    )

    if days_until_earnings is None:
        st.warning("Earnings date unavailable")
    elif days_until_earnings <= 3:
        st.error("Earnings Imminent")
    elif days_until_earnings <= 10:
        st.warning("Earnings Soon")


def render_latest_filing(latest_filing: dict) -> None:
    if not latest_filing:
        render_compact_metric_grid("Latest Filing", [("Status", "No SEC filing found.")], columns=1)
        return

    form = latest_filing.get("form", "Filing")
    filing_month = format_filing_month(latest_filing.get("date"))
    url = latest_filing.get("url")
    if not url:
        render_compact_metric_grid("Latest Filing", [("Status", "No SEC filing found.")], columns=1)
        return

    label = f"View {form}"
    if filing_month:
        label = f"{label} - {filing_month}"

    render_compact_metric_grid(
        "Latest Filing",
        [
            ("Form", form),
            ("Date", latest_filing.get("date") or "N/A"),
            ("Link", label),
        ],
        columns=3,
    )
    st.markdown(f'<a href="{url}" target="_blank" rel="noopener noreferrer">{escape(label)}</a>', unsafe_allow_html=True)


def render_pivot_context(metrics: dict) -> None:
    pivot = metrics.get("pivot")
    distance_to_pivot_pct = metrics.get("distance_to_pivot_pct")
    breakout_status = metrics.get("breakout_status") or "Unavailable"

    render_compact_metric_grid(
        "Pivot / Breakout Context",
        [
            ("Pivot Level", format_price(pivot)),
            ("Distance to Pivot", format_signed_percent(distance_to_pivot_pct)),
            ("Breakout Status", breakout_status),
        ],
        columns=3,
    )


def render_historical_reaction(detail_row: dict) -> None:
    historical_stats = detail_row.get("historical_stats", {})
    active_event = detail_row.get("active_reaction_event")
    best_event = detail_row.get("best_reaction_event")
    selected_event = active_event or best_event
    selected_stats = historical_stats.get(selected_event, {}) if selected_event else {}

    render_compact_metric_grid(
        "Historical Reaction Summary",
        [
            ("Historical Score", format_optional_number(detail_row.get("historical_score"), decimals=1)),
            ("Confidence", detail_row.get("historical_confidence", "Low")),
            ("Active Event", active_event or "None"),
            ("Best Event", best_event or "N/A"),
            ("Samples", format_optional_number(selected_stats.get("event_count"), decimals=0)),
            ("Median Drawdown 40D", format_signed_percent(selected_stats.get("median_max_drawdown_40d"))),
            ("20D Win Rate", format_plain_percent(selected_stats.get("win_rate_20d"))),
            ("Median 20D Return", format_signed_percent(selected_stats.get("median_return_20d"))),
            ("40D Win Rate", format_plain_percent(selected_stats.get("win_rate_40d"))),
            ("Median 40D Return", format_signed_percent(selected_stats.get("median_return_40d"))),
            ("60D Win Rate", format_plain_percent(selected_stats.get("win_rate_60d"))),
            ("Median 60D Return", format_signed_percent(selected_stats.get("median_return_60d"))),
        ],
        columns=4,
    )

    summary = detail_row.get("historical_summary")
    if summary:
        st.caption(summary)
    else:
        st.info("No historical reaction summary available.")


def render_forward_return_profile(forward_profile: dict) -> None:
    if not forward_profile:
        st.info("Forward return profile unavailable.")
        return

    render_compact_metric_grid(
        "Forward Return Profile",
        [
            ("20D Estimate", format_signed_percent(forward_profile.get("estimate_20d"))),
            ("40D Estimate", format_signed_percent(forward_profile.get("estimate_40d"))),
            ("60D Estimate", format_signed_percent(forward_profile.get("estimate_60d"))),
            ("Bear Case", format_signed_percent(forward_profile.get("bear_case"))),
            ("Base Case", format_signed_percent(forward_profile.get("base_case"))),
            ("Bull Case", format_signed_percent(forward_profile.get("bull_case"))),
            ("Confidence", forward_profile.get("confidence", "Low")),
            ("Samples", format_optional_number(forward_profile.get("sample_size"), decimals=0)),
            ("Active Event", forward_profile.get("active_event") or "N/A"),
        ],
        columns=3,
    )

    if forward_profile.get("summary"):
        st.caption(forward_profile["summary"])

    drivers = forward_profile.get("drivers", [])
    if drivers:
        st.markdown("Key drivers:")
        for driver in drivers:
            st.markdown(f"- {driver}")

    st.caption("Estimate based on historical technical reactions and current setup profile. Not a guaranteed forecast.")


def render_score_explanation(detail_row: dict) -> None:
    explanations = detail_row.get("explanation", [])
    if not explanations:
        render_compact_metric_grid("Why This Scored This Way", [("Explanation", "No scoring explanation available.")], columns=1)
        return

    rows = "".join(f"<div class='detail-explanation'>{escape(str(explanation))}</div>" for explanation in explanations)
    st.markdown(
        f"<div class='fs-box'><div class='fs-box-title'>Why This Scored This Way</div>{rows}</div>",
        unsafe_allow_html=True,
    )
