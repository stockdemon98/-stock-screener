from datetime import datetime

import streamlit as st

from charts import render_stock_chart
from data_fetch import get_company_profile, get_earnings_info, get_extended_hours_quote, get_forward_estimates, get_latest_sec_filing, get_stock_data
from data_fetch import fetch_fundamentals_snapshot, get_stock_chart_history, get_stock_fundamentals
from historical_reactions import calculate_ma_reclaim_stats
from scanner import build_long_trend_snapshot, load_snapshot_for_fast_scan, postprocess_trigger_rows, refresh_scanner_snapshot, run_breakout_scan_from_snapshot, run_fast_scan_from_snapshot, scan_trigger_events
from ui import render_controls, render_empty_state
from ui import render_main_trend_scanner, render_price_filter_note, render_scan_not_run, render_scan_summary, render_scanner_description
from ui import render_breakout_scan_controls, render_breakout_scan_results, render_search_result, render_stock_detail, render_ticker_search, render_trigger_scan_controls, render_trigger_scan_results
from ticker_lookup import build_ticker_lookup
from universes import get_ticker_memberships, get_universe_names, normalize_ticker


def build_search_result_row(ticker: str, stock_data: dict, score_data: dict) -> dict:
    long_trend = build_long_trend_snapshot(
        stock_data.get("history"),
        stock_data.get("price"),
        stock_data.get("volume_ratio"),
    )
    return {
        "ticker": ticker,
        "price": stock_data.get("price"),
        "change_pct": stock_data.get("change_pct"),
        "volume": stock_data.get("volume"),
        "avg_volume": stock_data.get("avg_volume_20"),
        "volume_ratio": stock_data.get("volume_ratio"),
        "addv": stock_data.get("addv_20"),
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
        "raw_tags": score_data.get("raw_tags", []),
        "metrics": score_data.get("metrics", {}),
        "explanation": score_data.get("explanation", []),
        **long_trend,
    }


def evaluate_selected_ticker(ticker: str, chart_timeframe: str) -> tuple[dict | None, list[str], dict, dict, dict, dict, object, dict, dict]:
    normalized_ticker = normalize_ticker(ticker)
    if not normalized_ticker:
        return None, [], {}, {}, {}, {}, None, {}, {}

    memberships = get_ticker_memberships(normalized_ticker)
    stock_data = get_stock_data(normalized_ticker)
    if not stock_data:
        return None, memberships, {}, {}, {}, {}, None, {}, {}

    score_data = {}
    detail_row = build_search_result_row(normalized_ticker, stock_data, score_data)
    detail_row["extended_quote"] = get_extended_hours_quote(normalized_ticker)
    detail_row["ma_reclaim_stats"] = calculate_ma_reclaim_stats(stock_data.get("history"))
    company_profile = get_company_profile(normalized_ticker)
    fundamentals = get_stock_fundamentals(normalized_ticker)
    fundamentals_snapshot = fetch_fundamentals_snapshot(normalized_ticker)
    latest_filing = get_latest_sec_filing(normalized_ticker)
    earnings_info = get_earnings_info(normalized_ticker)
    forward_profile = get_forward_estimates(normalized_ticker)
    chart_history = get_stock_chart_history(normalized_ticker, chart_timeframe)

    return detail_row, memberships, fundamentals, latest_filing, earnings_info, forward_profile, chart_history, fundamentals_snapshot, company_profile


def evaluate_ticker_search(raw_ticker: str) -> tuple[str, list[str], dict | None, str | None]:
    normalized_ticker = normalize_ticker(raw_ticker)
    if not normalized_ticker:
        return "", [], None, None

    memberships = get_ticker_memberships(normalized_ticker)
    stock_data = get_stock_data(normalized_ticker)
    if not stock_data:
        return normalized_ticker, memberships, None, f"Could not fetch usable data for {normalized_ticker}."

    score_data = {}
    result_row = build_search_result_row(normalized_ticker, stock_data, score_data)
    return normalized_ticker, memberships, result_row, None


def build_scan_key(universe_name: str, max_tickers: int, scan_mode: str) -> tuple:
    return (universe_name, int(max_tickers), scan_mode, "snapshot-v2")


def apply_snapshot_ticker_cap(snapshot_df, snapshot_meta: dict, max_tickers: int):
    if max_tickers is None or int(max_tickers) <= 0:
        uncapped_meta = snapshot_meta.copy()
        uncapped_meta["requested_count"] = len(snapshot_df.index)
        uncapped_meta["scanned_count"] = len(snapshot_df.index)
        uncapped_meta["max_scan_cap"] = len(snapshot_df.index)
        uncapped_meta["prefilter_scored_count"] = len(snapshot_df.index)
        return snapshot_df.copy(), uncapped_meta
    if "long_status" in snapshot_df.columns:
        status_rank = {
            "Strong Long Setup": 0,
            "Pullback Entry Setup": 1,
            "Long Watchlist": 2,
            "Too Extended": 3,
            "Below Key MAs": 4,
            "Weak / Avoid": 5,
            "Insufficient Data": 6,
        }
        sort_df = snapshot_df.assign(
            _long_status_rank=snapshot_df["long_status"].map(status_rank).fillna(6)
        )
        if "volume_ratio" not in sort_df.columns:
            sort_df["volume_ratio"] = 0
        sorted_df = sort_df.sort_values(["_long_status_rank", "volume_ratio"], ascending=[True, False]).drop(columns=["_long_status_rank"])
    else:
        sort_column = "signal_score" if "signal_score" in snapshot_df.columns else None
        sorted_df = snapshot_df.sort_values(sort_column, ascending=False) if sort_column else snapshot_df
    capped_df = sorted_df.head(max_tickers).copy()
    capped_meta = snapshot_meta.copy()
    capped_meta["requested_count"] = len(capped_df.index)
    capped_meta["scanned_count"] = len(capped_df.index)
    capped_meta["max_scan_cap"] = int(max_tickers)
    capped_meta["prefilter_scored_count"] = len(capped_df.index)
    return capped_df, capped_meta


@st.cache_data(ttl=20 * 60, show_spinner=False)
def run_cached_raw_trigger_scan(
    universe_name: str,
    event_types: tuple[str, ...],
    lookback_days: int,
    max_tickers: int,
    tickers: tuple[str, ...],
) -> list[dict]:
    return scan_trigger_events(
        universe_name=universe_name,
        event_types=list(event_types),
        lookback_days=lookback_days,
        only_currently_above_ema21_sma50=False,
        max_results=10_000,
        max_tickers=max_tickers,
        tickers=tickers,
        optional_filters=None,
    )


def build_company_name_map() -> dict[str, str]:
    return {
        row.get("ticker", ""): row.get("name", "")
        for row in build_ticker_lookup()
        if row.get("ticker") and row.get("name")
    }


def with_company_names(rows: list[dict], company_name_map: dict[str, str]) -> list[dict]:
    enriched_rows = []
    for row in rows:
        ticker = normalize_ticker(row.get("ticker", ""))
        enriched_rows.append(
            {
                **row,
                "company": row.get("company") or row.get("company_name") or company_name_map.get(ticker) or "N/A",
            }
        )
    return enriched_rows


def enrich_scan_results(scan_results: dict | None, company_name_map: dict[str, str]) -> dict | None:
    if not scan_results:
        return scan_results
    enriched = {**scan_results}
    for section_name in ("all_ranked", "top_setups"):
        enriched[section_name] = with_company_names(scan_results.get(section_name, []), company_name_map)
    return enriched


def render_snapshot_status(snapshot_meta: dict, snapshot_available: bool) -> None:
    if not snapshot_available:
        st.warning("No scanner snapshot found. Click Refresh Market Data / Build Snapshot first.")
        return

    last_updated = snapshot_meta.get("last_updated")
    ticker_count = snapshot_meta.get("result_count") or snapshot_meta.get("scanned_count") or 0
    if last_updated:
        st.caption(
            (
                f"Snapshot last updated: {last_updated} | "
                f"Universe tickers found: {snapshot_meta.get('universe_count', 0)} | "
                f"Tickers attempted: {snapshot_meta.get('tickers_attempted', snapshot_meta.get('requested_count', 0))} | "
                f"Download success: {snapshot_meta.get('download_success_count', snapshot_meta.get('valid_data_count', 0))} | "
                f"Failed/no data: {snapshot_meta.get('missing_data_count', 0)} | "
                f"Not enough history: {snapshot_meta.get('not_enough_history_count', 0)} | "
                f"Usable snapshot tickers: {snapshot_meta.get('usable_snapshot_count', ticker_count)}"
            )
        )
        timings = snapshot_meta.get("timings", {})
        if timings:
            st.caption(
                (
                    f"Refresh runtime: {snapshot_meta.get('duration_seconds', 0):.1f}s | "
                    f"Download: {timings.get('data_fetch_seconds', 0):.1f}s | "
                    f"Snapshot prep: {timings.get('prefilter_seconds', 0):.1f}s | "
                    f"Trend prep: {timings.get('scoring_seconds', 0):.1f}s | "
                    f"Save: {timings.get('save_seconds', 0):.1f}s"
                )
            )
        try:
            age_hours = (datetime.now() - datetime.fromisoformat(last_updated)).total_seconds() / 3600
            if age_hours > 6:
                st.warning(f"Scanner snapshot may be stale. Last updated {age_hours:.1f} hours ago.")
        except ValueError:
            pass
    else:
        st.caption(f"Snapshot tickers available: {ticker_count}")


def render_scan_progress_callback(progress_bar, status_box):
    def update_progress(done: int, total: int, message: str) -> None:
        denominator = max(total, 1)
        progress_bar.progress(min(max(done / denominator, 0.0), 1.0))
        status_box.info(message)

    return update_progress


def main() -> None:
    st.set_page_config(page_title="Stock Screener", layout="wide")
    if "selected_ticker" not in st.session_state:
        st.session_state["selected_ticker"] = None
    if "scan_results" not in st.session_state:
        st.session_state["scan_results"] = None
    if "scan_universe" not in st.session_state:
        st.session_state["scan_universe"] = None
    if "last_scan_results" not in st.session_state:
        st.session_state["last_scan_results"] = None
    if "last_scan_meta" not in st.session_state:
        st.session_state["last_scan_meta"] = None
    if "last_scan_key" not in st.session_state:
        st.session_state["last_scan_key"] = None
    if "last_snapshot_df" not in st.session_state:
        st.session_state["last_snapshot_df"] = None
    if "last_snapshot_meta" not in st.session_state:
        st.session_state["last_snapshot_meta"] = {}
    if "last_snapshot_universe" not in st.session_state:
        st.session_state["last_snapshot_universe"] = None
    if "ma_rows" not in st.session_state:
        st.session_state["ma_rows"] = None
    if "trigger_scan_rows" not in st.session_state:
        st.session_state["trigger_scan_rows"] = None
    if "trigger_raw_rows" not in st.session_state:
        st.session_state["trigger_raw_rows"] = None
    if "trigger_raw_key" not in st.session_state:
        st.session_state["trigger_raw_key"] = None
    if "trigger_scan_key" not in st.session_state:
        st.session_state["trigger_scan_key"] = None
    if "breakout_scan_rows" not in st.session_state:
        st.session_state["breakout_scan_rows"] = None
    if "breakout_scan_key" not in st.session_state:
        st.session_state["breakout_scan_key"] = None
    if "saved_setups" not in st.session_state:
        st.session_state["saved_setups"] = []

    universe_names = get_universe_names()
    valid_universes = set(universe_names)
    for state_key in ("scan_universe", "last_snapshot_universe"):
        if st.session_state.get(state_key) not in valid_universes:
            st.session_state[state_key] = None
    raw_search_ticker = render_ticker_search()
    (
        selected_universe,
        min_score,
        max_results,
        max_tickers,
        scan_mode,
        scanner_filters,
        refresh_snapshot,
        run_stock_scan,
        calculate_historical_edge,
    ) = render_controls(universe_names)
    search_ticker, memberships, search_result, search_error = evaluate_ticker_search(raw_search_ticker)

    if search_ticker:
        render_search_result(
            ticker=search_ticker,
            memberships=memberships,
            result_row=search_result,
            error_message=search_error,
        )

    current_scan_key = build_scan_key(selected_universe, max_tickers, scan_mode)
    if refresh_snapshot:
        progress_bar = st.progress(0)
        status_box = st.empty()
        snapshot_payload = refresh_scanner_snapshot(
            universe_name=selected_universe,
            max_tickers=max_tickers,
            scan_mode=scan_mode,
            calculate_historical_edge=calculate_historical_edge,
            progress_callback=render_scan_progress_callback(progress_bar, status_box),
        )
        status_box.success("Snapshot refreshed.")
        st.session_state["last_snapshot_df"] = snapshot_payload.get("snapshot_df")
        st.session_state["last_snapshot_meta"] = snapshot_payload.get("meta", {})
        st.session_state["last_snapshot_universe"] = selected_universe
        st.session_state["last_scan_results"] = None
        st.session_state["last_scan_meta"] = None
        st.session_state["last_scan_key"] = current_scan_key
        st.session_state["scan_universe"] = selected_universe

    if run_stock_scan:
        progress_bar = st.progress(0)
        status_box = st.empty()
        snapshot_df, snapshot_meta = load_snapshot_for_fast_scan(selected_universe)
        if snapshot_df.empty:
            status_box.warning("No scanner snapshot found. Click Refresh Market Data / Build Snapshot first.")
        else:
            st.session_state["last_snapshot_df"] = snapshot_df
            st.session_state["last_snapshot_meta"] = snapshot_meta
            st.session_state["last_snapshot_universe"] = selected_universe
            st.session_state["last_scan_key"] = current_scan_key
            st.session_state["scan_universe"] = selected_universe
            status_box.success("Snapshot loaded.")

    snapshot_df = st.session_state.get("last_snapshot_df")
    snapshot_meta = st.session_state.get("last_snapshot_meta") or {}
    snapshot_matches = (
        snapshot_df is not None
        and not snapshot_df.empty
        and st.session_state.get("last_snapshot_universe") == selected_universe
        and snapshot_meta.get("scan_mode") == scan_mode
    )
    render_snapshot_status(snapshot_meta, snapshot_matches)

    if snapshot_matches:
        capped_snapshot_df, capped_snapshot_meta = apply_snapshot_ticker_cap(snapshot_df, snapshot_meta, max_tickers)
        scan_results = run_fast_scan_from_snapshot(
            snapshot_df=capped_snapshot_df,
            snapshot_meta=capped_snapshot_meta,
            universe_name=selected_universe,
            min_score=min_score,
            max_results=max_results,
            optional_filters=scanner_filters,
            progress_callback=render_scan_progress_callback(st.progress(0), st.empty()) if run_stock_scan else None,
        )
        st.session_state["scan_results"] = scan_results
    else:
        scan_results = None

    company_name_map = build_company_name_map()
    scan_results = enrich_scan_results(scan_results, company_name_map)
    scanner_universe = st.session_state.get("scan_universe") or selected_universe
    render_scanner_description(scanner_universe)
    main_tab, trigger_tab, breakout_tab = st.tabs(["Main Trend Scanner", "Trigger Scan", "Breakout Search"])

    with main_tab:
        if scan_results is None:
            render_scan_not_run()
        elif not scan_results.get("all_ranked"):
            render_scan_summary(scan_results)
            render_price_filter_note(scanner_filters)
            render_empty_state("No stocks matched the selected filters.")
        else:
            render_scan_summary(scan_results)
            render_main_trend_scanner(scan_results, scanner_filters=scanner_filters)

    with trigger_tab:
        (
            selected_events,
            age_band_label,
            min_days_ago,
            max_days_ago,
            only_current,
            max_trigger_results,
            one_row_per_ticker,
            show_all_trigger_events,
            run_trigger_scan,
        ) = render_trigger_scan_controls()
        trigger_detection_days = 45
        snapshot_tickers = ()
        if snapshot_matches and "ticker" in snapshot_df.columns:
            snapshot_tickers = tuple(str(ticker).strip().upper() for ticker in snapshot_df["ticker"].dropna().tolist())
        trigger_max_tickers = len(snapshot_tickers) if int(max_tickers or 0) <= 0 else max_tickers
        raw_trigger_key = (
            selected_universe,
            tuple(selected_events),
            trigger_detection_days,
            trigger_max_tickers,
            snapshot_tickers,
        )
        trigger_scan_key = (
            raw_trigger_key,
            only_current,
            scanner_filters,
            max_trigger_results,
            one_row_per_ticker,
            show_all_trigger_events,
            age_band_label,
            min_days_ago,
            max_days_ago,
        )
        if run_trigger_scan:
            if not snapshot_tickers:
                st.warning("No saved scanner snapshot found. Click Refresh Market Data / Build Snapshot first.")
            else:
                with st.spinner("Running Trigger Scan..."):
                    raw_trigger_rows = run_cached_raw_trigger_scan(
                        universe_name=selected_universe,
                        event_types=tuple(selected_events),
                        lookback_days=trigger_detection_days,
                        max_tickers=trigger_max_tickers,
                        tickers=snapshot_tickers,
                    )
                st.session_state["trigger_raw_rows"] = raw_trigger_rows
                st.session_state["trigger_raw_key"] = raw_trigger_key

        if st.session_state.get("trigger_raw_key") == raw_trigger_key and st.session_state.get("trigger_raw_rows") is not None:
            trigger_rows = postprocess_trigger_rows(
                rows=st.session_state.get("trigger_raw_rows") or [],
                optional_filters=scanner_filters,
                only_currently_above_ema21_sma50=only_current,
                one_row_per_ticker=one_row_per_ticker,
                show_all_trigger_events=show_all_trigger_events,
                min_days_ago=min_days_ago,
                max_days_ago=max_days_ago,
                max_results=max_trigger_results,
            )
            st.session_state["trigger_scan_rows"] = with_company_names(trigger_rows, company_name_map)
            st.session_state["trigger_scan_key"] = trigger_scan_key

        if st.session_state.get("trigger_scan_key") != trigger_scan_key:
            render_trigger_scan_results(None, scanner_filters=scanner_filters, age_band_label=age_band_label)
        else:
            render_trigger_scan_results(st.session_state.get("trigger_scan_rows"), scanner_filters=scanner_filters, age_band_label=age_band_label)

    with breakout_tab:
        (
            breakout_mode,
            resistance_lookback,
            max_distance_below_pct,
            fresh_breakout_window_days,
            max_distance_above_ema21_pct,
            use_breakout_relative_volume_filter,
            breakout_min_relative_volume,
            max_breakout_results,
            run_breakout_scan,
        ) = render_breakout_scan_controls()
        breakout_scan_key = (
            selected_universe,
            st.session_state.get("last_snapshot_universe"),
            snapshot_meta.get("last_updated"),
            breakout_mode,
            resistance_lookback,
            max_distance_below_pct,
            fresh_breakout_window_days,
            max_distance_above_ema21_pct,
            use_breakout_relative_volume_filter,
            breakout_min_relative_volume,
            max_breakout_results,
            scanner_filters,
        )
        if run_breakout_scan:
            if not snapshot_matches:
                st.warning("No saved scanner snapshot found. Click Refresh Market Data / Build Snapshot first.")
            else:
                breakout_rows = run_breakout_scan_from_snapshot(
                    snapshot_df=snapshot_df,
                    mode=breakout_mode,
                    resistance_lookback=resistance_lookback,
                    max_distance_below_pct=max_distance_below_pct,
                    fresh_breakout_window_days=fresh_breakout_window_days,
                    max_distance_above_ema21_pct=max_distance_above_ema21_pct,
                    use_relative_volume_filter=use_breakout_relative_volume_filter,
                    min_relative_volume=breakout_min_relative_volume,
                    max_results=max_breakout_results,
                    optional_filters=scanner_filters,
                )
                st.session_state["breakout_scan_rows"] = with_company_names(breakout_rows, company_name_map)
                st.session_state["breakout_scan_key"] = breakout_scan_key

        if st.session_state.get("breakout_scan_key") != breakout_scan_key:
            render_breakout_scan_results(None, scanner_filters=scanner_filters)
        else:
            render_breakout_scan_results(st.session_state.get("breakout_scan_rows"), scanner_filters=scanner_filters)

    selected_ticker = st.session_state.get("selected_ticker")
    if selected_ticker:
        st.divider()
        st.markdown("## Selected Stock Details")
        chart_timeframe = st.session_state.get("detail-chart-timeframe", "1Y")
        detail_row, detail_memberships, fundamentals, latest_filing, earnings_info, forward_profile, chart_history, fundamentals_snapshot, company_profile = evaluate_selected_ticker(
            selected_ticker,
            chart_timeframe=chart_timeframe,
        )
        render_stock_detail(
            detail_row=detail_row,
            memberships=detail_memberships,
            fundamentals=fundamentals,
            fundamentals_snapshot=fundamentals_snapshot,
            company_profile=company_profile,
            latest_filing=latest_filing,
            earnings_info=earnings_info,
            forward_profile=forward_profile,
            chart_available=chart_history is not None and not chart_history.empty,
            chart_history=chart_history,
            chart_renderer=lambda: render_stock_chart(chart_history, ticker=detail_row.get("ticker") if detail_row else None),
        )


if __name__ == "__main__":
    main()
