from __future__ import annotations

import argparse

from scanner import refresh_scanner_snapshot
from symbol_universe import FULL_MARKET_UNIVERSE


def print_progress(done: int, total: int, message: str) -> None:
    denominator = max(int(total or 1), 1)
    pct = min(max(done / denominator, 0), 1) * 100
    print(f"[{pct:5.1f}%] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the full NYSE + Nasdaq common-stock scanner snapshot.")
    parser.add_argument(
        "--historical-edge",
        action="store_true",
        help="Calculate Historical Edge for every valid ticker. This can be slow on the full market universe.",
    )
    args = parser.parse_args()

    if args.historical_edge:
        print("Historical Edge on the full NYSE + Nasdaq universe can be slow.", flush=True)

    payload = refresh_scanner_snapshot(
        universe_name=FULL_MARKET_UNIVERSE,
        max_tickers=0,
        scan_mode="Fast Scan",
        calculate_historical_edge=args.historical_edge,
        progress_callback=print_progress,
    )
    meta = payload.get("meta", {})
    print("", flush=True)
    print(f"Snapshot saved: {meta.get('snapshot_path', 'N/A')}", flush=True)
    print(f"Raw symbols loaded: {meta.get('raw_symbols_loaded', 'N/A')}", flush=True)
    print(f"Non-common removed: {meta.get('non_common_removed', 'N/A')}", flush=True)
    print(f"Final common stocks: {meta.get('final_common_stocks', meta.get('universe_count', 'N/A'))}", flush=True)
    print(f"Tickers attempted: {meta.get('tickers_attempted', 'N/A')}", flush=True)
    print(f"Usable snapshot tickers: {meta.get('usable_snapshot_count', 'N/A')}", flush=True)
    print(f"Failed/no data: {meta.get('missing_data_count', 0)}", flush=True)
    print(f"Invalid OHLCV: {meta.get('invalid_ohlcv_count', 0)}", flush=True)
    print(f"Not enough history: {meta.get('not_enough_history_count', 0)}", flush=True)


if __name__ == "__main__":
    main()
