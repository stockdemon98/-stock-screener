# Session Notes - 2026-04-25

## Stock Screener Changes

- Cleaned up universe labels and removed the exposed 15-stock fast test universe from the main dropdown.
- Added full-universe sources/caches for:
  - S&P 500
  - Nasdaq 100
  - Russell 2000
  - S&P MidCap 400
- Added defensive warnings for incomplete full-universe lists.
- Added cached Yahoo batch downloads and chunking for scan data.
- Split expensive scan work from display filtering so UI filters and Max Results reuse cached/session data.
- Added a local scanner snapshot system under `data/scanner_cache/`.
- Added two main scan actions:
  - `Refresh Market Data / Build Snapshot`
  - `Run Fast Scan`
- Fast scan now loads/filter/sorts a saved local snapshot instead of downloading market data.
- Added progress/status messages and timing breakdowns for snapshot refresh and fast scan.
- Added `Max Price ($)` as a post-scan filter with presets and custom input.
- Replaced the automatic Watchlist with manual `Saved Setups`.
- Added `Save Setup`, duplicate prevention, notes, and remove actions for saved setups.
- Fixed `Save Setup` so one click immediately updates `Saved Setups`.
- Made `Top Setups` a stricter high-conviction subset instead of a duplicate of `All Ranked`.
- Changed `All Ranked` to a more compact table-style display.

## Behavior To Remember

- `Run Fast Scan` requires an existing scanner snapshot for the selected universe.
- If no snapshot exists, click `Refresh Market Data / Build Snapshot` first.
- Snapshot refresh can still be slow because it downloads Yahoo OHLCV data.
- Warm fast scans and filter changes should be much faster because they use local snapshot rows.
- Heavy details remain lazy-loaded only after selecting a ticker:
  - charts
  - fundamentals
  - SEC links
  - options flow
  - forward/DCF-style detail data

## Key Files Changed

- `app.py`
- `ui.py`
- `scanner.py`
- `data_fetch.py`
- `universes.py`
- `data/sp500_universe.csv`
- `data/nasdaq100_universe.csv`
- `data/russell2000_universe.csv`
- `data/midcap400_universe.csv`

## Verification Run During Session

- `python -m py_compile app.py scanner.py data_fetch.py ui.py universes.py`
- Streamlit endpoint checked at `http://localhost:8501` and returned `200`.

# Session Notes - 2026-04-28

## Simplified EMA Chart Screener Checkpoint

- Simplified the app around a long-only technical chart workflow focused on EMA8, EMA21, SMA50, and SMA200.
- Kept the main selected-stock view focused on price, moving-average summary, trend status, and the chart.
- Removed the old score/setup/options/historical clutter from the visible stock detail tabs.
- Kept the final visible tabs limited to:
  - `Chart`
  - `Fundamentals`
  - `DCF`
  - `SEC Filing`
- Added clean Fundamentals and SEC Filing tabs while keeping them separate from the main chart view.
- Added SEC latest 10-Q / 10-K lookup using SEC EDGAR submissions data with a declared User-Agent and cached selected-ticker requests.
- Improved chart controls with compact columns for timeframe, chart type, volume, moving averages, break/reclaim markers, and Golden/Death Cross markers.
- Added chart marker toggles for EMA8, EMA21, SMA50, and SMA200 price breaks/reclaims.
- Added separate Golden Cross and Death Cross marker toggles.
- Kept moving-average chart lines solid and light-mode friendly.

## DCF Upgrade

- Added `dcf.py` as a standalone FCFF intrinsic valuation helper module.
- Replaced the simple FCF-growth DCF with a Damodaran-style FCFF model in the DCF tab.
- New DCF structure includes:
  - Story / Assumptions
  - Operating Forecast
  - Reinvestment / FCFF Forecast
  - Cost of Capital
  - Terminal Value
  - Equity Bridge
  - Bear / Base / Bull Scenarios
  - Sensitivity Table
  - Key warnings and plain-English valuation summary
- The model values operating assets using FCFF discounted at WACC, then bridges enterprise value to equity value by adding cash/investments and subtracting debt/minority interest.
- Added a few extra Yahoo `stock.info` fields to `data_fetch.py` for DCF prefills without adding new heavy API calls:
  - `ebit`
  - `operatingIncome`
  - `minorityInterest`
  - `totalInvestments`
  - `longTermInvestments`

## Key Files Changed In This Checkpoint

- `app.py`
- `ui.py`
- `charts.py`
- `scanner.py`
- `data_fetch.py`
- `dcf.py`
- `.streamlit/config.toml`

## Verification Run During This Checkpoint

- `.\venv\Scripts\python.exe -m py_compile dcf.py ui.py data_fetch.py app.py charts.py scanner.py scoring.py`
- Direct DCF smoke test passed with AAPL-like inputs.
- No additional Streamlit server was launched.

## Save Note

- This project folder is not currently a Git repository, so there was no Git commit to create.
- Current changes are saved directly in the project files on disk.
