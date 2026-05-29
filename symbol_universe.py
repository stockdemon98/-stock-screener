from __future__ import annotations

from functools import lru_cache
from io import StringIO
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "data"
SYMBOL_CACHE_DIR = DATA_DIR / "symbol_cache"
NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
FULL_MARKET_UNIVERSE = "NYSE_NASDAQ_COMMON"
EX_MAJOR_INDEXES_UNIVERSE = "NYSE_NASDAQ_EX_MAJOR_INDEXES"

EXCLUDED_NAME_TERMS = (
    " ETF ",
    " FUND ",
    " TRUST ",
    " WARRANT ",
    " WARRANTS ",
    " UNIT ",
    " UNITS ",
    " RIGHT ",
    " RIGHTS ",
    " PREFERRED ",
    " PREFERENCE ",
    " NOTE ",
    " NOTES ",
    " BOND ",
    " BONDS ",
    " DEBENTURE ",
    " DEBENTURES ",
    " NEXTSHARES ",
    " ETN ",
    " ETFS ",
    " SPAC ",
)
EXCLUDED_SYMBOL_SUFFIXES = (
    "W",
    "WS",
    "WT",
    "U",
    "R",
    "RT",
    "RIGHT",
    "P",
    "PR",
    "PRA",
    "PRB",
    "PRC",
    "PRD",
    "PRE",
    "PRF",
    "PRG",
    "PRH",
    "PRI",
    "PRJ",
    "PRK",
    "PRL",
    "PRM",
    "PRN",
    "PRO",
    "PRP",
    "PRQ",
    "PRS",
    "PRT",
    "PRU",
    "PRV",
    "PRW",
    "PRX",
    "PRY",
    "PRZ",
)


def normalize_ticker_for_yahoo(symbol: object) -> str:
    text = str(symbol or "").strip().upper()
    if not text:
        return ""
    text = text.replace("/", "-").replace(".", "-")
    return text


def fetch_symbol_directory(url: str) -> pd.DataFrame:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=30) as response:
        content = response.read().decode("utf-8", errors="ignore")

    lines = [
        line
        for line in content.splitlines()
        if line and not line.startswith("File Creation Time:")
    ]
    if not lines:
        return pd.DataFrame()
    return pd.read_csv(StringIO("\n".join(lines)), sep="|")


def load_cached_directory(filename: str) -> pd.DataFrame:
    path = SYMBOL_CACHE_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def save_cached_directory(filename: str, frame: pd.DataFrame) -> None:
    try:
        SYMBOL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        frame.to_csv(SYMBOL_CACHE_DIR / filename, index=False)
    except Exception:
        return


def load_directory_with_cache(url: str, filename: str) -> pd.DataFrame:
    cached_frame = load_cached_directory(filename)
    if not cached_frame.empty:
        return cached_frame

    try:
        frame = fetch_symbol_directory(url)
        if not frame.empty:
            save_cached_directory(filename, frame)
            return frame
    except Exception:
        pass
    return load_cached_directory(filename)


def symbol_has_excluded_suffix(symbol: str) -> bool:
    parts = symbol.replace(".", "-").split("-")
    if len(parts) < 2:
        return False
    suffix = parts[-1].upper()
    return suffix in EXCLUDED_SYMBOL_SUFFIXES


def name_has_excluded_term(name: str) -> bool:
    padded_name = f" {name.upper()} "
    return any(term in padded_name for term in EXCLUDED_NAME_TERMS)


def is_common_stock_row(row: pd.Series, source: str) -> tuple[bool, str]:
    symbol = str(row.get("Symbol") or row.get("ACT Symbol") or "").strip().upper()
    security_name = str(row.get("Security Name") or "").strip()
    if not symbol:
        return False, "missing symbol"
    if row.get("Test Issue") == "Y":
        return False, "test issue"
    if source == "nasdaq" and row.get("ETF") == "Y":
        return False, "ETF"
    if source == "other" and row.get("ETF") == "Y":
        return False, "ETF"
    if source == "other" and str(row.get("Exchange", "")).strip().upper() != "N":
        return False, "not NYSE"
    if str(row.get("NextShares", "")).strip().upper() == "Y":
        return False, "NextShares"
    if symbol_has_excluded_suffix(symbol):
        return False, "non-common suffix"
    if name_has_excluded_term(security_name):
        return False, "non-common security name"
    return True, ""


def directory_rows(frame: pd.DataFrame, source: str) -> list[dict]:
    rows: list[dict] = []
    for _, row in frame.iterrows():
        keep, reason = is_common_stock_row(row, source)
        symbol = str(row.get("Symbol") or row.get("ACT Symbol") or "").strip().upper()
        if not keep:
            rows.append(
                {
                    "symbol": symbol,
                    "ticker": normalize_ticker_for_yahoo(symbol),
                    "company": str(row.get("Security Name") or "").strip(),
                    "exchange": "Nasdaq" if source == "nasdaq" else "NYSE",
                    "keep": False,
                    "reason": reason,
                }
            )
            continue
        rows.append(
            {
                "symbol": symbol,
                "ticker": normalize_ticker_for_yahoo(symbol),
                "company": str(row.get("Security Name") or "").strip(),
                "exchange": "Nasdaq" if source == "nasdaq" else "NYSE",
                "keep": True,
                "reason": "",
            }
        )
    return rows


@lru_cache(maxsize=1)
def load_nyse_nasdaq_common_stock_rows() -> tuple[list[dict], dict]:
    nasdaq_frame = load_directory_with_cache(NASDAQ_LISTED_URL, "nasdaqlisted.csv")
    other_frame = load_directory_with_cache(OTHER_LISTED_URL, "otherlisted.csv")

    raw_rows = directory_rows(nasdaq_frame, "nasdaq") + directory_rows(other_frame, "other")
    seen: set[str] = set()
    kept_rows: list[dict] = []
    removed_count = 0
    removal_reasons: dict[str, int] = {}
    for row in raw_rows:
        ticker = row.get("ticker", "")
        if not row.get("keep"):
            removed_count += 1
            reason = row.get("reason") or "removed"
            removal_reasons[reason] = removal_reasons.get(reason, 0) + 1
            continue
        if not ticker or ticker in seen:
            removed_count += 1
            removal_reasons["duplicate"] = removal_reasons.get("duplicate", 0) + 1
            continue
        seen.add(ticker)
        kept_rows.append(row)

    meta = {
        "raw_symbols_loaded": len(raw_rows),
        "non_common_removed": removed_count,
        "final_common_stocks": len(kept_rows),
        "removal_reasons": removal_reasons,
        "source": "Nasdaq Trader symbol directories",
    }
    return kept_rows, meta


def load_nyse_nasdaq_common_stocks() -> list[str]:
    rows, _meta = load_nyse_nasdaq_common_stock_rows()
    return [row["ticker"] for row in rows]


def get_nyse_nasdaq_common_stock_meta() -> dict:
    _rows, meta = load_nyse_nasdaq_common_stock_rows()
    return meta.copy()


def get_symbol_metadata(ticker: str) -> dict:
    normalized = normalize_ticker_for_yahoo(ticker)
    rows, _meta = load_nyse_nasdaq_common_stock_rows()
    for row in rows:
        if row.get("ticker") == normalized:
            return {"company": row.get("company"), "exchange": row.get("exchange")}
    return {}


def get_symbol_metadata_map() -> dict[str, dict]:
    rows, _meta = load_nyse_nasdaq_common_stock_rows()
    return {row["ticker"]: {"company": row.get("company"), "exchange": row.get("exchange")} for row in rows}
