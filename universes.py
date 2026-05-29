from __future__ import annotations

from collections.abc import Iterable
from io import StringIO
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

from symbol_universe import EX_MAJOR_INDEXES_UNIVERSE, FULL_MARKET_UNIVERSE, get_nyse_nasdaq_common_stock_meta, load_nyse_nasdaq_common_stocks, normalize_ticker_for_yahoo


DATA_DIR = Path(__file__).resolve().parent / "data"
RUSSELL3000_FILE = DATA_DIR / "russell3000_universe.csv"
SP500_FILE = DATA_DIR / "sp500_universe.csv"
NASDAQ100_FILE = DATA_DIR / "nasdaq100_universe.csv"
RUSSELL2000_FILE = DATA_DIR / "russell2000_universe.csv"
MIDCAP400_FILE = DATA_DIR / "midcap400_universe.csv"

UNIVERSE_NAMES = ["SP500_FULL", "QQQ", "IWM", "MDY", FULL_MARKET_UNIVERSE, EX_MAJOR_INDEXES_UNIVERSE]
UNIVERSE_LABELS = {
    "SP500_FULL": "S&P 500 - Full",
    "QQQ": "Nasdaq 100 - Full",
    "IWM": "Russell 2000 - Full",
    "MDY": "S&P MidCap 400 - Full",
    FULL_MARKET_UNIVERSE: "All NYSE + Nasdaq Common Stocks",
    EX_MAJOR_INDEXES_UNIVERSE: "All NYSE + Nasdaq — Ex Major Indexes",
    "RUSSELL3000": "Russell 3000 - Full",
}
UNIVERSE_NOTES = {
    "SP500_FULL": "Large-cap U.S. companies. Usually cleaner, more liquid, and more stable.",
    "QQQ": "Large-cap growth and tech-heavy companies. Good for momentum and stronger trend setups.",
    "IWM": "Small-cap U.S. companies. More volatile and higher risk, but can find earlier-stage breakouts.",
    "MDY": "Mid-cap U.S. companies. A balance between large-cap quality and smaller-cap growth potential.",
    FULL_MARKET_UNIVERSE: "Scans all NYSE and Nasdaq listed common stocks after removing ETFs, funds, warrants, units, rights, preferreds, notes, bonds, trusts, test issues, and other non-common-stock securities. No price, volume, market cap, or trend prefilter is applied.",
    EX_MAJOR_INDEXES_UNIVERSE: "Scans NYSE and Nasdaq listed common stocks after removing stocks already covered by S&P 500, Nasdaq 100, Russell 2000, and S&P MidCap 400. Useful for finding smaller or less-followed names outside the main index universes.",
    "RUSSELL3000": "Large scan. Keep max tickers and liquidity filters on for best speed.",
}
_CACHE: dict[str, list[str]] = {}
_WARNINGS: dict[str, str] = {}

_FALLBACK_SP500 = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "GOOGL",
    "GOOG",
    "BRK.B",
    "LLY",
    "AVGO",
    "JPM",
    "XOM",
    "UNH",
    "V",
    "COST",
]

_FALLBACK_NASDAQ100 = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "GOOGL",
    "GOOG",
    "AVGO",
    "TSLA",
    "NFLX",
    "AMD",
    "COST",
    "ADBE",
    "INTU",
    "CSCO",
]

_FALLBACK_RUSSELL2000 = [
    "INSM",
    "CROX",
    "AAON",
    "SFM",
    "ONTO",
    "FN",
    "NOVT",
    "TMDX",
    "GIII",
    "ITRI",
    "FIVE",
    "WSC",
    "GKOS",
    "BCPC",
    "SLAB",
]

_FALLBACK_MIDCAP400 = [
    "BLD",
    "CASY",
    "CELH",
    "CART",
    "COHR",
    "DKS",
    "EME",
    "FIX",
    "FND",
    "GGG",
    "GWRE",
    "HUBB",
    "ITT",
    "MANH",
    "MEDP",
    "PEN",
    "PSTG",
    "RS",
    "SAIA",
    "WSM",
]

def get_universe_names() -> list[str]:
    return UNIVERSE_NAMES.copy()


def get_universe_label(name: str) -> str:
    normalized_name = str(name).strip().upper()
    return UNIVERSE_LABELS.get(normalized_name, normalized_name)


def get_universe_option_label(name: str) -> str:
    label = get_universe_label(name)
    count = get_universe_count(name)
    return f"{label} - {count:,} tickers" if count else label


def get_universe_note(name: str) -> str:
    normalized_name = str(name).strip().upper()
    return UNIVERSE_NOTES.get(normalized_name, "")


def get_universe_count(name: str) -> int:
    return len(get_universe(name))


def get_universe_meta(name: str) -> dict:
    normalized_name = str(name).strip().upper()
    if normalized_name in {FULL_MARKET_UNIVERSE, EX_MAJOR_INDEXES_UNIVERSE}:
        return get_nyse_nasdaq_common_stock_meta()
    return {"final_common_stocks": len(get_universe(normalized_name))}


def get_universe_warning(name: str) -> str:
    normalized_name = str(name).strip().upper()
    if normalized_name == "RUSSELL3000" and not RUSSELL3000_FILE.exists():
        return "Russell 3000 list not found. Add a local ticker CSV to enable this universe."
    if normalized_name in _WARNINGS:
        return _WARNINGS[normalized_name]
    label = get_universe_label(normalized_name)
    ticker_count = len(get_universe(normalized_name))
    if "S&P 500" in label and ticker_count < 100:
        return "This S&P 500 universe appears incomplete. Check the live source or local cache."
    if "Nasdaq 100" in label and ticker_count < 50:
        return "This Nasdaq 100 universe appears incomplete. Check the live source or local cache."
    if "Russell 2000" in label and ticker_count < 500:
        return "This Russell 2000 universe appears incomplete. Check the live source or local cache."
    return ""


def is_universe_available(name: str) -> bool:
    normalized_name = str(name).strip().upper()
    if normalized_name == "RUSSELL3000":
        return RUSSELL3000_FILE.exists()
    return bool(get_universe(normalized_name))


def is_large_universe(name: str) -> bool:
    return str(name).strip().upper() in {"RUSSELL3000", FULL_MARKET_UNIVERSE, EX_MAJOR_INDEXES_UNIVERSE}


def normalize_ticker(ticker: str) -> str:
    return str(ticker).strip().upper().replace(".", "-")


def dedupe_tickers(tickers: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    normalized_tickers: list[str] = []

    for ticker in tickers:
        normalized = normalize_ticker(ticker)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        normalized_tickers.append(normalized)

    return normalized_tickers


def load_local_ticker_file(path: Path) -> list[str]:
    if not path.exists():
        return []

    try:
        frame = pd.read_csv(path)
    except Exception:
        return []

    if frame.empty:
        return []

    for column_name in ("ticker", "symbol", "Ticker", "Symbol"):
        if column_name in frame.columns:
            return dedupe_tickers(frame[column_name].dropna().tolist())

    return dedupe_tickers(frame.iloc[:, 0].dropna().tolist())


def save_ticker_cache(path: Path, tickers: list[str]) -> None:
    try:
        path.parent.mkdir(exist_ok=True)
        pd.DataFrame({"ticker": tickers}).to_csv(path, index=False)
    except Exception:
        return


def load_wikipedia_tickers(url: str, column_name: str) -> list[str]:
    tables = pd.read_html(url, flavor="bs4", storage_options={"User-Agent": "Mozilla/5.0"})

    for table in tables:
        if column_name not in table.columns:
            continue
        return dedupe_tickers(table[column_name].tolist())

    return []


def load_wikipedia_tickers_any_column(url: str, column_names: list[str]) -> list[str]:
    tables = pd.read_html(url, flavor="bs4", storage_options={"User-Agent": "Mozilla/5.0"})

    for table in tables:
        for column_name in column_names:
            if column_name in table.columns:
                return dedupe_tickers(table[column_name].tolist())

    return []


def load_ishares_holdings_tickers(url: str) -> list[str]:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=20) as response:
        content = response.read().decode("utf-8-sig", errors="ignore")

    lines = content.splitlines()
    header_index = next((index for index, line in enumerate(lines) if line.startswith("Ticker,")), None)
    if header_index is None:
        return []

    frame = pd.read_csv(StringIO("\n".join(lines[header_index:])))
    if "Ticker" not in frame.columns:
        return []

    tickers = []
    for ticker in frame["Ticker"].dropna().tolist():
        normalized = normalize_ticker(ticker)
        if normalized and normalized not in {"-", "CASH", "USD"}:
            tickers.append(normalized)
    return dedupe_tickers(tickers)


def get_sp500_tickers() -> list[str]:
    cache_key = "SP500_FULL"
    if cache_key not in _CACHE:
        cached_tickers = load_local_ticker_file(SP500_FILE)
        if len(cached_tickers) >= 100:
            _CACHE[cache_key] = cached_tickers
            return _CACHE[cache_key]
        try:
            live_tickers = load_wikipedia_tickers(
                "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                "Symbol",
            )
            if len(live_tickers) >= 100:
                save_ticker_cache(SP500_FILE, live_tickers)
                _CACHE[cache_key] = live_tickers
            else:
                cached_tickers = load_local_ticker_file(SP500_FILE)
                _CACHE[cache_key] = cached_tickers if cached_tickers else dedupe_tickers(_FALLBACK_SP500)
                _WARNINGS[cache_key] = "Live S&P 500 source returned too few tickers. Using the local fallback list."
        except Exception:
            cached_tickers = load_local_ticker_file(SP500_FILE)
            _CACHE[cache_key] = cached_tickers if cached_tickers else dedupe_tickers(_FALLBACK_SP500)
            if cached_tickers:
                _WARNINGS[cache_key] = "Could not refresh the live S&P 500 list. Using the local cached list."
            else:
                _WARNINGS[cache_key] = "Could not refresh the live S&P 500 list and no full local cache was found."
    return _CACHE[cache_key]


def get_fast_large_cap_tickers() -> list[str]:
    cache_key = "SPY"
    if cache_key not in _CACHE:
        _CACHE[cache_key] = dedupe_tickers(_FALLBACK_SP500)
    return _CACHE[cache_key]


def get_nasdaq100_tickers() -> list[str]:
    cache_key = "QQQ"
    if cache_key not in _CACHE:
        cached_tickers = load_local_ticker_file(NASDAQ100_FILE)
        if len(cached_tickers) >= 50:
            _CACHE[cache_key] = cached_tickers
            return _CACHE[cache_key]
        try:
            live_tickers = load_wikipedia_tickers(
                "https://en.wikipedia.org/wiki/Nasdaq-100",
                "Ticker",
            )
            if len(live_tickers) >= 50:
                save_ticker_cache(NASDAQ100_FILE, live_tickers)
                _CACHE[cache_key] = live_tickers
            else:
                cached_tickers = load_local_ticker_file(NASDAQ100_FILE)
                _CACHE[cache_key] = cached_tickers if cached_tickers else dedupe_tickers(_FALLBACK_NASDAQ100)
                _WARNINGS[cache_key] = "Live Nasdaq 100 source returned too few tickers. Using the local fallback list."
        except Exception:
            cached_tickers = load_local_ticker_file(NASDAQ100_FILE)
            _CACHE[cache_key] = cached_tickers if cached_tickers else dedupe_tickers(_FALLBACK_NASDAQ100)
            if cached_tickers:
                _WARNINGS[cache_key] = "Could not refresh the live Nasdaq 100 list. Using the local cached list."
            else:
                _WARNINGS[cache_key] = "Could not refresh the live Nasdaq 100 list and no full local cache was found."
    return _CACHE[cache_key]


def get_russell2000_tickers() -> list[str]:
    cache_key = "IWM"
    if cache_key not in _CACHE:
        cached_tickers = load_local_ticker_file(RUSSELL2000_FILE)
        if len(cached_tickers) >= 500:
            _CACHE[cache_key] = cached_tickers
            return _CACHE[cache_key]
        try:
            live_tickers = load_ishares_holdings_tickers(
                "https://www.ishares.com/us/products/239710/ishares-russell-2000-etf/1467271812596.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund"
            )
            if len(live_tickers) >= 500:
                save_ticker_cache(RUSSELL2000_FILE, live_tickers)
                _CACHE[cache_key] = live_tickers
            else:
                cached_tickers = load_local_ticker_file(RUSSELL2000_FILE)
                _CACHE[cache_key] = cached_tickers if cached_tickers else dedupe_tickers(_FALLBACK_RUSSELL2000)
                _WARNINGS[cache_key] = "Live Russell 2000 source returned too few tickers. Using the local fallback list."
        except Exception:
            cached_tickers = load_local_ticker_file(RUSSELL2000_FILE)
            _CACHE[cache_key] = cached_tickers if cached_tickers else dedupe_tickers(_FALLBACK_RUSSELL2000)
            if cached_tickers:
                _WARNINGS[cache_key] = "Could not refresh the live Russell 2000 list. Using the local cached list."
            else:
                _WARNINGS[cache_key] = "Could not refresh the live Russell 2000 list and no full local cache was found."
    return _CACHE[cache_key]


def get_midcap400_tickers() -> list[str]:
    cache_key = "MDY"
    if cache_key not in _CACHE:
        cached_tickers = load_local_ticker_file(MIDCAP400_FILE)
        if len(cached_tickers) >= 100:
            _CACHE[cache_key] = cached_tickers
            return _CACHE[cache_key]
        try:
            tickers = load_wikipedia_tickers_any_column(
                "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
                ["Symbol", "Ticker symbol", "Ticker"],
            )
            if len(tickers) >= 100:
                save_ticker_cache(MIDCAP400_FILE, tickers)
                _CACHE[cache_key] = tickers
            else:
                cached_tickers = load_local_ticker_file(MIDCAP400_FILE)
                _CACHE[cache_key] = cached_tickers if cached_tickers else dedupe_tickers(_FALLBACK_MIDCAP400)
                _WARNINGS[cache_key] = "Live S&P MidCap 400 source returned too few tickers. Using the local fallback list."
        except Exception:
            cached_tickers = load_local_ticker_file(MIDCAP400_FILE)
            _CACHE[cache_key] = cached_tickers if cached_tickers else dedupe_tickers(_FALLBACK_MIDCAP400)
            if cached_tickers:
                _WARNINGS[cache_key] = "Could not refresh the live S&P MidCap 400 list. Using the local cached list."
            else:
                _WARNINGS[cache_key] = "Could not refresh the live S&P MidCap 400 list and no full local cache was found."
    return _CACHE[cache_key]


def get_russell3000_universe() -> list[str]:
    cache_key = "RUSSELL3000"
    if cache_key not in _CACHE:
        _CACHE[cache_key] = load_local_ticker_file(RUSSELL3000_FILE)
    return _CACHE[cache_key]


def get_nyse_nasdaq_common_universe() -> list[str]:
    cache_key = FULL_MARKET_UNIVERSE
    if cache_key not in _CACHE:
        _CACHE[cache_key] = load_nyse_nasdaq_common_stocks()
    return _CACHE[cache_key]


def get_nyse_nasdaq_ex_major_indexes_universe() -> list[str]:
    cache_key = EX_MAJOR_INDEXES_UNIVERSE
    if cache_key not in _CACHE:
        all_common = {normalize_ticker_for_yahoo(ticker) for ticker in get_nyse_nasdaq_common_universe()}
        major_index_tickers = (
            {normalize_ticker_for_yahoo(ticker) for ticker in get_sp500_tickers()}
            | {normalize_ticker_for_yahoo(ticker) for ticker in get_nasdaq100_tickers()}
            | {normalize_ticker_for_yahoo(ticker) for ticker in get_russell2000_tickers()}
            | {normalize_ticker_for_yahoo(ticker) for ticker in get_midcap400_tickers()}
        )
        _CACHE[cache_key] = sorted(all_common - major_index_tickers)
    return _CACHE[cache_key]


def get_universe(name: str) -> list[str]:
    normalized_name = str(name).strip().upper()

    if normalized_name == "SPY":
        return get_fast_large_cap_tickers()
    if normalized_name == "SP500_FULL":
        return get_sp500_tickers()
    if normalized_name == "QQQ":
        return get_nasdaq100_tickers()
    if normalized_name == "IWM":
        return get_russell2000_tickers()
    if normalized_name == "MDY":
        return get_midcap400_tickers()
    if normalized_name == FULL_MARKET_UNIVERSE:
        return get_nyse_nasdaq_common_universe()
    if normalized_name == EX_MAJOR_INDEXES_UNIVERSE:
        return get_nyse_nasdaq_ex_major_indexes_universe()
    if normalized_name == "RUSSELL3000":
        return get_russell3000_universe()

    return []


def get_ticker_memberships(ticker: str) -> list[str]:
    normalized_ticker = normalize_ticker(ticker)
    memberships: list[str] = []

    for universe_name in get_universe_names():
        if normalized_ticker in get_universe(universe_name):
            memberships.append(universe_name)

    return memberships
