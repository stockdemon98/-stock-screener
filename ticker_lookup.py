from __future__ import annotations

import json
import os
import ssl
from io import StringIO
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd
import streamlit as st

from universes import DATA_DIR, get_universe, get_universe_names, normalize_ticker


SP500_WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
VERIFY_EXTERNAL_SSL = os.environ.get("STOCK_SCREENER_VERIFY_EXTERNAL_SSL", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
COMPANY_SUFFIXES = {
    "Corp": "Corporation",
    "Inc": "Inc.",
    "Co": "Company",
    "Ltd": "Ltd.",
}


def _external_urlopen(request: Request, timeout: int = 10):
    if VERIFY_EXTERNAL_SSL:
        return urlopen(request, timeout=timeout)
    return urlopen(request, timeout=timeout, context=ssl._create_unverified_context())


def _clean_company_name(value: object) -> str:
    if pd.isna(value):
        return ""
    name = " ".join(str(value).strip().title().split())
    for suffix, replacement in COMPANY_SUFFIXES.items():
        if name.endswith(f" {suffix}"):
            return f"{name[:-(len(suffix) + 1)]} {replacement}"
    return name


def _read_local_names(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    try:
        frame = pd.read_csv(path)
    except Exception:
        return {}

    if frame.empty:
        return {}

    ticker_column = next((name for name in ("ticker", "symbol", "Ticker", "Symbol") if name in frame.columns), "")
    name_column = next(
        (
            name
            for name in ("company", "company_name", "name", "security", "Company", "Company Name", "Name", "Security")
            if name in frame.columns
        ),
        "",
    )
    if not ticker_column or not name_column:
        return {}

    names: dict[str, str] = {}
    for _, row in frame.iterrows():
        ticker = normalize_ticker(row.get(ticker_column, ""))
        company_name = _clean_company_name(row.get(name_column, ""))
        if ticker and company_name:
            names[ticker] = company_name
    return names


def _load_sp500_company_names() -> dict[str, str]:
    try:
        request = Request(SP500_WIKIPEDIA_URL, headers={"User-Agent": "Mozilla/5.0"})
        with _external_urlopen(request, timeout=10) as response:
            html = response.read().decode("utf-8", errors="replace")
        tables = pd.read_html(StringIO(html), flavor="bs4")
    except Exception:
        return {}

    for table in tables:
        if "Symbol" not in table.columns or "Security" not in table.columns:
            continue
        names: dict[str, str] = {}
        for _, row in table.iterrows():
            ticker = normalize_ticker(row.get("Symbol", ""))
            company_name = _clean_company_name(row.get("Security", ""))
            if ticker and company_name:
                names[ticker] = company_name
        return names
    return {}


def _load_sec_company_names() -> dict[str, str]:
    try:
        request = Request(
            SEC_COMPANY_TICKERS_URL,
            headers={"User-Agent": "stock-screener contact@example.com"},
        )
        with _external_urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return {}

    if not isinstance(payload, dict):
        return {}

    names: dict[str, str] = {}
    for company in payload.values():
        if not isinstance(company, dict):
            continue
        ticker = normalize_ticker(company.get("ticker", ""))
        company_name = _clean_company_name(company.get("title", ""))
        if ticker and company_name:
            names[ticker] = company_name
    return names


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def build_ticker_lookup() -> list[dict[str, str]]:
    company_names: dict[str, str] = {}
    company_names.update(_read_local_names(DATA_DIR / "sp500_universe.csv"))
    company_names.update(_load_sp500_company_names())
    sec_company_names = _load_sec_company_names()

    tickers: list[str] = []
    seen: set[str] = set()
    for universe_name in get_universe_names():
        for ticker in get_universe(universe_name):
            normalized = normalize_ticker(ticker)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            tickers.append(normalized)

    return [
        {
            "ticker": ticker,
            "name": sec_company_names.get(ticker) or company_names.get(ticker, ""),
            "label": format_ticker_option(ticker, sec_company_names.get(ticker) or company_names.get(ticker, "")),
        }
        for ticker in tickers
    ]


def format_ticker_option(ticker: str, company_name: str = "") -> str:
    normalized = normalize_ticker(ticker)
    clean_name = _clean_company_name(company_name)
    return f"{normalized} - {clean_name}" if clean_name else normalized


def ticker_from_option(option: str) -> str:
    option_text = str(option)
    for separator in (" - ", " \u2014 ", " \u00e2\u20ac\u201d "):
        if separator in option_text:
            return normalize_ticker(option_text.split(separator, 1)[0])
    return normalize_ticker(option_text)


def search_ticker_lookup(query: str, limit: int = 25) -> list[dict[str, str]]:
    normalized_query = str(query).strip().lower()
    if not normalized_query:
        return []

    starts_with_matches: list[dict[str, str]] = []
    contains_matches: list[dict[str, str]] = []
    for row in build_ticker_lookup():
        ticker = row["ticker"].lower()
        name = row["name"].lower()
        label = row["label"].lower()
        if ticker.startswith(normalized_query) or name.startswith(normalized_query):
            starts_with_matches.append(row)
        elif normalized_query in label:
            contains_matches.append(row)

    return (starts_with_matches + contains_matches)[:limit]
