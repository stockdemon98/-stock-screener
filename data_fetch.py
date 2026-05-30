from __future__ import annotations

import logging
from functools import lru_cache
import json
import os
import ssl
from datetime import date, datetime
from pathlib import Path
from math import isnan
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
import streamlit as st
import yfinance as yf

try:
    from curl_cffi import requests as curl_requests
except Exception:  # pragma: no cover - yfinance can still fall back to its defaults.
    curl_requests = None

from symbol_universe import FULL_MARKET_UNIVERSE

logging.getLogger("yfinance").setLevel(logging.CRITICAL)

SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
SEC_USER_AGENT = "StockScreener/1.0 contact@example.com"
SECTOR_WACC_FALLBACKS = {
    "Technology": 0.0975,
    "Healthcare": 0.0875,
    "Industrials": 0.0925,
    "Consumer Cyclical": 0.0950,
    "Consumer Defensive": 0.0775,
    "Financial Services": 0.0900,
    "Energy": 0.1000,
    "Utilities": 0.0725,
    "Real Estate": 0.0775,
    "Communication Services": 0.0875,
    "Basic Materials": 0.0950,
    "Default": 0.0900,
}
DEFAULT_RISK_FREE_RATE = 0.0425
DEFAULT_EQUITY_RISK_PREMIUM = 0.0500
DEFAULT_PRE_TAX_COST_OF_DEBT = 0.0600
DEFAULT_TAX_RATE = 0.2100
CHART_TIMEFRAME_FETCH_PERIODS = {
    "6M": "1y",
    "1Y": "2y",
    "2Y": "3y",
    "5Y": "5y",
}
SCANNER_CACHE_DIR = Path(__file__).resolve().parent / "data" / "scanner_cache"
SNAPSHOT_DIR = Path(__file__).resolve().parent / "data" / "snapshots"
FUNDAMENTALS_TTL_SECONDS = 12 * 60 * 60
FUNDAMENTAL_NA = "N/A"
VERIFY_EXTERNAL_SSL = os.environ.get("STOCK_SCREENER_VERIFY_EXTERNAL_SSL", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def external_urlopen(request: Request, timeout: int = 10):
    if VERIFY_EXTERNAL_SSL:
        return urlopen(request, timeout=timeout)
    return urlopen(request, timeout=timeout, context=ssl._create_unverified_context())


@lru_cache(maxsize=1)
def get_yfinance_session():
    if curl_requests is None:
        return None
    session = curl_requests.Session(impersonate="chrome")
    session.verify = VERIFY_EXTERNAL_SSL
    return session


def get_yfinance_ticker(ticker: str):
    session = get_yfinance_session()
    if session is None:
        return yf.Ticker(ticker)
    return yf.Ticker(ticker, session=session)


def normalize_yahoo_ticker(ticker: str) -> str:
    return str(ticker).strip().upper().replace(".", "-")


def safe_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        numeric_value = float(value)
        if isnan(numeric_value):
            return None
        return numeric_value
    except (TypeError, ValueError):
        return None


def safe_get_fundamental(info: dict, possible_keys: list[str] | tuple[str, ...], default: object = FUNDAMENTAL_NA) -> object:
    for key in possible_keys:
        value = (info or {}).get(key)
        if value is None:
            continue
        if isinstance(value, float) and pd.isna(value):
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return default


def calculate_percent_change_from_history(history: pd.DataFrame, lookback_days: int) -> float | None:
    if history is None or history.empty or "Close" not in history.columns:
        return None
    closes = history["Close"].dropna()
    if len(closes.index) <= lookback_days:
        return None
    current_close = safe_float(closes.iloc[-1])
    previous_close = safe_float(closes.iloc[-lookback_days - 1])
    if current_close is None or previous_close in (None, 0):
        return None
    return ((current_close / previous_close) - 1) * 100


def calculate_ytd_performance(history: pd.DataFrame) -> float | None:
    if history is None or history.empty or "Close" not in history.columns:
        return None
    frame = history.copy()
    if not isinstance(frame.index, pd.DatetimeIndex):
        try:
            frame.index = pd.to_datetime(frame.index)
        except Exception:
            return None
    closes = frame["Close"].dropna()
    if closes.empty:
        return None
    current_close = safe_float(closes.iloc[-1])
    current_year = closes.index[-1].year
    year_closes = closes.loc[closes.index.year == current_year]
    if year_closes.empty:
        return None
    first_close = safe_float(year_closes.iloc[0])
    if current_close is None or first_close in (None, 0):
        return None
    return ((current_close / first_close) - 1) * 100


def calculate_atr_percent(history: pd.DataFrame, length: int = 14) -> float | None:
    if history is None or history.empty or any(column not in history.columns for column in ("High", "Low", "Close")):
        return None
    frame = history[["High", "Low", "Close"]].dropna().copy()
    if len(frame.index) < length + 1:
        return None
    previous_close = frame["Close"].shift(1)
    true_range = pd.concat(
        [
            frame["High"] - frame["Low"],
            (frame["High"] - previous_close).abs(),
            (frame["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = safe_float(true_range.rolling(length, min_periods=length).mean().dropna().iloc[-1])
    close = safe_float(frame["Close"].iloc[-1])
    if atr is None or close in (None, 0):
        return None
    return (atr / close) * 100


def calculate_rsi(history: pd.DataFrame, length: int = 14) -> float | None:
    if history is None or history.empty or "Close" not in history.columns:
        return None
    closes = history["Close"].dropna()
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
    return safe_float(latest.iloc[-1]) if not latest.empty else None


def calculate_free_cash_flow_yield(free_cash_flow: object, market_cap: object) -> float | None:
    free_cash_flow_value = safe_float(free_cash_flow)
    market_cap_value = safe_float(market_cap)
    if free_cash_flow_value is None or market_cap_value in (None, 0):
        return None
    return free_cash_flow_value / market_cap_value


def calculate_ratio(numerator: object, denominator: object) -> float | None:
    numerator_value = safe_float(numerator)
    denominator_value = safe_float(denominator)
    if numerator_value is None or denominator_value in (None, 0):
        return None
    return numerator_value / denominator_value


def calculate_target_mean_upside(current_price: object, target_mean_price: object) -> float | None:
    current_price_value = safe_float(current_price)
    target_mean_value = safe_float(target_mean_price)
    if current_price_value in (None, 0) or target_mean_value is None:
        return None
    return ((target_mean_value / current_price_value) - 1) * 100


def calculate_distance_from_level(current_price: object, reference_level: object) -> float | None:
    current_price_value = safe_float(current_price)
    reference_level_value = safe_float(reference_level)
    if current_price_value is None or reference_level_value in (None, 0):
        return None
    return ((current_price_value / reference_level_value) - 1) * 100


def value_or_na(value: object) -> object:
    return FUNDAMENTAL_NA if value is None else value


def build_fundamental_quality_summary(snapshot: dict) -> dict:
    def score_positive(section_name: str, labels: tuple[str, ...], points: int) -> float:
        values = [safe_float(snapshot.get(section_name, {}).get(label)) for label in labels]
        values = [value for value in values if value is not None]
        if not values:
            return 0.0
        positive_count = sum(1 for value in values if value > 0)
        return points * (positive_count / len(values))

    def score_balance_sheet(points: int) -> float:
        section = snapshot.get("Balance Sheet", {})
        current_ratio = safe_float(section.get("Current Ratio"))
        debt_to_equity = safe_float(section.get("Debt / Equity"))
        score = 0.0
        checks = 0
        if current_ratio is not None:
            checks += 1
            if current_ratio >= 1:
                score += 1
        if debt_to_equity is not None:
            checks += 1
            normalized_debt = debt_to_equity / 100 if abs(debt_to_equity) > 20 else debt_to_equity
            if normalized_debt <= 1.5:
                score += 1
        return points * (score / checks) if checks else 0.0

    def score_valuation(points: int) -> float:
        section = snapshot.get("Valuation", {})
        values = [
            safe_float(section.get("Forward P/E")),
            safe_float(section.get("Price / Sales")),
            safe_float(section.get("EV / EBITDA")),
        ]
        values = [value for value in values if value is not None and value > 0]
        if not values:
            return 0.0
        favorable = sum(1 for value in values if value <= 30)
        return points * (favorable / len(values))

    score = 0.0
    income_statement_score = score_positive(
        "Income Statement - TTM",
        (
            "Revenue (TTM)",
            "Gross Profit (TTM)",
            "Cost of Revenue (TTM)",
            "R&D Expense (TTM)",
            "SG&A Expense (TTM)",
            "Operating Income (TTM)",
            "EBITDA (TTM)",
            "D&A (TTM)",
            "Interest Expense (TTM)",
            "Pretax Income (TTM)",
            "Income Tax Expense (TTM)",
            "Net Income (TTM)",
            "Basic EPS (TTM)",
            "Diluted EPS (TTM)",
        ),
        12,
    )
    margins_score = score_positive(
        "Margins / Returns",
        (
            "Gross Margin (TTM)",
            "Latest Quarter Gross Margin",
            "Operating Margin (TTM)",
            "Latest Quarter Operating Margin",
            "Net Margin (TTM)",
            "Latest Quarter Net Margin",
            "Return on Equity",
            "Return on Assets",
            "Free Cash Flow Margin (TTM)",
        ),
        18,
    )
    score += max(income_statement_score, margins_score)
    score += score_positive("Growth", ("Revenue Growth", "Earnings Growth", "Quarterly Revenue Growth YoY", "Quarterly Earnings Growth YoY"), 20)
    score += score_balance_sheet(20)
    score += score_positive("Cash Flow", ("Operating Cash Flow", "Free Cash Flow", "Levered Free Cash Flow"), 15)
    score += score_valuation(10)
    score += score_positive("Analyst / Target Info", ("Current Price vs Target Mean %",), 10)

    available_metrics = sum(
        1
        for section_name, section in snapshot.items()
        if not section_name.startswith("__")
        for value in section.values()
        if value not in (None, FUNDAMENTAL_NA, "")
    )
    if available_metrics < 8:
        label = "Insufficient Data"
    elif score >= 75:
        label = "Strong Fundamentals"
    elif score >= 60:
        label = "Good Fundamentals"
    elif score >= 40:
        label = "Mixed Fundamentals"
    else:
        label = "Weak Fundamentals"
    return {"score": round(score), "label": label, "available_metrics": available_metrics}


def clamp_ratio(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def fetch_sec_json(url: str) -> dict:
    request = Request(
        url,
        headers={
            "User-Agent": SEC_USER_AGENT,
            "Accept": "application/json",
        },
    )
    with external_urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


@lru_cache(maxsize=1)
def get_sec_ticker_cik_map() -> dict[str, str]:
    try:
        companies = fetch_sec_json(SEC_COMPANY_TICKERS_URL)
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError):
        return {}

    ticker_cik_map = {}
    for company in companies.values():
        ticker = str(company.get("ticker", "")).upper()
        cik = company.get("cik_str")
        if ticker and cik is not None:
            ticker_cik_map[ticker] = str(cik).zfill(10)
    return ticker_cik_map


@lru_cache(maxsize=256)
def get_latest_sec_filing(ticker: str) -> dict:
    normalized_ticker = ticker.strip().upper()
    if not normalized_ticker:
        return {}

    cik = get_sec_ticker_cik_map().get(normalized_ticker)
    if not cik:
        return {}

    try:
        submission = fetch_sec_json(SEC_SUBMISSIONS_URL.format(cik=cik))
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError):
        return {}

    recent_filings = submission.get("filings", {}).get("recent", {})
    forms = recent_filings.get("form", [])
    accession_numbers = recent_filings.get("accessionNumber", [])
    primary_documents = recent_filings.get("primaryDocument", [])
    filing_dates = recent_filings.get("filingDate", [])
    report_dates = recent_filings.get("reportDate", [])

    for index, form in enumerate(forms):
        if form not in {"10-Q", "10-K"}:
            continue
        try:
            accession_number = accession_numbers[index]
            primary_document = primary_documents[index]
            filing_date = filing_dates[index]
            report_date = report_dates[index] if index < len(report_dates) else None
        except IndexError:
            continue

        if not accession_number or not primary_document or not filing_date:
            continue

        accession_no_no_dashes = accession_number.replace("-", "")
        return {
            "form": form,
            "date": filing_date,
            "filing_date": filing_date,
            "report_date": report_date,
            "accession_number": accession_number,
            "cik": cik,
            "primary_document": primary_document,
            "url": SEC_ARCHIVES_URL.format(
                cik=str(int(cik)),
                accession=accession_no_no_dashes,
                document=primary_document,
            ),
        }

    return {}


def get_sec_fact_entries(company_facts: dict, tag_names: tuple[str, ...], units: tuple[str, ...]) -> list[dict]:
    us_gaap = (company_facts.get("facts") or {}).get("us-gaap") or {}
    entries: list[dict] = []
    for tag_name in tag_names:
        unit_map = (us_gaap.get(tag_name) or {}).get("units") or {}
        for unit_name in units:
            for entry in unit_map.get(unit_name, []):
                value = safe_float(entry.get("val"))
                if value is None:
                    continue
                entries.append(
                    {
                        **entry,
                        "tag": tag_name,
                        "unit": unit_name,
                        "value": value,
                    }
                )
    return entries


def sec_entry_sort_key(entry: dict) -> tuple:
    return (
        str(entry.get("end") or ""),
        str(entry.get("filed") or ""),
        str(entry.get("form") or ""),
    )


def is_quarterly_duration_fact(entry: dict) -> bool:
    frame = str(entry.get("frame") or "")
    form = str(entry.get("form") or "")
    return form in {"10-Q", "10-K"} and frame.startswith("CY") and "Q" in frame and not frame.endswith("I")


def latest_quarterly_facts(company_facts: dict, tag_names: tuple[str, ...], quarters: int = 4) -> list[dict]:
    entries = [
        entry
        for entry in get_sec_fact_entries(company_facts, tag_names, ("USD",))
        if is_quarterly_duration_fact(entry)
    ]
    entries.sort(key=sec_entry_sort_key, reverse=True)
    latest_by_period: dict[tuple[str, str], dict] = {}
    for entry in entries:
        period_key = (str(entry.get("start") or ""), str(entry.get("end") or ""))
        if period_key in latest_by_period:
            continue
        latest_by_period[period_key] = entry
        if len(latest_by_period) >= quarters:
            break
    return list(latest_by_period.values())


def sum_latest_quarters(company_facts: dict, tag_names: tuple[str, ...], quarters: int = 4) -> tuple[float | None, list[dict]]:
    facts = latest_quarterly_facts(company_facts, tag_names, quarters=quarters)
    if not facts:
        return None, []
    return sum(float(entry["value"]) for entry in facts), facts


def parse_sec_date(value: object) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date()
    except ValueError:
        return None


def sec_fact_duration_days(entry: dict) -> int | None:
    start_date = parse_sec_date(entry.get("start"))
    end_date = parse_sec_date(entry.get("end"))
    if start_date is None or end_date is None or end_date < start_date:
        return None
    return (end_date - start_date).days + 1


def get_sec_duration_facts(company_facts: dict, tag_names: tuple[str, ...], units: tuple[str, ...] = ("USD",)) -> list[dict]:
    return [
        entry
        for entry in get_sec_fact_entries(company_facts, tag_names, units)
        if str(entry.get("form") or "") in {"10-Q", "10-K"} and sec_fact_duration_days(entry) is not None
    ]


def latest_annual_fact(company_facts: dict, tag_names: tuple[str, ...], before_end: date | None = None, units: tuple[str, ...] = ("USD",)) -> dict | None:
    entries = []
    for entry in get_sec_duration_facts(company_facts, tag_names, units=units):
        end_date = parse_sec_date(entry.get("end"))
        duration = sec_fact_duration_days(entry)
        if (
            str(entry.get("form") or "") == "10-K"
            and duration is not None
            and 300 <= duration <= 400
            and (before_end is None or (end_date is not None and end_date < before_end))
        ):
            entries.append(entry)
    return sorted(entries, key=sec_entry_sort_key, reverse=True)[0] if entries else None


def latest_interim_ytd_fact(company_facts: dict, tag_names: tuple[str, ...], units: tuple[str, ...] = ("USD",)) -> dict | None:
    entries = [
        entry
        for entry in get_sec_duration_facts(company_facts, tag_names, units=units)
        if str(entry.get("form") or "") == "10-Q"
    ]
    if not entries:
        return None

    latest_end = max((parse_sec_date(entry.get("end")) for entry in entries), default=None)
    if latest_end is None:
        return None
    latest_entries = [entry for entry in entries if parse_sec_date(entry.get("end")) == latest_end]
    latest_entries.sort(
        key=lambda entry: (
            sec_fact_duration_days(entry) or 0,
            str(entry.get("filed") or ""),
            str(entry.get("frame") or ""),
        ),
        reverse=True,
    )
    return latest_entries[0] if latest_entries else None


def latest_standalone_quarter_fact(company_facts: dict, tag_names: tuple[str, ...], units: tuple[str, ...] = ("USD",)) -> dict | None:
    entries = [
        entry
        for entry in get_sec_duration_facts(company_facts, tag_names, units=units)
        if is_quarterly_duration_fact(entry)
    ]
    return sorted(entries, key=sec_entry_sort_key, reverse=True)[0] if entries else None


def comparable_prior_ytd_fact(company_facts: dict, tag_names: tuple[str, ...], current_ytd: dict, annual_fact: dict | None, units: tuple[str, ...] = ("USD",)) -> dict | None:
    current_end = parse_sec_date(current_ytd.get("end"))
    annual_end = parse_sec_date((annual_fact or {}).get("end"))
    current_duration = sec_fact_duration_days(current_ytd)
    if current_end is None or annual_end is None or current_duration is None:
        return None

    duration_gap = max(14, int(current_duration * 0.20))
    current_fp = str(current_ytd.get("fp") or "")
    entries = []
    for entry in get_sec_duration_facts(company_facts, tag_names, units=units):
        if str(entry.get("form") or "") != "10-Q":
            continue
        end_date = parse_sec_date(entry.get("end"))
        duration = sec_fact_duration_days(entry)
        if end_date is None or duration is None:
            continue
        if end_date >= annual_end:
            continue
        if abs(duration - current_duration) > duration_gap:
            continue
        entries.append(entry)

    if current_fp:
        matching_fp = [entry for entry in entries if str(entry.get("fp") or "") == current_fp]
        if matching_fp:
            entries = matching_fp
    if not entries:
        return None

    entries.sort(
        key=lambda entry: (
            abs((current_end - parse_sec_date(entry.get("end"))).days - 365),
            str(entry.get("filed") or ""),
        )
    )
    return entries[0]


def calculate_sec_ttm_fact(company_facts: dict, tag_names: tuple[str, ...], units: tuple[str, ...] = ("USD",)) -> tuple[float | None, list[dict]]:
    interim_ytd = latest_interim_ytd_fact(company_facts, tag_names, units=units)
    latest_annual = latest_annual_fact(company_facts, tag_names, units=units)

    if latest_annual is not None:
        annual_end = parse_sec_date(latest_annual.get("end"))
        interim_end = parse_sec_date((interim_ytd or {}).get("end"))
        if interim_ytd is None or (annual_end is not None and interim_end is not None and annual_end >= interim_end):
            return float(latest_annual["value"]), [latest_annual]

    if interim_ytd is not None:
        interim_end = parse_sec_date(interim_ytd.get("end"))
        prior_annual = latest_annual_fact(company_facts, tag_names, before_end=interim_end, units=units)
        prior_ytd = comparable_prior_ytd_fact(company_facts, tag_names, interim_ytd, prior_annual, units=units)
        if prior_annual is not None and prior_ytd is not None:
            return (
                float(interim_ytd["value"]) + float(prior_annual["value"]) - float(prior_ytd["value"]),
                [interim_ytd, prior_annual, prior_ytd],
            )

    if units == ("USD",):
        return sum_latest_quarters(company_facts, tag_names)
    return None, []


def latest_quarter_value(company_facts: dict, tag_names: tuple[str, ...], units: tuple[str, ...] = ("USD",)) -> tuple[float | None, dict | None]:
    fact = latest_standalone_quarter_fact(company_facts, tag_names, units=units)
    return (float(fact["value"]), fact) if fact else (None, None)


def latest_sec_fact(company_facts: dict, tag_names: tuple[str, ...], units: tuple[str, ...] = ("USD",)) -> dict | None:
    entries = get_sec_fact_entries(company_facts, tag_names, units)
    if not entries:
        return None
    return sorted(entries, key=sec_entry_sort_key, reverse=True)[0]


def get_sec_fact_value(company_facts: dict, tag_names: tuple[str, ...], units: tuple[str, ...] = ("USD",)) -> tuple[float | None, dict | None]:
    entry = latest_sec_fact(company_facts, tag_names, units=units)
    return (float(entry["value"]), entry) if entry else (None, None)


@st.cache_data(ttl=12 * 60 * 60, show_spinner=False)
def get_sec_company_facts(ticker: str) -> dict:
    normalized_ticker = ticker.strip().upper()
    if not normalized_ticker:
        return {}
    cik = get_sec_ticker_cik_map().get(normalized_ticker)
    if not cik:
        return {}
    try:
        return fetch_sec_json(SEC_COMPANY_FACTS_URL.format(cik=cik))
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError):
        return {}


@st.cache_data(ttl=12 * 60 * 60, show_spinner=False)
def get_sec_dcf_inputs(ticker: str) -> dict:
    normalized_ticker = ticker.strip().upper()
    company_facts = get_sec_company_facts(normalized_ticker)
    if not company_facts:
        return {"available": False, "source": "SEC company facts unavailable."}

    revenue_tags = (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    )
    operating_income_tags = (
        "OperatingIncomeLoss",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    )
    gross_profit_tags = ("GrossProfit",)
    cost_of_revenue_tags = (
        "CostOfRevenue",
        "CostOfGoodsAndServicesSold",
        "CostOfGoodsSold",
        "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization",
        "CostOfGoodsAndServiceIncludingDepreciationDepletionAndAmortization",
    )
    rd_tags = ("ResearchAndDevelopmentExpense", "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost")
    sga_tags = ("SellingGeneralAndAdministrativeExpense", "SellingAndMarketingExpense")
    net_income_tags = (
        "NetIncomeLoss",
        "ProfitLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
    )
    pretax_income_tags = (
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxes",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    )
    tax_expense_tags = ("IncomeTaxExpenseBenefit",)
    interest_expense_tags = ("InterestExpenseNonOperating", "InterestExpense")
    depreciation_tags = (
        "DepreciationDepletionAndAmortization",
        "DepreciationDepletionAndAmortizationExpense",
        "DepreciationAndAmortization",
    )
    eps_basic_tags = ("EarningsPerShareBasic",)
    eps_diluted_tags = ("EarningsPerShareDiluted",)

    revenue_ttm, revenue_facts = calculate_sec_ttm_fact(company_facts, revenue_tags)
    operating_income_ttm, operating_income_facts = calculate_sec_ttm_fact(company_facts, operating_income_tags)
    gross_profit_ttm, gross_profit_facts = calculate_sec_ttm_fact(company_facts, gross_profit_tags)
    cost_of_revenue_ttm, cost_of_revenue_facts = calculate_sec_ttm_fact(company_facts, cost_of_revenue_tags)
    rd_ttm, rd_facts = calculate_sec_ttm_fact(company_facts, rd_tags)
    sga_ttm, sga_facts = calculate_sec_ttm_fact(company_facts, sga_tags)
    net_income_ttm, net_income_facts = calculate_sec_ttm_fact(company_facts, net_income_tags)
    pretax_income_ttm, pretax_income_facts = calculate_sec_ttm_fact(company_facts, pretax_income_tags)
    tax_expense_ttm, tax_expense_facts = calculate_sec_ttm_fact(company_facts, tax_expense_tags)
    interest_expense_ttm, interest_expense_facts = calculate_sec_ttm_fact(company_facts, interest_expense_tags)
    depreciation_ttm, depreciation_facts = calculate_sec_ttm_fact(company_facts, depreciation_tags)
    eps_basic_ttm, eps_basic_facts = calculate_sec_ttm_fact(company_facts, eps_basic_tags, units=("USD/shares",))
    eps_diluted_ttm, eps_diluted_facts = calculate_sec_ttm_fact(company_facts, eps_diluted_tags, units=("USD/shares",))
    operating_cash_flow_ttm, _operating_cash_flow_facts = calculate_sec_ttm_fact(
        company_facts,
        ("NetCashProvidedByUsedInOperatingActivities",),
    )
    capex_ttm, _capex_facts = calculate_sec_ttm_fact(
        company_facts,
        ("PaymentsToAcquirePropertyPlantAndEquipment", "CapitalExpenditures"),
    )
    if gross_profit_ttm is None and revenue_ttm is not None and cost_of_revenue_ttm is not None:
        gross_profit_ttm = revenue_ttm - cost_of_revenue_ttm

    latest_quarter_revenue, latest_quarter_revenue_fact = latest_quarter_value(company_facts, revenue_tags)
    latest_quarter_operating_income, latest_quarter_operating_income_fact = latest_quarter_value(
        company_facts,
        operating_income_tags,
    )
    latest_quarter_gross_profit, latest_quarter_gross_profit_fact = latest_quarter_value(company_facts, gross_profit_tags)
    latest_quarter_cost_of_revenue, latest_quarter_cost_of_revenue_fact = latest_quarter_value(
        company_facts,
        cost_of_revenue_tags,
    )
    latest_quarter_rd, _latest_quarter_rd_fact = latest_quarter_value(company_facts, rd_tags)
    latest_quarter_sga, _latest_quarter_sga_fact = latest_quarter_value(company_facts, sga_tags)
    latest_quarter_net_income, _latest_quarter_net_income_fact = latest_quarter_value(company_facts, net_income_tags)
    latest_quarter_pretax_income, _latest_quarter_pretax_income_fact = latest_quarter_value(company_facts, pretax_income_tags)
    latest_quarter_tax_expense, _latest_quarter_tax_expense_fact = latest_quarter_value(company_facts, tax_expense_tags)
    latest_quarter_interest_expense, _latest_quarter_interest_expense_fact = latest_quarter_value(company_facts, interest_expense_tags)
    latest_quarter_eps_basic, _latest_quarter_eps_basic_fact = latest_quarter_value(company_facts, eps_basic_tags, units=("USD/shares",))
    latest_quarter_eps_diluted, _latest_quarter_eps_diluted_fact = latest_quarter_value(company_facts, eps_diluted_tags, units=("USD/shares",))
    if (
        latest_quarter_gross_profit is None
        and latest_quarter_revenue is not None
        and latest_quarter_cost_of_revenue is not None
    ):
        latest_quarter_gross_profit = latest_quarter_revenue - latest_quarter_cost_of_revenue
    cash, cash_fact = get_sec_fact_value(
        company_facts,
        (
            "CashAndCashEquivalentsAtCarryingValue",
            "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        ),
    )
    short_debt, _short_debt_fact = get_sec_fact_value(
        company_facts,
        (
            "ShortTermBorrowings",
            "ShortTermDebt",
            "LongTermDebtAndFinanceLeaseObligationsCurrent",
            "LongTermDebtCurrent",
        ),
    )
    long_debt, long_debt_fact = get_sec_fact_value(
        company_facts,
        (
            "LongTermDebtNoncurrent",
            "LongTermDebtAndFinanceLeaseObligationsNoncurrent",
        ),
    )
    total_debt, total_debt_fact = get_sec_fact_value(
        company_facts,
        (
            "LongTermDebtAndFinanceLeaseObligations",
            "LongTermDebt",
            "DebtAndFinanceLeaseObligations",
        ),
    )
    if total_debt is None:
        total_debt = (short_debt or 0.0) + (long_debt or 0.0)
        total_debt_fact = long_debt_fact

    shares, shares_fact = get_sec_fact_value(
        company_facts,
        ("EntityCommonStockSharesOutstanding", "CommonStocksIncludingAdditionalPaidInCapitalMember"),
        units=("shares",),
    )
    diluted_shares, diluted_shares_fact = get_sec_fact_value(
        company_facts,
        ("WeightedAverageNumberOfDilutedSharesOutstanding", "WeightedAverageNumberOfSharesOutstandingDiluted"),
        units=("shares",),
    )
    if shares is None:
        shares = diluted_shares
        shares_fact = diluted_shares_fact
    minority_interest, minority_fact = get_sec_fact_value(
        company_facts,
        ("MinorityInterest", "NoncontrollingInterestInConsolidatedEntity"),
    )
    investments, investments_fact = get_sec_fact_value(
        company_facts,
        ("ShortTermInvestments", "MarketableSecuritiesCurrent", "AvailableForSaleSecuritiesCurrent"),
    )
    total_assets, total_assets_fact = get_sec_fact_value(company_facts, ("Assets",))
    current_assets, _current_assets_fact = get_sec_fact_value(company_facts, ("AssetsCurrent",))
    total_liabilities, _total_liabilities_fact = get_sec_fact_value(company_facts, ("Liabilities",))
    current_liabilities, _current_liabilities_fact = get_sec_fact_value(company_facts, ("LiabilitiesCurrent",))
    stockholders_equity, stockholders_equity_fact = get_sec_fact_value(
        company_facts,
        ("StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
    )
    retained_earnings, _retained_earnings_fact = get_sec_fact_value(company_facts, ("RetainedEarningsAccumulatedDeficit",))
    inventory, _inventory_fact = get_sec_fact_value(company_facts, ("InventoryNet", "InventoryFinishedGoodsNetOfReserves"))
    receivables, _receivables_fact = get_sec_fact_value(
        company_facts,
        ("AccountsReceivableNetCurrent", "ReceivablesNetCurrent"),
    )
    payables, _payables_fact = get_sec_fact_value(company_facts, ("AccountsPayableCurrent", "AccountsPayableTradeCurrent"))
    free_cash_flow = None
    if operating_cash_flow_ttm is not None:
        free_cash_flow = operating_cash_flow_ttm - (capex_ttm or 0.0)

    ebitda_ttm = None
    if operating_income_ttm is not None and depreciation_ttm is not None:
        ebitda_ttm = operating_income_ttm + depreciation_ttm
    operating_margin = None
    if revenue_ttm not in (None, 0) and operating_income_ttm is not None:
        operating_margin = operating_income_ttm / revenue_ttm
    gross_margin = None
    if revenue_ttm not in (None, 0) and gross_profit_ttm is not None:
        gross_margin = gross_profit_ttm / revenue_ttm
    latest_quarter_operating_margin = None
    if latest_quarter_revenue not in (None, 0) and latest_quarter_operating_income is not None:
        latest_quarter_operating_margin = latest_quarter_operating_income / latest_quarter_revenue
    latest_quarter_gross_margin = None
    if latest_quarter_revenue not in (None, 0) and latest_quarter_gross_profit is not None:
        latest_quarter_gross_margin = latest_quarter_gross_profit / latest_quarter_revenue
    net_margin = None
    if revenue_ttm not in (None, 0) and net_income_ttm is not None:
        net_margin = net_income_ttm / revenue_ttm
    latest_quarter_net_margin = None
    if latest_quarter_revenue not in (None, 0) and latest_quarter_net_income is not None:
        latest_quarter_net_margin = latest_quarter_net_income / latest_quarter_revenue
    effective_tax_rate = None
    if pretax_income_ttm not in (None, 0) and tax_expense_ttm is not None:
        effective_tax_rate = tax_expense_ttm / pretax_income_ttm
    current_ratio = None
    if current_assets is not None and current_liabilities not in (None, 0):
        current_ratio = current_assets / current_liabilities
    debt_to_equity = None
    if total_debt is not None and stockholders_equity not in (None, 0):
        debt_to_equity = total_debt / stockholders_equity
    return_on_assets = None
    if net_income_ttm is not None and total_assets not in (None, 0):
        return_on_assets = net_income_ttm / total_assets
    return_on_equity = None
    if net_income_ttm is not None and stockholders_equity not in (None, 0):
        return_on_equity = net_income_ttm / stockholders_equity
    fcf_margin = None
    if revenue_ttm not in (None, 0) and free_cash_flow is not None:
        fcf_margin = free_cash_flow / revenue_ttm

    source_entry = next(
        (
            entry
            for entry in (
                revenue_facts
                + operating_income_facts
                + gross_profit_facts
                + cost_of_revenue_facts
                + net_income_facts
                + pretax_income_facts
                + tax_expense_facts
                + rd_facts
                + sga_facts
                + interest_expense_facts
                + depreciation_facts
                + eps_basic_facts
                + eps_diluted_facts
            )
            if entry
        ),
        latest_quarter_revenue_fact
        or latest_quarter_operating_income_fact
        or latest_quarter_gross_profit_fact
        or latest_quarter_cost_of_revenue_fact
        or cash_fact
        or total_debt_fact
        or total_assets_fact
        or stockholders_equity_fact
        or shares_fact
        or minority_fact
        or investments_fact,
    )
    source_periods = sorted(
        {
            str(entry.get("end"))
            for entry in (
                revenue_facts
                + operating_income_facts
                + gross_profit_facts
                + cost_of_revenue_facts
                + net_income_facts
                + pretax_income_facts
                + tax_expense_facts
                + rd_facts
                + sga_facts
                + interest_expense_facts
                + depreciation_facts
                + eps_basic_facts
                + eps_diluted_facts
            )
            if entry.get("end")
        },
        reverse=True,
    )
    return {
        "available": any(value is not None for value in (revenue_ttm, operating_income_ttm, cash, total_debt, shares)),
        "source": "SEC companyfacts XBRL",
        "cik": company_facts.get("cik"),
        "entity_name": company_facts.get("entityName"),
        "latest_form": source_entry.get("form") if source_entry else None,
        "latest_filed": source_entry.get("filed") if source_entry else None,
        "latest_period": source_entry.get("end") if source_entry else None,
        "source_periods": source_periods[:4],
        "totalRevenue": revenue_ttm,
        "grossProfit": gross_profit_ttm,
        "costOfRevenue": cost_of_revenue_ttm,
        "grossMargins": gross_margin,
        "researchAndDevelopment": rd_ttm,
        "sellingGeneralAdministrative": sga_ttm,
        "operatingIncome": operating_income_ttm,
        "operatingMargins": operating_margin,
        "ebitda": ebitda_ttm,
        "depreciationAmortization": depreciation_ttm,
        "interestExpense": interest_expense_ttm,
        "pretaxIncome": pretax_income_ttm,
        "incomeTaxExpense": tax_expense_ttm,
        "effectiveTaxRate": effective_tax_rate,
        "netIncome": net_income_ttm,
        "profitMargins": net_margin,
        "basicEps": eps_basic_ttm,
        "dilutedEps": eps_diluted_ttm,
        "latestQuarterRevenue": latest_quarter_revenue,
        "latestQuarterGrossProfit": latest_quarter_gross_profit,
        "latestQuarterCostOfRevenue": latest_quarter_cost_of_revenue,
        "latestQuarterGrossMargins": latest_quarter_gross_margin,
        "latestQuarterResearchAndDevelopment": latest_quarter_rd,
        "latestQuarterSellingGeneralAdministrative": latest_quarter_sga,
        "latestQuarterOperatingIncome": latest_quarter_operating_income,
        "latestQuarterOperatingMargins": latest_quarter_operating_margin,
        "latestQuarterInterestExpense": latest_quarter_interest_expense,
        "latestQuarterPretaxIncome": latest_quarter_pretax_income,
        "latestQuarterIncomeTaxExpense": latest_quarter_tax_expense,
        "latestQuarterNetIncome": latest_quarter_net_income,
        "latestQuarterProfitMargins": latest_quarter_net_margin,
        "latestQuarterBasicEps": latest_quarter_eps_basic,
        "latestQuarterDilutedEps": latest_quarter_eps_diluted,
        "operatingCashflow": operating_cash_flow_ttm,
        "capitalExpenditures": capex_ttm,
        "freeCashflow": free_cash_flow,
        "freeCashFlowMargin": fcf_margin,
        "totalCash": cash,
        "shortTermDebt": short_debt,
        "longTermDebt": long_debt,
        "totalDebt": total_debt,
        "totalAssets": total_assets,
        "currentAssets": current_assets,
        "totalLiabilities": total_liabilities,
        "currentLiabilities": current_liabilities,
        "stockholdersEquity": stockholders_equity,
        "retainedEarnings": retained_earnings,
        "inventory": inventory,
        "accountsReceivable": receivables,
        "accountsPayable": payables,
        "currentRatio": current_ratio,
        "debtToEquity": debt_to_equity,
        "returnOnAssets": return_on_assets,
        "returnOnEquity": return_on_equity,
        "sharesOutstanding": shares,
        "minorityInterest": minority_interest,
        "totalInvestments": investments,
    }


def clean_history(history: pd.DataFrame | None) -> pd.DataFrame:
    if history is None or history.empty:
        return pd.DataFrame()

    columns = ["Open", "High", "Low", "Close", "Volume"]
    available_columns = [column for column in columns if column in history.columns]
    if len(available_columns) < len(columns):
        return pd.DataFrame()

    cleaned = history[columns].dropna().copy()
    if isinstance(cleaned.index, pd.DatetimeIndex):
        cleaned.index = cleaned.index.tz_localize(None)
    return cleaned


def build_stock_data_from_history(
    ticker: str,
    history: pd.DataFrame | None,
    display_ticker: str | None = None,
    min_history_bars: int = 30,
) -> dict:
    normalized_ticker = normalize_yahoo_ticker(ticker)
    display_ticker = str(display_ticker or normalized_ticker).strip().upper()
    history = clean_history(history)
    if history.empty or len(history.index) < min_history_bars:
        return {}

    closes = history["Close"].dropna()
    volumes = history["Volume"].dropna()

    if len(closes.index) < 20 or len(volumes.index) < 20:
        return {}

    last_price = safe_float(closes.iloc[-1])
    previous_close = safe_float(closes.iloc[-2])
    last_volume = safe_float(volumes.iloc[-1])
    avg_volume_20 = safe_float(volumes.tail(20).mean())

    if last_price is None or previous_close is None or avg_volume_20 in (None, 0):
        return {}

    change_pct = ((last_price - previous_close) / previous_close) * 100 if previous_close else None
    volume_ratio = (last_volume / avg_volume_20) if last_volume is not None and avg_volume_20 else None
    addv_20 = last_price * avg_volume_20
    high_52w = safe_float(history["High"].tail(252).max())
    low_52w = safe_float(history["Low"].tail(252).min())

    if change_pct is None or volume_ratio is None:
        return {}

    return {
        "ticker": display_ticker,
        "yahoo_ticker": normalized_ticker,
        "history": history,
        "price": last_price,
        "change_pct": change_pct,
        "volume": last_volume,
        "avg_volume_20": avg_volume_20,
        "volume_ratio": volume_ratio,
        "addv_20": addv_20,
        "high_52w": high_52w,
        "low_52w": low_52w,
    }


@lru_cache(maxsize=1)
def get_benchmark_history() -> pd.DataFrame:
    try:
        history = get_yfinance_ticker("SPY").history(period="5y", interval="1d", auto_adjust=False)
    except Exception:
        return pd.DataFrame()

    return clean_history(history)


@st.cache_data(ttl=20 * 60, show_spinner=False)
def get_market_regime_histories() -> dict[str, pd.DataFrame]:
    histories: dict[str, pd.DataFrame] = {}
    for ticker in ("SPY", "QQQ", "IWM"):
        try:
            history = get_yfinance_ticker(ticker).history(period="2y", interval="1d", auto_adjust=False)
        except Exception:
            histories[ticker] = pd.DataFrame()
            continue
        histories[ticker] = clean_history(history)
    return histories


@lru_cache(maxsize=256)
def get_stock_data(ticker: str) -> dict:
    normalized_ticker = normalize_yahoo_ticker(ticker)
    try:
        history = get_yfinance_ticker(normalized_ticker).history(period="5y", interval="1d", auto_adjust=False)
    except Exception:
        return {}

    return build_stock_data_from_history(normalized_ticker, history)


def build_extended_hours_quote(info: dict) -> dict:
    if not info:
        return {}

    regular_price = safe_float(info.get("regularMarketPrice") or info.get("currentPrice"))
    previous_close = safe_float(info.get("regularMarketPreviousClose") or info.get("previousClose"))
    market_state = str(info.get("marketState") or "").upper()

    candidates = [
        {
            "label": "After Hours",
            "price": safe_float(info.get("postMarketPrice")),
            "change": safe_float(info.get("postMarketChange")),
            "change_pct": safe_float(info.get("postMarketChangePercent")),
        },
        {
            "label": "Pre-market",
            "price": safe_float(info.get("preMarketPrice")),
            "change": safe_float(info.get("preMarketChange")),
            "change_pct": safe_float(info.get("preMarketChangePercent")),
        },
    ]
    if market_state.startswith("PRE"):
        candidates = [candidates[1], candidates[0]]

    quote = next((candidate for candidate in candidates if candidate["price"] is not None), None)
    if quote is None:
        return {}

    if quote["change"] is None and regular_price is not None:
        quote["change"] = quote["price"] - regular_price
    if quote["change_pct"] is None and regular_price not in (None, 0):
        quote["change_pct"] = ((quote["price"] / regular_price) - 1) * 100

    return {
        **quote,
        "regular_price": regular_price,
        "previous_close": previous_close,
        "market_state": market_state,
    }


@st.cache_data(ttl=60, show_spinner=False)
def get_extended_hours_quote(ticker: str) -> dict:
    normalized_ticker = normalize_yahoo_ticker(ticker)
    if not normalized_ticker:
        return {}
    try:
        info = get_yfinance_ticker(normalized_ticker).info or {}
    except Exception:
        return {}
    return build_extended_hours_quote(info)


def get_downloaded_ticker_history(downloaded_history: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if downloaded_history.empty:
        return pd.DataFrame()

    normalized_ticker = normalize_yahoo_ticker(ticker)
    if not isinstance(downloaded_history.columns, pd.MultiIndex):
        return downloaded_history

    if normalized_ticker not in downloaded_history.columns.get_level_values(0):
        return pd.DataFrame()

    return downloaded_history[normalized_ticker]


def chunk_tickers(tickers: tuple[str, ...], chunk_size: int = 75) -> list[tuple[str, ...]]:
    return [tickers[index:index + chunk_size] for index in range(0, len(tickers), chunk_size)]


@st.cache_data(ttl=20 * 60, show_spinner=False)
def get_batch_stock_data_chunk_detailed(
    tickers: tuple[str, ...],
    period: str = "5y",
    min_history_bars: int = 30,
) -> dict:
    display_to_yahoo = {
        str(ticker).strip().upper(): normalize_yahoo_ticker(ticker)
        for ticker in tickers
        if str(ticker).strip()
    }
    yahoo_tickers = tuple(dict.fromkeys(display_to_yahoo.values()))
    if not yahoo_tickers:
        return {"data": {}, "failures": {}}

    try:
        download_kwargs = {
            "period": period,
            "interval": "1d",
            "auto_adjust": False,
            "group_by": "ticker",
            "threads": True,
            "progress": False,
            "timeout": 30,
        }
        session = get_yfinance_session()
        if session is not None:
            download_kwargs["session"] = session
        downloaded_history = yf.download(list(yahoo_tickers), **download_kwargs)
    except Exception:
        return {"data": {}, "failures": {ticker: "failed/no data" for ticker in display_to_yahoo}}

    if downloaded_history is None or downloaded_history.empty:
        return {"data": {}, "failures": {ticker: "failed/no data" for ticker in display_to_yahoo}}

    batch_data: dict[str, dict] = {}
    failures: dict[str, str] = {}
    for display_ticker, yahoo_ticker in display_to_yahoo.items():
        ticker_history = get_downloaded_ticker_history(downloaded_history, yahoo_ticker)
        if ticker_history.empty:
            failures[display_ticker] = "failed/no data"
            continue
        cleaned_history = clean_history(ticker_history)
        if cleaned_history.empty:
            failures[display_ticker] = "invalid OHLCV"
            continue
        if len(cleaned_history.index) < min_history_bars:
            failures[display_ticker] = "not enough history"
            continue
        stock_data = build_stock_data_from_history(
            yahoo_ticker,
            cleaned_history,
            display_ticker=display_ticker,
            min_history_bars=min_history_bars,
        )
        if stock_data:
            batch_data[display_ticker] = stock_data
        else:
            failures[display_ticker] = "failed/no data"
    return {"data": batch_data, "failures": failures}


def get_batch_stock_data_chunk(tickers: tuple[str, ...], period: str = "5y") -> dict[str, dict]:
    return get_batch_stock_data_chunk_detailed(tickers, period=period, min_history_bars=30).get("data", {})


def get_batch_stock_data(tickers: tuple[str, ...], period: str = "5y") -> dict[str, dict]:
    normalized_tickers = tuple(dict.fromkeys(str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()))
    batch_data: dict[str, dict] = {}
    for ticker_chunk in chunk_tickers(normalized_tickers):
        batch_data.update(get_batch_stock_data_chunk(ticker_chunk, period=period))
    return batch_data


def get_scanner_snapshot_base_path(universe_name: str) -> Path:
    if str(universe_name).strip().upper() == FULL_MARKET_UNIVERSE:
        return SNAPSHOT_DIR / "latest_nyse_nasdaq_common_scan"
    safe_name = "".join(character if character.isalnum() else "_" for character in universe_name.lower()).strip("_")
    return SCANNER_CACHE_DIR / safe_name


def save_scanner_snapshot(universe_name: str, snapshot_df: pd.DataFrame, meta: dict) -> dict:
    SCANNER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    base_path = get_scanner_snapshot_base_path(universe_name)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    saved_meta = {
        **meta,
        "universe_name": universe_name,
        "last_updated": datetime.now().isoformat(timespec="seconds"),
    }

    try:
        snapshot_df.to_parquet(base_path.with_suffix(".parquet"), index=False)
        saved_meta["format"] = "parquet"
        saved_meta["snapshot_path"] = str(base_path.with_suffix(".parquet"))
    except Exception:
        if str(universe_name).strip().upper() == FULL_MARKET_UNIVERSE:
            snapshot_df.to_csv(base_path.with_suffix(".csv"), index=False)
            saved_meta["format"] = "csv"
            saved_meta["snapshot_path"] = str(base_path.with_suffix(".csv"))
        else:
            try:
                snapshot_df.to_pickle(base_path.with_suffix(".pkl"))
                saved_meta["format"] = "pickle"
                saved_meta["snapshot_path"] = str(base_path.with_suffix(".pkl"))
            except Exception:
                snapshot_df.to_csv(base_path.with_suffix(".csv"), index=False)
                saved_meta["format"] = "csv"
                saved_meta["snapshot_path"] = str(base_path.with_suffix(".csv"))

    base_path.with_suffix(".json").write_text(json.dumps(saved_meta, indent=2), encoding="utf-8")
    return saved_meta


def load_scanner_snapshot(universe_name: str) -> tuple[pd.DataFrame, dict]:
    base_path = get_scanner_snapshot_base_path(universe_name)
    meta_path = base_path.with_suffix(".json")
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}

    for suffix, reader in (
        (".parquet", pd.read_parquet),
        (".pkl", pd.read_pickle),
        (".csv", pd.read_csv),
    ):
        snapshot_path = base_path.with_suffix(suffix)
        if not snapshot_path.exists():
            continue
        try:
            return reader(snapshot_path), meta
        except Exception:
            continue

    return pd.DataFrame(), meta


def get_scanner_snapshot_meta(universe_name: str) -> dict:
    base_path = get_scanner_snapshot_base_path(universe_name)
    meta_path = base_path.with_suffix(".json")
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


@lru_cache(maxsize=256)
def get_stock_fundamentals(ticker: str) -> dict:
    ticker = normalize_yahoo_ticker(ticker)
    try:
        stock = get_yfinance_ticker(ticker)
        info = stock.info or {}
    except Exception:
        return {}

    return {
        "shortName": info.get("shortName"),
        "longName": info.get("longName"),
        "marketCap": info.get("marketCap"),
        "enterpriseValue": info.get("enterpriseValue"),
        "trailingPE": info.get("trailingPE"),
        "forwardPE": info.get("forwardPE"),
        "pegRatio": info.get("pegRatio") or info.get("trailingPegRatio"),
        "priceToSalesTrailing12Months": info.get("priceToSalesTrailing12Months"),
        "priceToBook": info.get("priceToBook"),
        "enterpriseToRevenue": info.get("enterpriseToRevenue"),
        "enterpriseToEbitda": info.get("enterpriseToEbitda"),
        "revenueGrowth": info.get("revenueGrowth"),
        "earningsGrowth": info.get("earningsGrowth"),
        "earningsQuarterlyGrowth": info.get("earningsQuarterlyGrowth"),
        "revenueQuarterlyGrowth": info.get("revenueQuarterlyGrowth") or info.get("quarterlyRevenueGrowth"),
        "earningsGrowthEstimateCurrentYear": info.get("earningsGrowthEstimateCurrentYear"),
        "earningsGrowthEstimateNextYear": info.get("earningsGrowthEstimateNextYear"),
        "earningsGrowthEstimateNext5Years": info.get("earningsGrowthEstimateNext5Years"),
        "grossMargins": info.get("grossMargins"),
        "operatingMargins": info.get("operatingMargins"),
        "profitMargins": info.get("profitMargins"),
        "returnOnEquity": info.get("returnOnEquity"),
        "returnOnAssets": info.get("returnOnAssets"),
        "returnOnCapital": info.get("returnOnCapital"),
        "freeCashflow": info.get("freeCashflow"),
        "operatingCashflow": info.get("operatingCashflow"),
        "totalRevenue": info.get("totalRevenue"),
        "ebit": info.get("ebit"),
        "operatingIncome": info.get("operatingIncome"),
        "ebitda": info.get("ebitda"),
        "debtToEquity": info.get("debtToEquity"),
        "totalCash": info.get("totalCash"),
        "totalDebt": info.get("totalDebt"),
        "minorityInterest": info.get("minorityInterest"),
        "totalInvestments": info.get("totalInvestments"),
        "longTermInvestments": info.get("longTermInvestments"),
        "totalCashPerShare": info.get("totalCashPerShare"),
        "currentRatio": info.get("currentRatio"),
        "quickRatio": info.get("quickRatio"),
        "totalDebtToEbitda": info.get("totalDebtToEbitda"),
        "averageVolume": info.get("averageVolume"),
        "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
        "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
        "targetMeanPrice": info.get("targetMeanPrice"),
        "targetMedianPrice": info.get("targetMedianPrice"),
        "recommendationMean": info.get("recommendationMean"),
        "recommendationKey": info.get("recommendationKey"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "beta": info.get("beta"),
        "heldPercentInstitutions": info.get("heldPercentInstitutions"),
        "heldPercentInsiders": info.get("heldPercentInsiders"),
        "shortPercentOfFloat": info.get("shortPercentOfFloat"),
        "sharesShort": info.get("sharesShort"),
        "shortRatio": info.get("shortRatio"),
        "sharesShortPriorMonth": info.get("sharesShortPriorMonth"),
        "shortPercentOfSharesOutstanding": info.get("shortPercentOfSharesOutstanding"),
        "floatShares": info.get("floatShares"),
        "sharesOutstanding": info.get("sharesOutstanding"),
    }


def trim_company_summary(summary: object, max_chars: int = 420) -> str:
    text = " ".join(str(summary or "").split())
    if not text:
        return ""

    sentences = []
    total_length = 0
    for sentence in text.split(". "):
        cleaned = sentence.strip()
        if not cleaned:
            continue
        if not cleaned.endswith("."):
            cleaned = f"{cleaned}."
        next_length = total_length + len(cleaned) + (1 if sentences else 0)
        if sentences and (len(sentences) >= 4 or next_length > max_chars):
            break
        sentences.append(cleaned)
        total_length = next_length
        if len(sentences) >= 4:
            break

    trimmed = " ".join(sentences).strip()
    if trimmed and (len(sentences) >= 4 or len(text) > max_chars):
        if len(trimmed) <= max_chars:
            return trimmed
        return f"{trimmed[:max_chars].rsplit(' ', 1)[0].rstrip()}..."
    if len(text) <= max_chars:
        return text
    if trimmed and len(trimmed) <= max_chars:
        return trimmed
    return f"{text[:max_chars].rsplit(' ', 1)[0].rstrip()}..."


@lru_cache(maxsize=256)
def get_company_profile(ticker: str) -> dict:
    normalized_ticker = normalize_yahoo_ticker(ticker)
    if not normalized_ticker:
        return {}

    try:
        info = get_yfinance_ticker(normalized_ticker).info or {}
    except Exception:
        return {}

    return {
        "longBusinessSummary": trim_company_summary(info.get("longBusinessSummary")),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "exchange": info.get("exchange") or info.get("fullExchangeName"),
        "marketCap": info.get("marketCap"),
    }


@st.cache_data(ttl=FUNDAMENTALS_TTL_SECONDS, show_spinner=False)
def fetch_fundamentals_snapshot(ticker: str) -> dict:
    normalized_ticker = normalize_yahoo_ticker(ticker)
    if not normalized_ticker:
        return {}

    stock = None
    try:
        stock = get_yfinance_ticker(normalized_ticker)
        info = stock.info or {}
    except Exception:
        info = {}

    try:
        history = stock.history(period="1y", interval="1d", auto_adjust=False) if stock is not None else pd.DataFrame()
        history = clean_history(history)
    except Exception:
        history = pd.DataFrame()

    current_price = safe_get_fundamental(
        info,
        ("currentPrice", "regularMarketPrice", "previousClose", "open"),
        default=FUNDAMENTAL_NA,
    )
    market_cap = safe_get_fundamental(info, ("marketCap",), default=FUNDAMENTAL_NA)
    free_cash_flow = safe_get_fundamental(info, ("freeCashflow", "freeCashFlow"), default=FUNDAMENTAL_NA)
    target_mean_price = safe_get_fundamental(info, ("targetMeanPrice", "targetMedianPrice"), default=FUNDAMENTAL_NA)
    fifty_two_week_high = safe_get_fundamental(info, ("fiftyTwoWeekHigh", "52WeekChangeHigh"), default=FUNDAMENTAL_NA)
    fifty_two_week_low = safe_get_fundamental(info, ("fiftyTwoWeekLow", "52WeekChangeLow"), default=FUNDAMENTAL_NA)
    average_volume = safe_get_fundamental(info, ("averageVolume", "averageDailyVolume10Day"), default=FUNDAMENTAL_NA)
    ten_day_average_volume = safe_get_fundamental(info, ("averageVolume10days", "averageDailyVolume10Day"), default=FUNDAMENTAL_NA)
    current_volume = safe_get_fundamental(info, ("volume", "regularMarketVolume"), default=FUNDAMENTAL_NA)

    if current_volume == FUNDAMENTAL_NA and history is not None and not history.empty and "Volume" in history.columns:
        current_volume = safe_float(history["Volume"].dropna().iloc[-1]) if not history["Volume"].dropna().empty else FUNDAMENTAL_NA
    if history is not None and not history.empty:
        close_values = history["Close"].dropna() if "Close" in history.columns else pd.Series(dtype=float)
        high_values = history["High"].dropna() if "High" in history.columns else pd.Series(dtype=float)
        low_values = history["Low"].dropna() if "Low" in history.columns else pd.Series(dtype=float)
        if not close_values.empty:
            current_price = safe_float(close_values.iloc[-1]) or current_price
        if not high_values.empty:
            fifty_two_week_high = safe_float(high_values.tail(252).max()) or fifty_two_week_high
        if not low_values.empty:
            fifty_two_week_low = safe_float(low_values.tail(252).min()) or fifty_two_week_low

    snapshot = {
        "Company Profile": {
            "Company Name": safe_get_fundamental(info, ("longName", "shortName", "displayName")),
            "Sector": safe_get_fundamental(info, ("sector",)),
            "Industry": safe_get_fundamental(info, ("industry",)),
            "Country": safe_get_fundamental(info, ("country",)),
            "Exchange": safe_get_fundamental(info, ("exchange", "fullExchangeName")),
            "Currency": safe_get_fundamental(info, ("currency", "financialCurrency")),
            "Website": safe_get_fundamental(info, ("website",)),
            "Full-Time Employees": safe_get_fundamental(info, ("fullTimeEmployees",)),
            "Fiscal Year End": safe_get_fundamental(info, ("lastFiscalYearEnd", "fiscalYearEnd")),
            "Most Recent Quarter": safe_get_fundamental(info, ("mostRecentQuarter",)),
        },
        "Valuation": {
            "Market Cap": market_cap,
            "Enterprise Value": safe_get_fundamental(info, ("enterpriseValue",)),
            "Trailing P/E": safe_get_fundamental(info, ("trailingPE",)),
            "Forward P/E": safe_get_fundamental(info, ("forwardPE",)),
            "PEG Ratio": safe_get_fundamental(info, ("pegRatio", "trailingPegRatio")),
            "Price / Sales": safe_get_fundamental(info, ("priceToSalesTrailing12Months",)),
            "Price / Book": safe_get_fundamental(info, ("priceToBook",)),
            "EV / Revenue": safe_get_fundamental(info, ("enterpriseToRevenue",)),
            "EV / EBITDA": safe_get_fundamental(info, ("enterpriseToEbitda",)),
            "Price / Cash Flow": safe_get_fundamental(info, ("priceToCashflow", "priceToCashFlow")),
            "Forward EPS": safe_get_fundamental(info, ("forwardEps",)),
        },
        "Earnings / EPS": {
            "EPS (TTM)": safe_get_fundamental(info, ("trailingEps",)),
            "Forward EPS": safe_get_fundamental(info, ("forwardEps",)),
            "EPS Current Year": safe_get_fundamental(info, ("currentYearEstimate", "epsCurrentYear")),
            "EPS Next Year": safe_get_fundamental(info, ("nextYearEstimate", "epsNextYear")),
            "EPS Next Quarter": safe_get_fundamental(info, ("nextQuarterEstimate", "epsNextQuarter")),
            "EPS Growth YoY": safe_get_fundamental(info, ("earningsGrowth",)),
            "Earnings Growth": safe_get_fundamental(info, ("earningsGrowth",)),
            "Quarterly Earnings Growth YoY": safe_get_fundamental(info, ("earningsQuarterlyGrowth", "quarterlyEarningsGrowth")),
            "Revenue Per Share": safe_get_fundamental(info, ("revenuePerShare",)),
            "Net Income to Common": safe_get_fundamental(info, ("netIncomeToCommon",)),
            "EBITDA": safe_get_fundamental(info, ("ebitda",)),
            "Earnings Date": safe_get_fundamental(info, ("earningsDate", "earningsTimestamp")),
            "EPS Estimate Current Quarter": safe_get_fundamental(info, ("epsEstimateCurrentQuarter", "currentQuarterEstimate")),
            "EPS Estimate Next Quarter": safe_get_fundamental(info, ("epsEstimateNextQuarter", "nextQuarterEstimate")),
        },
        "Growth": {
            "Revenue Growth": safe_get_fundamental(info, ("revenueGrowth",)),
            "Earnings Growth": safe_get_fundamental(info, ("earningsGrowth",)),
            "Quarterly Revenue Growth YoY": safe_get_fundamental(info, ("revenueQuarterlyGrowth", "quarterlyRevenueGrowth")),
            "Quarterly Earnings Growth YoY": safe_get_fundamental(info, ("earningsQuarterlyGrowth", "quarterlyEarningsGrowth")),
            "Revenue Per Share": safe_get_fundamental(info, ("revenuePerShare",)),
            "EPS Trailing Twelve Months": safe_get_fundamental(info, ("trailingEps",)),
            "EPS Forward": safe_get_fundamental(info, ("forwardEps",)),
            "EPS Current Year": safe_get_fundamental(info, ("epsCurrentYear", "earningsGrowthEstimateCurrentYear")),
            "EPS Next Year": safe_get_fundamental(info, ("epsNextYear", "earningsGrowthEstimateNextYear")),
            "Analyst Revenue Estimate": safe_get_fundamental(info, ("revenueEstimate", "averageRevenueEstimate")),
        },
        "Profitability": {
            "Gross Margin": safe_get_fundamental(info, ("grossMargins",)),
            "Operating Margin": safe_get_fundamental(info, ("operatingMargins",)),
            "Profit Margin": safe_get_fundamental(info, ("profitMargins",)),
            "EBITDA Margin": safe_get_fundamental(info, ("ebitdaMargins",)),
            "Return on Equity": safe_get_fundamental(info, ("returnOnEquity",)),
            "Return on Assets": safe_get_fundamental(info, ("returnOnAssets",)),
            "Return on Invested Capital": safe_get_fundamental(info, ("returnOnCapital", "returnOnInvestedCapital")),
            "EBITDA": safe_get_fundamental(info, ("ebitda",)),
            "Net Income": safe_get_fundamental(info, ("netIncomeToCommon", "netIncome")),
        },
        "Balance Sheet": {
            "Total Cash": safe_get_fundamental(info, ("totalCash",)),
            "Cash Per Share": safe_get_fundamental(info, ("totalCashPerShare",)),
            "Total Debt": safe_get_fundamental(info, ("totalDebt",)),
            "Debt / Equity": safe_get_fundamental(info, ("debtToEquity",)),
            "Current Ratio": safe_get_fundamental(info, ("currentRatio",)),
            "Quick Ratio": safe_get_fundamental(info, ("quickRatio",)),
            "Book Value Per Share": safe_get_fundamental(info, ("bookValue",)),
            "Working Capital": safe_get_fundamental(info, ("workingCapital",)),
            "Enterprise Value": safe_get_fundamental(info, ("enterpriseValue",)),
        },
        "Cash Flow": {
            "Operating Cash Flow": safe_get_fundamental(info, ("operatingCashflow", "operatingCashFlow")),
            "Free Cash Flow": free_cash_flow,
            "Levered Free Cash Flow": safe_get_fundamental(info, ("leveredFreeCashFlow",)),
            "Free Cash Flow Yield": value_or_na(calculate_free_cash_flow_yield(free_cash_flow, market_cap)),
            "CapEx": safe_get_fundamental(info, ("capitalExpenditures", "capitalExpenditure")),
        },
        "Share Structure / Ownership": {
            "Shares Outstanding": safe_get_fundamental(info, ("sharesOutstanding",)),
            "Float Shares": safe_get_fundamental(info, ("floatShares",)),
            "Shares Short": safe_get_fundamental(info, ("sharesShort",)),
            "Short Ratio": safe_get_fundamental(info, ("shortRatio",)),
            "Short Percent of Float": safe_get_fundamental(info, ("shortPercentOfFloat",)),
            "Percent Held by Insiders": safe_get_fundamental(info, ("heldPercentInsiders",)),
            "Percent Held by Institutions": safe_get_fundamental(info, ("heldPercentInstitutions",)),
            "Implied Shares Outstanding": safe_get_fundamental(info, ("impliedSharesOutstanding",)),
        },
        "Dividends": {
            "Dividend Rate": safe_get_fundamental(info, ("dividendRate", "trailingAnnualDividendRate")),
            "Dividend Yield": safe_get_fundamental(info, ("dividendYield", "trailingAnnualDividendYield")),
            "Payout Ratio": safe_get_fundamental(info, ("payoutRatio",)),
            "Ex-Dividend Date": safe_get_fundamental(info, ("exDividendDate",)),
            "Five-Year Average Dividend Yield": safe_get_fundamental(info, ("fiveYearAvgDividendYield",)),
            "Dividend Growth": safe_get_fundamental(info, ("dividendGrowth",)),
        },
        "Analyst / Target Info": {
            "Recommendation Mean": safe_get_fundamental(info, ("recommendationMean",)),
            "Recommendation Key": safe_get_fundamental(info, ("recommendationKey",)),
            "Number of Analyst Opinions": safe_get_fundamental(info, ("numberOfAnalystOpinions",)),
            "Target High Price": safe_get_fundamental(info, ("targetHighPrice",)),
            "Target Mean Price": target_mean_price,
            "Target Low Price": safe_get_fundamental(info, ("targetLowPrice",)),
            "Current Price vs Target Mean %": value_or_na(calculate_target_mean_upside(current_price, target_mean_price)),
            "Earnings Date": safe_get_fundamental(info, ("earningsTimestamp", "earningsDate")),
        },
        "Performance / Risk Snapshot": {
            "52 Week High": fifty_two_week_high,
            "52 Week Low": fifty_two_week_low,
            "Distance From 52W High %": value_or_na(calculate_distance_from_level(current_price, fifty_two_week_high)),
            "Distance From 52W Low %": value_or_na(calculate_distance_from_level(current_price, fifty_two_week_low)),
            "Beta": safe_get_fundamental(info, ("beta",)),
            "Average Volume": average_volume,
            "10-Day Average Volume": ten_day_average_volume,
            "Relative Volume": value_or_na(calculate_ratio(current_volume, average_volume)),
            "ATR %": value_or_na(calculate_atr_percent(history)),
            "RSI 14": value_or_na(calculate_rsi(history)),
            "Performance 1W": value_or_na(calculate_percent_change_from_history(history, 5)),
            "Performance 1M": value_or_na(calculate_percent_change_from_history(history, 21)),
            "Performance 3M": value_or_na(calculate_percent_change_from_history(history, 63)),
            "Performance 6M": value_or_na(calculate_percent_change_from_history(history, 126)),
            "Performance YTD": value_or_na(calculate_ytd_performance(history)),
            "Performance 1Y": value_or_na(calculate_percent_change_from_history(history, 252)),
        },
    }
    snapshot["__quality__"] = build_fundamental_quality_summary(snapshot)
    return snapshot


@lru_cache(maxsize=256)
def get_default_wacc_inputs(ticker: str) -> dict:
    normalized_ticker = normalize_yahoo_ticker(ticker)
    fallback = {
        "sector": None,
        "industry": None,
        "beta": None,
        "risk_free_rate": DEFAULT_RISK_FREE_RATE,
        "equity_risk_premium": DEFAULT_EQUITY_RISK_PREMIUM,
        "cost_of_equity": None,
        "pre_tax_cost_of_debt": DEFAULT_PRE_TAX_COST_OF_DEBT,
        "tax_rate": DEFAULT_TAX_RATE,
        "debt_weight": 0.0,
        "equity_weight": 1.0,
        "wacc": SECTOR_WACC_FALLBACKS["Default"],
        "source_note": "Using default WACC fallback.",
    }
    if not normalized_ticker:
        return fallback

    try:
        info = get_stock_fundamentals(normalized_ticker)
    except Exception:
        return fallback

    sector = info.get("sector")
    industry = info.get("industry")
    beta = safe_float(info.get("beta"))
    if beta is not None:
        beta = clamp_ratio(beta, 0.3, 3.0)

    market_cap = safe_float(info.get("marketCap"))
    total_debt = safe_float(info.get("totalDebt")) or 0.0
    enterprise_value = safe_float(info.get("enterpriseValue"))
    debt_base = total_debt if total_debt > 0 else 0.0
    capital_base = (market_cap or 0.0) + debt_base

    risk_free_rate = DEFAULT_RISK_FREE_RATE
    equity_risk_premium = DEFAULT_EQUITY_RISK_PREMIUM
    pre_tax_cost_of_debt = DEFAULT_PRE_TAX_COST_OF_DEBT
    tax_rate = DEFAULT_TAX_RATE

    if beta is not None and market_cap and market_cap > 0 and capital_base > 0:
        debt_weight = clamp_ratio(debt_base / capital_base, 0.0, 0.95)
        equity_weight = 1.0 - debt_weight
        cost_of_equity = risk_free_rate + beta * equity_risk_premium
        after_tax_cost_of_debt = pre_tax_cost_of_debt * (1 - tax_rate)
        wacc = equity_weight * cost_of_equity + debt_weight * after_tax_cost_of_debt
        return {
            "sector": sector,
            "industry": industry,
            "beta": beta,
            "risk_free_rate": risk_free_rate,
            "equity_risk_premium": equity_risk_premium,
            "cost_of_equity": cost_of_equity,
            "pre_tax_cost_of_debt": pre_tax_cost_of_debt,
            "tax_rate": tax_rate,
            "debt_weight": debt_weight,
            "equity_weight": equity_weight,
            "wacc": clamp_ratio(wacc, 0.04, 0.20),
            "source_note": "Using company beta and capital structure.",
        }

    sector_wacc = SECTOR_WACC_FALLBACKS.get(str(sector), SECTOR_WACC_FALLBACKS["Default"])
    return {
        **fallback,
        "sector": sector,
        "industry": industry,
        "beta": beta,
        "cost_of_equity": risk_free_rate + beta * equity_risk_premium if beta is not None else None,
        "debt_weight": clamp_ratio(debt_base / enterprise_value, 0.0, 0.95) if enterprise_value and enterprise_value > 0 else 0.0,
        "equity_weight": 1.0 - (clamp_ratio(debt_base / enterprise_value, 0.0, 0.95) if enterprise_value and enterprise_value > 0 else 0.0),
        "wacc": sector_wacc,
        "source_note": "Using sector fallback." if sector in SECTOR_WACC_FALLBACKS else "Using default WACC fallback.",
    }


@lru_cache(maxsize=256)
def get_earnings_info(ticker: str) -> dict:
    normalized_ticker = normalize_yahoo_ticker(ticker)
    if not normalized_ticker:
        return {"status": "unavailable", "source": None}

    today = date.today()
    earnings_dates = []

    try:
        dates = get_yfinance_ticker(normalized_ticker).get_earnings_dates(limit=12)
        if dates is not None and not dates.empty:
            earnings_dates = [pd.Timestamp(index_value).date() for index_value in dates.index]
            source = "yfinance.get_earnings_dates"
        else:
            source = None
    except Exception:
        source = None

    if not earnings_dates:
        try:
            calendar = get_yfinance_ticker(normalized_ticker).calendar
            calendar_date = extract_calendar_earnings_date(calendar)
            if calendar_date is not None:
                earnings_dates = [calendar_date]
                source = "yfinance.calendar"
        except Exception:
            source = None

    if not earnings_dates:
        return {
            "next_earnings_date": None,
            "days_until_earnings": None,
            "last_earnings_date": None,
            "source": source,
            "status": "unavailable",
        }

    future_dates = sorted(earnings_date for earnings_date in earnings_dates if earnings_date >= today)
    past_dates = sorted((earnings_date for earnings_date in earnings_dates if earnings_date < today), reverse=True)
    next_earnings_date = future_dates[0] if future_dates else None
    last_earnings_date = past_dates[0] if past_dates else None

    return {
        "next_earnings_date": next_earnings_date.isoformat() if next_earnings_date else None,
        "days_until_earnings": (next_earnings_date - today).days if next_earnings_date else None,
        "last_earnings_date": last_earnings_date.isoformat() if last_earnings_date else None,
        "source": source,
        "status": "available" if next_earnings_date or last_earnings_date else "unavailable",
    }


def get_estimate_value(frame: object, period: str, column: str) -> object:
    if frame is None or getattr(frame, "empty", True):
        return None
    if column not in frame.columns:
        return None
    try:
        if period in frame.index:
            value = frame.loc[period, column]
        else:
            return None
        if isinstance(value, pd.Series):
            value = value.dropna().iloc[0] if not value.dropna().empty else None
        if pd.isna(value):
            return None
        return value
    except Exception:
        return None


@lru_cache(maxsize=256)
def get_forward_estimates(ticker: str) -> dict:
    normalized_ticker = normalize_yahoo_ticker(ticker)
    if not normalized_ticker:
        return {}

    try:
        stock = get_yfinance_ticker(normalized_ticker)
    except Exception:
        return {}

    try:
        earnings_estimate = stock.get_earnings_estimate()
    except Exception:
        earnings_estimate = pd.DataFrame()

    try:
        revenue_estimate = stock.get_revenue_estimate()
    except Exception:
        revenue_estimate = pd.DataFrame()

    try:
        growth_estimate = stock.get_growth_estimates()
    except Exception:
        growth_estimate = pd.DataFrame()

    return {
        "eps_current_quarter_avg": get_estimate_value(earnings_estimate, "0q", "avg"),
        "eps_next_quarter_avg": get_estimate_value(earnings_estimate, "+1q", "avg"),
        "eps_current_year_avg": get_estimate_value(earnings_estimate, "0y", "avg"),
        "eps_next_year_avg": get_estimate_value(earnings_estimate, "+1y", "avg"),
        "eps_current_quarter_growth": get_estimate_value(earnings_estimate, "0q", "growth"),
        "eps_next_quarter_growth": get_estimate_value(earnings_estimate, "+1q", "growth"),
        "eps_current_year_growth": get_estimate_value(earnings_estimate, "0y", "growth"),
        "eps_next_year_growth": get_estimate_value(earnings_estimate, "+1y", "growth"),
        "eps_current_year_analysts": get_estimate_value(earnings_estimate, "0y", "numberOfAnalysts"),
        "eps_next_year_analysts": get_estimate_value(earnings_estimate, "+1y", "numberOfAnalysts"),
        "revenue_current_quarter_avg": get_estimate_value(revenue_estimate, "0q", "avg"),
        "revenue_next_quarter_avg": get_estimate_value(revenue_estimate, "+1q", "avg"),
        "revenue_current_year_avg": get_estimate_value(revenue_estimate, "0y", "avg"),
        "revenue_next_year_avg": get_estimate_value(revenue_estimate, "+1y", "avg"),
        "revenue_current_quarter_growth": get_estimate_value(revenue_estimate, "0q", "growth"),
        "revenue_next_quarter_growth": get_estimate_value(revenue_estimate, "+1q", "growth"),
        "revenue_current_year_growth": get_estimate_value(revenue_estimate, "0y", "growth"),
        "revenue_next_year_growth": get_estimate_value(revenue_estimate, "+1y", "growth"),
        "growth_current_quarter": get_estimate_value(growth_estimate, "0q", "stockTrend"),
        "growth_next_quarter": get_estimate_value(growth_estimate, "+1q", "stockTrend"),
        "growth_current_year": get_estimate_value(growth_estimate, "0y", "stockTrend"),
        "growth_next_year": get_estimate_value(growth_estimate, "+1y", "stockTrend"),
        "growth_long_term": get_estimate_value(growth_estimate, "LTG", "stockTrend"),
        "source": "yfinance estimate tables",
    }


def extract_calendar_earnings_date(calendar: object) -> date | None:
    if calendar is None:
        return None

    value = None
    if isinstance(calendar, dict):
        value = calendar.get("Earnings Date") or calendar.get("EarningsDate")
    elif isinstance(calendar, pd.DataFrame):
        if "Earnings Date" in calendar.index:
            value = calendar.loc["Earnings Date"].dropna().iloc[0]
        elif "Earnings Date" in calendar.columns:
            value = calendar["Earnings Date"].dropna().iloc[0]

    if isinstance(value, (list, tuple, pd.Series)):
        value = next((item for item in value if item is not None), None)

    if value is None:
        return None

    try:
        return pd.Timestamp(value).date()
    except Exception:
        return None


@lru_cache(maxsize=64)
def get_stock_chart_history(ticker: str, timeframe: str = "1Y") -> pd.DataFrame:
    period = CHART_TIMEFRAME_FETCH_PERIODS.get(timeframe, CHART_TIMEFRAME_FETCH_PERIODS["1Y"])
    normalized_ticker = normalize_yahoo_ticker(ticker)
    try:
        history = get_yfinance_ticker(normalized_ticker).history(period=period, interval="1d", auto_adjust=False)
    except Exception:
        return pd.DataFrame()

    history = clean_history(history)
    if history.empty:
        return pd.DataFrame()
    return history.copy()
