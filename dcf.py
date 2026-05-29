from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any


def safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return default
    if not isfinite(numeric_value):
        return default
    return numeric_value


def build_revenue_growth_path(
    high_growth_rate: float,
    fade_growth_rate: float,
    terminal_growth_rate: float,
    projection_years: int,
    high_growth_years: int = 5,
) -> list[float]:
    years = max(0, int(projection_years))
    if years == 0:
        return []

    growth_rates: list[float] = []
    fade_years = max(1, years - high_growth_years)
    for year in range(1, years + 1):
        if year <= high_growth_years:
            growth_rates.append(high_growth_rate)
            continue
        fade_step = (year - high_growth_years - 1) / max(1, fade_years - 1)
        growth_rates.append(fade_growth_rate + (terminal_growth_rate - fade_growth_rate) * fade_step)
    return growth_rates


def build_margin_path(
    current_margin: float,
    target_margin: float,
    years_to_target: int,
    projection_years: int,
) -> list[float]:
    years = max(0, int(projection_years))
    if years == 0:
        return []

    target_year = max(1, int(years_to_target))
    margins: list[float] = []
    for year in range(1, years + 1):
        progress = min(1.0, year / target_year)
        margins.append(current_margin + (target_margin - current_margin) * progress)
    return margins


def calculate_terminal_value(
    terminal_revenue: float,
    terminal_margin: float,
    tax_rate: float,
    terminal_growth_rate: float,
    terminal_roic: float,
    wacc: float,
) -> dict[str, float | None]:
    if wacc <= terminal_growth_rate or terminal_roic <= terminal_growth_rate:
        return {
            "terminal_revenue": terminal_revenue,
            "terminal_ebit": None,
            "terminal_nopat": None,
            "terminal_reinvestment": None,
            "terminal_reinvestment_rate": None,
            "terminal_fcff": None,
            "terminal_value": None,
        }

    terminal_ebit = terminal_revenue * terminal_margin
    terminal_nopat = terminal_ebit * (1 - tax_rate)
    reinvestment_rate = terminal_growth_rate / terminal_roic if terminal_roic else None
    terminal_reinvestment = terminal_nopat * reinvestment_rate if reinvestment_rate is not None else None
    terminal_fcff = terminal_nopat - terminal_reinvestment if terminal_reinvestment is not None else None
    terminal_value = terminal_fcff / (wacc - terminal_growth_rate) if terminal_fcff is not None else None

    return {
        "terminal_revenue": terminal_revenue,
        "terminal_ebit": terminal_ebit,
        "terminal_nopat": terminal_nopat,
        "terminal_reinvestment": terminal_reinvestment,
        "terminal_reinvestment_rate": reinvestment_rate,
        "terminal_fcff": terminal_fcff,
        "terminal_value": terminal_value,
    }


def calculate_fcff_forecast(
    starting_revenue: float,
    current_margin: float,
    target_margin: float,
    years_to_target_margin: int,
    high_growth_rate: float,
    fade_growth_rate: float,
    terminal_growth_rate: float,
    tax_rate: float,
    sales_to_capital_ratio: float,
    wacc: float,
    terminal_margin: float,
    terminal_roic: float,
    projection_years: int,
) -> dict[str, Any]:
    growth_path = build_revenue_growth_path(
        high_growth_rate,
        fade_growth_rate,
        terminal_growth_rate,
        projection_years,
    )
    margin_path = build_margin_path(current_margin, target_margin, years_to_target_margin, projection_years)

    rows: list[dict[str, float]] = []
    revenue = starting_revenue
    pv_explicit_fcff = 0.0

    for index, growth_rate in enumerate(growth_path, start=1):
        prior_revenue = revenue
        revenue = prior_revenue * (1 + growth_rate)
        revenue_change = revenue - prior_revenue
        operating_margin = margin_path[index - 1]
        ebit = revenue * operating_margin
        nopat = ebit * (1 - tax_rate)
        reinvestment = revenue_change / sales_to_capital_ratio
        fcff = nopat - reinvestment
        discount_factor = (1 + wacc) ** index
        pv_fcff = fcff / discount_factor
        pv_explicit_fcff += pv_fcff

        rows.append(
            {
                "Year": index,
                "Revenue": revenue,
                "Revenue Growth": growth_rate,
                "Operating Margin": operating_margin,
                "EBIT": ebit,
                "NOPAT": nopat,
                "Reinvestment": reinvestment,
                "FCFF": fcff,
                "PV FCFF": pv_fcff,
            }
        )

    terminal_revenue = revenue * (1 + terminal_growth_rate)
    terminal = calculate_terminal_value(
        terminal_revenue,
        terminal_margin,
        tax_rate,
        terminal_growth_rate,
        terminal_roic,
        wacc,
    )
    terminal_value = terminal.get("terminal_value")
    pv_terminal_value = (
        terminal_value / ((1 + wacc) ** projection_years)
        if terminal_value is not None and projection_years > 0
        else None
    )

    enterprise_value = pv_explicit_fcff + pv_terminal_value if pv_terminal_value is not None else None
    return {
        "forecast": rows,
        "terminal": terminal,
        "pv_explicit_fcff": pv_explicit_fcff,
        "pv_terminal_value": pv_terminal_value,
        "enterprise_value": enterprise_value,
    }


def calculate_enterprise_value(pv_explicit_fcff: float, pv_terminal_value: float) -> float:
    return pv_explicit_fcff + pv_terminal_value


def calculate_equity_value(
    enterprise_value: float,
    cash: float = 0.0,
    debt: float = 0.0,
    minority_interest: float = 0.0,
    investments: float = 0.0,
) -> float:
    return enterprise_value + cash + investments - debt - minority_interest


def calculate_fair_value_per_share(equity_value: float, diluted_shares: float) -> float | None:
    if diluted_shares <= 0:
        return None
    return equity_value / diluted_shares


@dataclass(frozen=True)
class DcfAssumptions:
    starting_revenue: float
    current_operating_margin: float
    tax_rate: float
    cash: float
    debt: float
    diluted_shares: float
    minority_interest: float
    investments: float
    high_growth_rate: float
    fade_growth_rate: float
    target_operating_margin: float
    years_to_target_margin: int
    projection_years: int
    sales_to_capital_ratio: float
    wacc: float
    terminal_growth_rate: float
    terminal_operating_margin: float
    terminal_roic: float
    current_price: float | None = None


def validate_assumptions(assumptions: DcfAssumptions) -> list[str]:
    warnings: list[str] = []
    if assumptions.starting_revenue <= 0:
        warnings.append("Revenue is missing or invalid; the DCF cannot be calculated.")
    if assumptions.diluted_shares <= 0:
        warnings.append("Diluted shares outstanding are missing or invalid; fair value per share cannot be calculated.")
    if assumptions.sales_to_capital_ratio <= 0:
        warnings.append("Sales-to-capital ratio must be greater than zero.")
    if assumptions.wacc <= assumptions.terminal_growth_rate:
        warnings.append("WACC must be greater than terminal growth.")
    if assumptions.terminal_roic <= assumptions.terminal_growth_rate:
        warnings.append("Terminal ROIC must be greater than terminal growth.")
    if assumptions.current_operating_margin < 0:
        warnings.append("Current operating margin is negative; the valuation may be unreliable.")
    if assumptions.terminal_growth_rate > 0.04:
        warnings.append("Terminal growth is above 4%; use caution for mature-company assumptions.")
    return warnings


def calculate_damodaran_dcf(assumptions: DcfAssumptions) -> dict[str, Any]:
    warnings = validate_assumptions(assumptions)
    blocking_errors = [
        warning
        for warning in warnings
        if warning.startswith(("Revenue", "Diluted shares", "Sales-to-capital", "WACC", "Terminal ROIC"))
    ]
    if blocking_errors:
        return {"ok": False, "warnings": warnings}

    forecast_result = calculate_fcff_forecast(
        starting_revenue=assumptions.starting_revenue,
        current_margin=assumptions.current_operating_margin,
        target_margin=assumptions.target_operating_margin,
        years_to_target_margin=assumptions.years_to_target_margin,
        high_growth_rate=assumptions.high_growth_rate,
        fade_growth_rate=assumptions.fade_growth_rate,
        terminal_growth_rate=assumptions.terminal_growth_rate,
        tax_rate=assumptions.tax_rate,
        sales_to_capital_ratio=assumptions.sales_to_capital_ratio,
        wacc=assumptions.wacc,
        terminal_margin=assumptions.terminal_operating_margin,
        terminal_roic=assumptions.terminal_roic,
        projection_years=assumptions.projection_years,
    )
    enterprise_value = forecast_result.get("enterprise_value")
    if enterprise_value is None:
        warnings.append("Terminal value could not be calculated with the current assumptions.")
        return {"ok": False, "warnings": warnings, **forecast_result}

    equity_value = calculate_equity_value(
        enterprise_value,
        cash=assumptions.cash,
        debt=assumptions.debt,
        minority_interest=assumptions.minority_interest,
        investments=assumptions.investments,
    )
    fair_value_per_share = calculate_fair_value_per_share(equity_value, assumptions.diluted_shares)
    current_price = assumptions.current_price
    upside_downside = (
        ((fair_value_per_share / current_price) - 1) * 100
        if fair_value_per_share is not None and current_price not in (None, 0)
        else None
    )
    margin_of_safety_price = fair_value_per_share * 0.75 if fair_value_per_share is not None else None
    pv_terminal_value = forecast_result.get("pv_terminal_value") or 0.0
    terminal_value_share = pv_terminal_value / enterprise_value if enterprise_value else None
    if terminal_value_share is not None and terminal_value_share > 0.80:
        warnings.append("Terminal value is more than 80% of enterprise value; WACC and terminal growth matter a lot.")

    return {
        "ok": True,
        "warnings": warnings,
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "fair_value_per_share": fair_value_per_share,
        "current_price": current_price,
        "upside_downside": upside_downside,
        "margin_of_safety_price": margin_of_safety_price,
        "terminal_value_share": terminal_value_share,
        **forecast_result,
    }


def calculate_scenarios(assumptions: DcfAssumptions) -> list[dict[str, Any]]:
    scenario_inputs = [
        (
            "Bear",
            max(-0.20, assumptions.high_growth_rate - 0.03),
            max(-0.20, assumptions.target_operating_margin - 0.03),
            assumptions.wacc + 0.01,
            max(0.0, assumptions.terminal_growth_rate - 0.005),
            max(0.5, assumptions.sales_to_capital_ratio * 0.85),
        ),
        (
            "Base",
            assumptions.high_growth_rate,
            assumptions.target_operating_margin,
            assumptions.wacc,
            assumptions.terminal_growth_rate,
            assumptions.sales_to_capital_ratio,
        ),
        (
            "Bull",
            assumptions.high_growth_rate + 0.03,
            assumptions.target_operating_margin + 0.03,
            max(0.01, assumptions.wacc - 0.005),
            min(assumptions.wacc - 0.005, assumptions.terminal_growth_rate + 0.005),
            assumptions.sales_to_capital_ratio * 1.15,
        ),
    ]

    rows: list[dict[str, Any]] = []
    for name, growth, margin, wacc, terminal_growth, sales_to_capital in scenario_inputs:
        scenario = DcfAssumptions(
            **{
                **assumptions.__dict__,
                "high_growth_rate": growth,
                "target_operating_margin": margin,
                "wacc": wacc,
                "terminal_growth_rate": terminal_growth,
                "terminal_roic": max(assumptions.terminal_roic, terminal_growth + 0.01),
                "sales_to_capital_ratio": sales_to_capital,
            }
        )
        result = calculate_damodaran_dcf(scenario)
        rows.append(
            {
                "Scenario": name,
                "Revenue CAGR": growth,
                "Target Operating Margin": margin,
                "WACC": wacc,
                "Terminal Growth": terminal_growth,
                "Sales-to-Capital": sales_to_capital,
                "Fair Value / Share": result.get("fair_value_per_share"),
                "Current Price": assumptions.current_price,
                "Upside / Downside": result.get("upside_downside"),
                "MOS Buy Price": result.get("margin_of_safety_price"),
                "Enterprise Value": result.get("enterprise_value"),
                "Equity Value": result.get("equity_value"),
            }
        )
    return rows


def build_sensitivity_table(
    assumptions: DcfAssumptions,
    wacc_offsets: list[float] | None = None,
    terminal_growth_offsets: list[float] | None = None,
) -> list[dict[str, Any]]:
    wacc_offsets = wacc_offsets or [-0.01, 0.0, 0.01, 0.02]
    terminal_growth_offsets = terminal_growth_offsets or [-0.01, 0.0, 0.005, 0.01]
    rows: list[dict[str, Any]] = []

    for wacc_offset in wacc_offsets:
        row_wacc = max(0.001, assumptions.wacc + wacc_offset)
        row: dict[str, Any] = {"WACC": row_wacc}
        for growth_offset in terminal_growth_offsets:
            terminal_growth = max(0.0, assumptions.terminal_growth_rate + growth_offset)
            label = f"{terminal_growth * 100:.1f}%"
            if row_wacc <= terminal_growth:
                row[label] = None
                continue
            scenario = DcfAssumptions(
                **{
                    **assumptions.__dict__,
                    "wacc": row_wacc,
                    "terminal_growth_rate": terminal_growth,
                    "terminal_roic": max(assumptions.terminal_roic, terminal_growth + 0.01),
                }
            )
            result = calculate_damodaran_dcf(scenario)
            row[label] = result.get("fair_value_per_share")
        rows.append(row)
    return rows
