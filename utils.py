def sort_rows_desc(rows: list[dict], key: str) -> list[dict]:
    return sorted(rows, key=lambda row: row.get(key, 0), reverse=True)


def format_price(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"${value:,.2f}"


def format_change_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:+.1f}%"


def format_section_rows(rows: list[dict]) -> list[dict]:
    formatted_rows = []

    for row in rows:
        formatted_rows.append(
            {
                "Ticker": row.get("ticker", "N/A"),
                "Price": format_price(row.get("price")),
                "Change %": format_change_pct(row.get("change_pct")),
                "Signal": f"{row.get('signal_score', 0):.1f}",
                "Label": row.get("label", "Pass"),
                "Base": f"{row.get('base_score', 0):.1f}",
                "Trigger": f"{row.get('trigger_score', 0):.1f}",
                "Follow": f"{row.get('follow_through_score', 0):.1f}",
                "Risk": f"{row.get('risk_score', 0):.1f}",
                "Penalty": f"{row.get('penalty_score', 0):.1f}",
                "Volume Ratio": f"{row.get('volume_ratio', 0):.2f}",
                "ADDV": format_price(row.get("addv")),
                "Tags": ", ".join(row.get("tags", [])[:4]),
            }
        )

    return formatted_rows
