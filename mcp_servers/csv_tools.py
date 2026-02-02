#!/usr/bin/env python3
"""MCP server providing CSV file tools.

Example MCP server that exposes tools for reading and analyzing CSV files.
Uses only the Python standard library (csv, statistics) with no pandas dependency.

Usage:
    uv run python -m mcp_servers.csv_tools
"""

import csv
import statistics

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("csv-tools")


@mcp.tool()
def read_csv_file(file_path: str) -> str:
    """Read a CSV file and return its contents as formatted text.

    Args:
        file_path: Path to the CSV file to read.

    Returns:
        The CSV contents formatted as a readable table.
    """
    with open(file_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return "The CSV file is empty."

    headers = list(rows[0].keys())
    lines = [", ".join(headers)]
    for row in rows:
        lines.append(", ".join(row[h] for h in headers))

    return "\n".join(lines)


@mcp.tool()
def csv_stats(file_path: str, column: str) -> str:
    """Compute basic statistics for a numeric column in a CSV file.

    Args:
        file_path: Path to the CSV file.
        column: Name of the numeric column to analyze.

    Returns:
        A summary of count, mean, median, min, max, and stdev for the column.
    """
    with open(file_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return "The CSV file is empty."

    if column not in rows[0]:
        available = ", ".join(rows[0].keys())
        return f"Column '{column}' not found. Available columns: {available}"

    values = []
    for row in rows:
        try:
            values.append(float(row[column]))
        except (ValueError, TypeError):
            continue

    if not values:
        return f"No numeric values found in column '{column}'."

    result = {
        "column": column,
        "count": len(values),
        "mean": round(statistics.mean(values), 2),
        "median": round(statistics.median(values), 2),
        "min": min(values),
        "max": max(values),
    }
    if len(values) >= 2:
        result["stdev"] = round(statistics.stdev(values), 2)

    lines = [f"Statistics for '{column}':"]
    for key, val in result.items():
        if key == "column":
            continue
        lines.append(f"  {key}: {val}")

    return "\n".join(lines)


@mcp.tool()
def count_rows(file_path: str) -> str:
    """Count the number of data rows in a CSV file (excluding the header).

    Args:
        file_path: Path to the CSV file.

    Returns:
        The row count as a string.
    """
    with open(file_path, newline="") as f:
        reader = csv.DictReader(f)
        count = sum(1 for _ in reader)

    return f"The CSV file has {count} rows."


if __name__ == "__main__":
    mcp.run(transport="stdio")
