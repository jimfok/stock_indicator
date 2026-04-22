"""Compute per-year capital utilization for a trade_detail CSV.

Supports both the complex_simulation (column ``set``) and
multi_bucket_simulation (column ``bucket``) CSV shapes. For each year:

- total (all scopes combined)  and per-bucket:
  - number of trades entered in that year
  - average daily concurrent positions (across business days in the year)
  - peak daily concurrent positions
  - slot occupancy % = avg_concurrent / max_positions × 100 (when max_positions provided)
  - business days in the year

The idea is to see how much of the available slot capacity each bucket
actually uses, year by year, and how much overlap exists between buckets.

Usage:
    python -m scripts.capital_utilization_report <trade-detail-csv> \
        [--max-positions N] [--output PATH]
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import pandas


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Trade detail CSV from complex_simulation or multi_bucket_simulation",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=None,
        help="Global max_position_count used for the run (enables slot occupancy %%)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: logs/capital_utilization_reports/...)",
    )
    return parser.parse_args()


def _load_trades(csv_path: Path) -> tuple[pandas.DataFrame, str]:
    trade_frame = pandas.read_csv(csv_path)
    if "bucket" in trade_frame.columns:
        bucket_column = "bucket"
    elif "set" in trade_frame.columns:
        bucket_column = "set"
    else:
        raise ValueError(
            "input CSV must contain either a 'bucket' or 'set' column"
        )
    trade_frame["entry_date"] = pandas.to_datetime(trade_frame["entry_date"])
    trade_frame["exit_date"] = pandas.to_datetime(trade_frame["exit_date"])
    trade_frame = trade_frame.dropna(subset=["entry_date", "exit_date"]).copy()
    return trade_frame, bucket_column


def _concurrent_series(
    trades: pandas.DataFrame,
    business_day_index: pandas.DatetimeIndex,
) -> pandas.Series:
    """Return a business-day indexed series of concurrent open positions.

    Uses the same ``[entry_date, exit_date)`` half-open interval convention
    as the simulator's event loop: on the exit date the slot is released
    BEFORE any new entry on that same date is considered. This matches the
    simulator's ``max concurrent positions`` reading.
    """
    if trades.empty:
        return pandas.Series(0, index=business_day_index, dtype="int64")
    entries = trades["entry_date"].value_counts().rename("delta_in")
    exits = trades["exit_date"].value_counts().rename("delta_out")

    delta_series = pandas.Series(0, index=business_day_index, dtype="int64")
    for event_date, count in entries.items():
        if event_date in delta_series.index:
            delta_series.at[event_date] += int(count)
        else:
            matching = business_day_index[business_day_index >= event_date]
            if len(matching):
                delta_series.at[matching[0]] += int(count)
    for event_date, count in exits.items():
        if event_date in delta_series.index:
            delta_series.at[event_date] -= int(count)
        else:
            matching = business_day_index[business_day_index >= event_date]
            if len(matching):
                delta_series.at[matching[0]] -= int(count)
    return delta_series.cumsum()


def build_utilization_report(
    trade_frame: pandas.DataFrame,
    bucket_column: str,
    max_positions: int | None,
) -> pandas.DataFrame:
    if trade_frame.empty:
        raise ValueError("no trades to analyse")

    overall_start = trade_frame["entry_date"].min()
    overall_end = trade_frame["exit_date"].max()
    business_day_index = pandas.bdate_range(overall_start, overall_end)
    if business_day_index.empty:
        raise ValueError("could not build business day calendar from trade dates")

    bucket_labels = sorted(trade_frame[bucket_column].dropna().unique().tolist())

    # Per-scope concurrent series. "Total" = all trades combined.
    scope_series: dict[str, pandas.Series] = {
        "Total": _concurrent_series(trade_frame, business_day_index),
    }
    for label in bucket_labels:
        scope_series[label] = _concurrent_series(
            trade_frame[trade_frame[bucket_column] == label],
            business_day_index,
        )

    years = sorted({date.year for date in business_day_index})
    rows: list[dict] = []
    for year in years:
        year_mask = business_day_index.year == year
        year_bdays = int(year_mask.sum())
        for scope_name, series in scope_series.items():
            year_series = series[year_mask]
            if year_series.empty:
                avg_concurrent = 0.0
                peak_concurrent = 0
            else:
                avg_concurrent = float(year_series.mean())
                peak_concurrent = int(year_series.max())
            if scope_name == "Total":
                trade_count = int(
                    (trade_frame["entry_date"].dt.year == year).sum()
                )
            else:
                trade_count = int(
                    (
                        (trade_frame[bucket_column] == scope_name)
                        & (trade_frame["entry_date"].dt.year == year)
                    ).sum()
                )
            row = {
                "year": year,
                "scope": scope_name,
                "trades_in_year": trade_count,
                "avg_concurrent": round(avg_concurrent, 3),
                "peak_concurrent": peak_concurrent,
                "year_bdays": year_bdays,
            }
            if max_positions is not None and max_positions > 0:
                row["slot_occupancy_pct"] = round(
                    avg_concurrent / max_positions * 100, 2
                )
                row["peak_occupancy_pct"] = round(
                    peak_concurrent / max_positions * 100, 2
                )
            rows.append(row)

    # Overall summary (across all years) per scope
    for scope_name, series in scope_series.items():
        if series.empty:
            avg_concurrent = 0.0
            peak_concurrent = 0
        else:
            avg_concurrent = float(series.mean())
            peak_concurrent = int(series.max())
        if scope_name == "Total":
            trade_count = int(len(trade_frame))
        else:
            trade_count = int((trade_frame[bucket_column] == scope_name).sum())
        row = {
            "year": "ALL",
            "scope": scope_name,
            "trades_in_year": trade_count,
            "avg_concurrent": round(avg_concurrent, 3),
            "peak_concurrent": peak_concurrent,
            "year_bdays": int(len(business_day_index)),
        }
        if max_positions is not None and max_positions > 0:
            row["slot_occupancy_pct"] = round(
                avg_concurrent / max_positions * 100, 2
            )
            row["peak_occupancy_pct"] = round(
                peak_concurrent / max_positions * 100, 2
            )
        rows.append(row)

    return pandas.DataFrame(rows)


def main() -> None:
    arguments = _parse_arguments()
    csv_path = arguments.csv_path
    if not csv_path.exists():
        print(f"error: CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    trade_frame, bucket_column = _load_trades(csv_path)
    report_frame = build_utilization_report(
        trade_frame, bucket_column, arguments.max_positions
    )

    if arguments.output is not None:
        output_path = arguments.output
    else:
        output_directory = PROJECT_ROOT / "logs" / "capital_utilization_reports"
        output_directory.mkdir(parents=True, exist_ok=True)
        timestamp_string = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = (
            output_directory
            / f"capital_utilization_{csv_path.stem}_{timestamp_string}.csv"
        )
    report_frame.to_csv(output_path, index=False)
    print(f"bucket_column: {bucket_column}")
    print(f"rows: {len(report_frame)}")
    print(f"written: {output_path}")


if __name__ == "__main__":
    main()
