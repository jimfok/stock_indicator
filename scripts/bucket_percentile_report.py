"""Dump the full MFE/MAE/pct_change percentile ladders per bucket.

Reads a trade_detail CSV produced by an all-pass (s99) run and, for each of
the six edge buckets defined in project_multi_bucket_approach_2026_04.md,
computes a wide percentile table (p05 ... p95) for:

- winner MAE (how deep winners dipped before winning)
- winner MFE (how high winners actually flew)
- loser  MAE (how deep losers went)
- loser  MFE (how high losers flew before rolling over)
- all-trade percentage_change (final realised P&L)

Values are expressed as percentage numbers (i.e. -10.92 means -10.92%).

Usage:
    python -m scripts.bucket_percentile_report <trade-detail-csv> [output-csv]

If output-csv is omitted, writes to
logs/bucket_percentile_<stem>_<timestamp>.csv next to the input file.
"""

from __future__ import annotations

import datetime
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class BucketDefinition:
    bucket_id: int
    label: str
    column: str
    lower_bound: float
    upper_bound: float


BUCKETS: list[BucketDefinition] = [
    BucketDefinition(1, "near_price_volume_ratio", "near_price_volume_ratio", -0.001, 0.078),
    BucketDefinition(2, "sma_angle (negative)", "sma_angle", -5.84, -0.652),
    BucketDefinition(3, "ema_angle (negative)", "ema_angle", -5.455, -0.601),
    BucketDefinition(4, "d_ema_angle (mild up)", "d_ema_angle", 0.031, 0.234),
    BucketDefinition(5, "price_concentration_score (low)", "price_concentration_score", 0.031, 0.045),
    BucketDefinition(6, "above_price_volume_ratio (high)", "above_price_volume_ratio", 0.943, 0.973),
]


PERCENTILES: list[int] = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95]


def _percentile_row(
    series: pandas.Series,
    *,
    bucket_id: int,
    bucket_label: str,
    population: str,
    metric: str,
) -> dict:
    values = series.dropna()
    row: dict = {
        "bucket_id": bucket_id,
        "bucket": bucket_label,
        "population": population,
        "metric": metric,
        "n": int(len(values)),
    }
    if values.empty:
        row["mean"] = None
        for percentile in PERCENTILES:
            row[f"p{percentile:02d}"] = None
        return row
    row["mean"] = round(float(values.mean()) * 100, 4)
    for percentile in PERCENTILES:
        row[f"p{percentile:02d}"] = round(
            float(values.quantile(percentile / 100)) * 100, 4
        )
    return row


def build_percentile_report(trade_frame: pandas.DataFrame) -> pandas.DataFrame:
    required_columns = {
        "result",
        "percentage_change",
        "max_favorable_excursion_pct",
        "max_adverse_excursion_pct",
    }
    missing = required_columns - set(trade_frame.columns)
    if missing:
        raise ValueError(f"input CSV missing columns: {sorted(missing)}")

    trade_frame = trade_frame.dropna(subset=["result", "percentage_change"]).copy()
    output_rows: list[dict] = []

    # Global baseline (all trades, no bucket filter)
    baseline_winners = trade_frame[trade_frame["result"] == "win"]
    baseline_losers = trade_frame[trade_frame["result"] == "lose"]
    baseline_definitions = [
        ("ALL", "all", "percentage_change", trade_frame["percentage_change"]),
        ("ALL", "winners", "MAE", baseline_winners["max_adverse_excursion_pct"]),
        ("ALL", "winners", "MFE", baseline_winners["max_favorable_excursion_pct"]),
        ("ALL", "losers", "MAE", baseline_losers["max_adverse_excursion_pct"]),
        ("ALL", "losers", "MFE", baseline_losers["max_favorable_excursion_pct"]),
    ]
    for bucket_label, population, metric, series in baseline_definitions:
        output_rows.append(
            _percentile_row(
                series,
                bucket_id=0,
                bucket_label=bucket_label,
                population=population,
                metric=metric,
            )
        )

    for bucket in BUCKETS:
        if bucket.column not in trade_frame.columns:
            continue
        mask = (
            trade_frame[bucket.column].between(
                bucket.lower_bound, bucket.upper_bound, inclusive="both"
            )
            & trade_frame[bucket.column].notna()
        )
        bucket_frame = trade_frame.loc[mask]
        winners = bucket_frame[bucket_frame["result"] == "win"]
        losers = bucket_frame[bucket_frame["result"] == "lose"]
        for population, metric, series in [
            ("all", "percentage_change", bucket_frame["percentage_change"]),
            ("winners", "MAE", winners["max_adverse_excursion_pct"]),
            ("winners", "MFE", winners["max_favorable_excursion_pct"]),
            ("losers", "MAE", losers["max_adverse_excursion_pct"]),
            ("losers", "MFE", losers["max_favorable_excursion_pct"]),
        ]:
            output_rows.append(
                _percentile_row(
                    series,
                    bucket_id=bucket.bucket_id,
                    bucket_label=bucket.label,
                    population=population,
                    metric=metric,
                )
            )

    column_order = ["bucket_id", "bucket", "population", "metric", "n", "mean"] + [
        f"p{percentile:02d}" for percentile in PERCENTILES
    ]
    return pandas.DataFrame(output_rows)[column_order]


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "usage: python -m scripts.bucket_percentile_report "
            "<trade-detail-csv> [output-csv]",
            file=sys.stderr,
        )
        sys.exit(1)
    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"error: input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if len(sys.argv) >= 3:
        output_path = Path(sys.argv[2])
    else:
        output_directory = PROJECT_ROOT / "logs" / "bucket_percentile_reports"
        output_directory.mkdir(parents=True, exist_ok=True)
        timestamp_string = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = (
            output_directory / f"bucket_percentile_{input_path.stem}_{timestamp_string}.csv"
        )

    trade_frame = pandas.read_csv(input_path)
    report_frame = build_percentile_report(trade_frame)
    report_frame.to_csv(output_path, index=False)
    print(f"rows: {len(report_frame)}")
    print(f"written: {output_path}")
    print(
        "columns: bucket_id, bucket, population, metric, n, mean, p05..p95 "
        "(values in percent)"
    )


if __name__ == "__main__":
    main()
