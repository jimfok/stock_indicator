"""Joint-replay SL/TP grid search over a trade_detail CSV.

For each (stop_loss, take_profit) candidate pair and each edge bucket, this
script walks every trade in the input CSV and uses its ACTUAL per-trade
(max_adverse_excursion_pct, max_favorable_excursion_pct,
max_adverse_excursion_date, max_favorable_excursion_date, percentage_change)
to determine the trade's outcome under the candidate SL/TP.

This is a JOINT analysis — no independence assumption between MAE and MFE.
Every trade carries its own full trajectory extremes, so the evaluation
respects the actual co-movement of winner dips and winner peaks, and of
loser peaks and loser dips.

Usage:
    python -m scripts.replay_sltp_grid_search <trade-detail-csv> \
        [--output PATH] [--sl-grid 0,0.05,0.08,0.10,0.12,0.15,0.20,0.30] \
        [--tp-grid 0,0.05,0.08,0.10,0.12,0.15,0.20,0.30,0.50]

SL / TP grid values are absolute fractions. A value of ``0`` means the
corresponding mechanism is disabled (simulator semantics: SL=0 or SL>=1
never fires; TP=0 or TP>=1 never fires).
"""

from __future__ import annotations

import argparse
import datetime
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class BucketDefinition:
    bucket_id: int
    label: str
    column: str | None
    lower_bound: float
    upper_bound: float


# bucket_id=0 means "ALL trades, no filter" (baseline).
BUCKETS: list[BucketDefinition] = [
    BucketDefinition(0, "ALL", None, 0.0, 0.0),
    BucketDefinition(1, "near_price_volume_ratio", "near_price_volume_ratio", -0.001, 0.078),
    BucketDefinition(2, "sma_angle (negative)", "sma_angle", -5.84, -0.652),
    BucketDefinition(3, "ema_angle (negative)", "ema_angle", -5.455, -0.601),
    BucketDefinition(4, "d_ema_angle (mild up)", "d_ema_angle", 0.031, 0.234),
    BucketDefinition(5, "price_concentration_score (low)", "price_concentration_score", 0.031, 0.045),
    BucketDefinition(6, "above_price_volume_ratio (high)", "above_price_volume_ratio", 0.943, 0.973),
]


DEFAULT_SL_GRID: list[float] = [0.0, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30]
DEFAULT_TP_GRID: list[float] = [0.0, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.50]


def replay_trade(
    mae_pct: float | None,
    mfe_pct: float | None,
    mae_ordinal: int | None,
    mfe_ordinal: int | None,
    pct_change: float,
    stop_loss: float,
    take_profit: float,
) -> tuple[float, str]:
    """Return ``(replayed_pct_change, exit_reason)`` under the given SL/TP.

    The decision tree:

    1. If neither SL nor TP would be triggered, the trade exits via the
       original signal with its original ``percentage_change``.
    2. If only one side would be triggered, that side fires.
    3. If both would be triggered, the earlier excursion date wins:
       - ``mae_ordinal < mfe_ordinal`` → SL fires first
       - ``mfe_ordinal < mae_ordinal`` → TP fires first
       - tie → use the sign of the original ``percentage_change`` as a
         proxy for which extreme the bar closed nearer to.

    ``stop_loss``/``take_profit`` are disabled when the value is outside
    ``(0, 1)`` (matching the simulator's convention).
    """
    sl_active = 0 < stop_loss < 1
    tp_active = 0 < take_profit < 1

    # Normalise nan MAE/MFE to "did not dip / did not rise"
    if mae_pct is None or (isinstance(mae_pct, float) and math.isnan(mae_pct)):
        mae_pct = 0.0
    if mfe_pct is None or (isinstance(mfe_pct, float) and math.isnan(mfe_pct)):
        mfe_pct = 0.0

    hit_sl = sl_active and mae_pct <= -stop_loss
    hit_tp = tp_active and mfe_pct >= take_profit

    if not hit_sl and not hit_tp:
        return pct_change, "signal"
    if hit_sl and not hit_tp:
        return -stop_loss, "stop_loss"
    if hit_tp and not hit_sl:
        return take_profit, "take_profit"

    # Both hit — compare dates.
    if mae_ordinal is None and mfe_ordinal is None:
        # No date info; use sign of pct_change as tiebreaker.
        if pct_change >= 0:
            return take_profit, "take_profit"
        return -stop_loss, "stop_loss"
    if mae_ordinal is None:
        return take_profit, "take_profit"
    if mfe_ordinal is None:
        return -stop_loss, "stop_loss"
    if mae_ordinal < mfe_ordinal:
        return -stop_loss, "stop_loss"
    if mfe_ordinal < mae_ordinal:
        return take_profit, "take_profit"
    # Same date — tiebreak by original outcome.
    if pct_change >= 0:
        return take_profit, "take_profit"
    return -stop_loss, "stop_loss"


def _prepare_trades(trade_frame: pandas.DataFrame) -> pandas.DataFrame:
    required = {
        "result",
        "percentage_change",
        "max_favorable_excursion_pct",
        "max_adverse_excursion_pct",
    }
    missing = required - set(trade_frame.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")
    trade_frame = trade_frame.dropna(subset=["result", "percentage_change"]).copy()

    # Convert MAE/MFE dates into pandas Timestamps if present; else leave None.
    for column_name in ("max_adverse_excursion_date", "max_favorable_excursion_date"):
        if column_name in trade_frame.columns:
            trade_frame[column_name] = pandas.to_datetime(
                trade_frame[column_name], errors="coerce"
            )
    return trade_frame


def evaluate_bucket(
    bucket_trades: pandas.DataFrame,
    stop_loss: float,
    take_profit: float,
) -> dict:
    """Run the replay on a set of trades and return aggregate stats."""
    if bucket_trades.empty:
        return {
            "n": 0,
            "n_signal": 0,
            "n_sl_hit": 0,
            "n_tp_hit": 0,
            "wr": None,
            "mean_profit_pct": None,
            "mean_loss_pct": None,
            "pl_ratio": None,
            "mean_trade_pct": None,
            "median_trade_pct": None,
        }

    has_mae_date = "max_adverse_excursion_date" in bucket_trades.columns
    has_mfe_date = "max_favorable_excursion_date" in bucket_trades.columns

    replayed_pct_list: list[float] = []
    exit_reason_counter = {"signal": 0, "stop_loss": 0, "take_profit": 0}
    for _, row in bucket_trades.iterrows():
        mae_pct = row.get("max_adverse_excursion_pct")
        mfe_pct = row.get("max_favorable_excursion_pct")
        pct_change = float(row["percentage_change"])
        if has_mae_date:
            value = row.get("max_adverse_excursion_date")
            mae_ordinal = value.toordinal() if pandas.notna(value) else None
        else:
            mae_ordinal = None
        if has_mfe_date:
            value = row.get("max_favorable_excursion_date")
            mfe_ordinal = value.toordinal() if pandas.notna(value) else None
        else:
            mfe_ordinal = None

        replayed, reason = replay_trade(
            mae_pct=float(mae_pct) if mae_pct is not None and not pandas.isna(mae_pct) else None,
            mfe_pct=float(mfe_pct) if mfe_pct is not None and not pandas.isna(mfe_pct) else None,
            mae_ordinal=mae_ordinal,
            mfe_ordinal=mfe_ordinal,
            pct_change=pct_change,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        replayed_pct_list.append(replayed)
        exit_reason_counter[reason] += 1

    replayed_series = pandas.Series(replayed_pct_list)
    winners = replayed_series[replayed_series > 0]
    losers = replayed_series[replayed_series < 0]
    n = len(replayed_series)
    wr = len(winners) / n if n else 0.0
    mean_profit = float(winners.mean()) if not winners.empty else 0.0
    mean_loss = float(abs(losers.mean())) if not losers.empty else 0.0
    pl_ratio = (mean_profit / mean_loss) if mean_loss > 0 else None
    return {
        "n": int(n),
        "n_signal": int(exit_reason_counter["signal"]),
        "n_sl_hit": int(exit_reason_counter["stop_loss"]),
        "n_tp_hit": int(exit_reason_counter["take_profit"]),
        "wr": round(wr * 100, 2),
        "mean_profit_pct": round(mean_profit * 100, 4),
        "mean_loss_pct": round(mean_loss * 100, 4),
        "pl_ratio": round(pl_ratio, 4) if pl_ratio is not None else None,
        "mean_trade_pct": round(float(replayed_series.mean()) * 100, 4),
        "median_trade_pct": round(float(replayed_series.median()) * 100, 4),
    }


def run_grid(
    trade_frame: pandas.DataFrame,
    sl_grid: list[float],
    tp_grid: list[float],
) -> pandas.DataFrame:
    rows: list[dict] = []
    for bucket in BUCKETS:
        if bucket.bucket_id == 0:
            bucket_trades = trade_frame
        else:
            if bucket.column not in trade_frame.columns:
                continue
            mask = (
                trade_frame[bucket.column].between(
                    bucket.lower_bound, bucket.upper_bound, inclusive="both"
                )
                & trade_frame[bucket.column].notna()
            )
            bucket_trades = trade_frame.loc[mask]
        for stop_loss in sl_grid:
            for take_profit in tp_grid:
                stats = evaluate_bucket(bucket_trades, stop_loss, take_profit)
                rows.append(
                    {
                        "bucket_id": bucket.bucket_id,
                        "bucket": bucket.label,
                        "sl": round(stop_loss, 4),
                        "tp": round(take_profit, 4),
                        **stats,
                    }
                )
    column_order = [
        "bucket_id",
        "bucket",
        "sl",
        "tp",
        "n",
        "n_signal",
        "n_sl_hit",
        "n_tp_hit",
        "wr",
        "mean_profit_pct",
        "mean_loss_pct",
        "pl_ratio",
        "mean_trade_pct",
        "median_trade_pct",
    ]
    return pandas.DataFrame(rows)[column_order]


def _parse_grid(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part.strip() != ""]


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--sl-grid", type=str, default=None)
    parser.add_argument("--tp-grid", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_arguments()
    if not arguments.csv_path.exists():
        print(f"error: CSV not found: {arguments.csv_path}", file=sys.stderr)
        sys.exit(1)
    sl_grid = _parse_grid(arguments.sl_grid) if arguments.sl_grid else DEFAULT_SL_GRID
    tp_grid = _parse_grid(arguments.tp_grid) if arguments.tp_grid else DEFAULT_TP_GRID

    trade_frame = pandas.read_csv(arguments.csv_path)
    trade_frame = _prepare_trades(trade_frame)
    report_frame = run_grid(trade_frame, sl_grid, tp_grid)

    if arguments.output is not None:
        output_path = arguments.output
    else:
        output_directory = PROJECT_ROOT / "logs" / "sltp_grid_reports"
        output_directory.mkdir(parents=True, exist_ok=True)
        timestamp_string = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = (
            output_directory
            / f"sltp_grid_{arguments.csv_path.stem}_{timestamp_string}.csv"
        )
    report_frame.to_csv(output_path, index=False)
    print(f"rows: {len(report_frame)}")
    print(f"buckets: {report_frame['bucket'].nunique()}")
    print(f"sl grid: {sl_grid}")
    print(f"tp grid: {tp_grid}")
    print(f"written: {output_path}")


if __name__ == "__main__":
    main()
