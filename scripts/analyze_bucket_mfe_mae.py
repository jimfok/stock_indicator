"""Bucket-level MFE / MAE analysis for complex_simulation trade detail CSVs.

Reads a trade_detail CSV that contains ``max_favorable_excursion_pct`` and
``max_adverse_excursion_pct`` columns (produced after the 2026-04-10 simulator
changes) and reports, for each of the six edge buckets identified in
project_multi_bucket_approach_2026_04.md:

- bucket size, win rate, mean profit, mean loss, net mean vs overall baseline
- winner MAE distribution percentiles  (how far winners dipped before winning)
- loser  MFE distribution percentiles  (how high losers flew before losing)
- a suggested stop-loss floor and take-profit ceiling derived from those
  percentiles
- an overlap diagnostic that flags buckets whose winner-MAE and loser-MAE
  distributions are not separable (edge cannot be rescued by SL alone)

Usage:
    python -m scripts.analyze_bucket_mfe_mae <path-to-trade-detail.csv>
    python -m scripts.analyze_bucket_mfe_mae  # defaults to latest file
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULT_DIRECTORY = PROJECT_ROOT / "logs" / "complex_simulation_result"


@dataclass(frozen=True)
class BucketDefinition:
    """Describe a single entry-condition bucket for analysis."""

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


def _pct(value: float) -> str:
    return f"{value * 100:+.2f}%"


def _safe_quantile(series: pandas.Series, q: float) -> float | None:
    cleaned = series.dropna()
    if cleaned.empty:
        return None
    return float(cleaned.quantile(q))


def _format_optional(value: float | None, formatter: Callable[[float], str]) -> str:
    if value is None:
        return "   n/a  "
    return formatter(value)


def _latest_csv(directory: Path) -> Path | None:
    candidates = sorted(
        directory.glob("complex_simulation_*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def analyse_bucket(
    trade_frame: pandas.DataFrame,
    bucket: BucketDefinition,
    baseline_mean_pct: float,
    baseline_win_rate: float,
) -> None:
    column = bucket.column
    if column not in trade_frame.columns:
        print(f"\n== Bucket {bucket.bucket_id}: {bucket.label} ==")
        print(f"  column '{column}' not found in CSV, skipping")
        return

    mask = (
        trade_frame[column].between(bucket.lower_bound, bucket.upper_bound, inclusive="both")
        & trade_frame[column].notna()
    )
    bucket_frame = trade_frame.loc[mask]
    if bucket_frame.empty:
        print(f"\n== Bucket {bucket.bucket_id}: {bucket.label} ==")
        print(f"  range=[{bucket.lower_bound}, {bucket.upper_bound}] -> 0 trades")
        return

    winners = bucket_frame[bucket_frame["result"] == "win"]
    losers = bucket_frame[bucket_frame["result"] == "lose"]

    bucket_mean = float(bucket_frame["percentage_change"].mean())
    bucket_wr = float((bucket_frame["result"] == "win").mean())
    winner_mean_profit = float(winners["percentage_change"].mean()) if not winners.empty else 0.0
    loser_mean_loss = float(losers["percentage_change"].mean()) if not losers.empty else 0.0

    print(f"\n== Bucket {bucket.bucket_id}: {bucket.label} ==")
    print(
        f"  range={column} ∈ [{bucket.lower_bound}, {bucket.upper_bound}] "
        f"| n={len(bucket_frame)} ({len(bucket_frame)/len(trade_frame):.1%} of all)"
    )
    print(
        f"  win_rate={bucket_wr:.2%}  (baseline {baseline_win_rate:.2%}, "
        f"delta {bucket_wr - baseline_win_rate:+.2%})"
    )
    print(
        f"  mean_trade={_pct(bucket_mean)}  "
        f"(baseline {_pct(baseline_mean_pct)}, "
        f"x{bucket_mean / baseline_mean_pct:.2f} of baseline)"
        if baseline_mean_pct
        else f"  mean_trade={_pct(bucket_mean)}"
    )
    print(
        f"  winner_mean_profit={_pct(winner_mean_profit)}  "
        f"loser_mean_loss={_pct(loser_mean_loss)}  "
        f"(winners={len(winners)}, losers={len(losers)})"
    )

    # Winner MAE: how deep winners dipped before winning.
    # MAE is negative. More-negative percentiles = deeper dips.
    winner_mae_p10 = _safe_quantile(winners["max_adverse_excursion_pct"], 0.10)
    winner_mae_p20 = _safe_quantile(winners["max_adverse_excursion_pct"], 0.20)
    winner_mae_p50 = _safe_quantile(winners["max_adverse_excursion_pct"], 0.50)
    winner_mae_p90 = _safe_quantile(winners["max_adverse_excursion_pct"], 0.90)

    # Winner MFE: how high winners actually flew (the upside we want to keep).
    winner_mfe_p10 = _safe_quantile(winners["max_favorable_excursion_pct"], 0.10)
    winner_mfe_p50 = _safe_quantile(winners["max_favorable_excursion_pct"], 0.50)
    winner_mfe_p80 = _safe_quantile(winners["max_favorable_excursion_pct"], 0.80)
    winner_mfe_p90 = _safe_quantile(winners["max_favorable_excursion_pct"], 0.90)

    # Loser MFE: how high losers flew before losing.
    loser_mfe_p50 = _safe_quantile(losers["max_favorable_excursion_pct"], 0.50)
    loser_mfe_p80 = _safe_quantile(losers["max_favorable_excursion_pct"], 0.80)
    loser_mfe_p90 = _safe_quantile(losers["max_favorable_excursion_pct"], 0.90)

    # For overlap diagnostic: loser MAE — do losers dip the same way winners do?
    loser_mae_p10 = _safe_quantile(losers["max_adverse_excursion_pct"], 0.10)
    loser_mae_p50 = _safe_quantile(losers["max_adverse_excursion_pct"], 0.50)

    print("  -- winner MAE (how deep winners dipped before turning) --")
    print(
        f"    p10={_format_optional(winner_mae_p10, _pct)} "
        f"p20={_format_optional(winner_mae_p20, _pct)} "
        f"p50={_format_optional(winner_mae_p50, _pct)} "
        f"p90={_format_optional(winner_mae_p90, _pct)}"
    )
    print("  -- winner MFE (how high winners actually flew — upside to keep) --")
    print(
        f"    p10={_format_optional(winner_mfe_p10, _pct)} "
        f"p50={_format_optional(winner_mfe_p50, _pct)} "
        f"p80={_format_optional(winner_mfe_p80, _pct)} "
        f"p90={_format_optional(winner_mfe_p90, _pct)}"
    )
    print("  -- loser MFE (how high losers flew before rolling over) --")
    print(
        f"    p50={_format_optional(loser_mfe_p50, _pct)} "
        f"p80={_format_optional(loser_mfe_p80, _pct)} "
        f"p90={_format_optional(loser_mfe_p90, _pct)}"
    )
    print("  -- loser MAE (for SL overlap check) --")
    print(
        f"    p10={_format_optional(loser_mae_p10, _pct)} "
        f"p50={_format_optional(loser_mae_p50, _pct)}"
    )

    # Suggested SL: loose enough to not cut the bottom 10-20% of winners.
    # Choose the more-conservative of the two (the less negative one).
    suggested_sl_candidates = [v for v in (winner_mae_p10, winner_mae_p20) if v is not None]
    if suggested_sl_candidates:
        suggested_sl_magnitude = abs(min(suggested_sl_candidates))
    else:
        suggested_sl_magnitude = None

    # Suggested TP: tight enough to catch the top 80-90% of losers' peaks.
    suggested_tp_candidates = [v for v in (loser_mfe_p80, loser_mfe_p90) if v is not None]
    if suggested_tp_candidates:
        suggested_tp_magnitude = min(suggested_tp_candidates)
    else:
        suggested_tp_magnitude = None

    # SL overlap diagnostic:
    # count losers whose MAE is DEEPER than winner_mae_p20. A bucket is only
    # SL-separable if many losers dip below winner_mae_p20 while few winners
    # do (winners by construction have only 20% below their own p20).
    sl_separable_fraction: float | None = None
    if winner_mae_p20 is not None and not losers.empty:
        losers_below = (losers["max_adverse_excursion_pct"] < winner_mae_p20).sum()
        sl_separable_fraction = losers_below / len(losers)

    # TP overlap diagnostic:
    # If TP is set at loser_mfe_p80, how many winners would ALSO be capped at
    # that level (i.e. winners whose MFE is ABOVE loser_mfe_p80 — these are
    # the winners whose upside would be truncated). Report the fraction of
    # winners whose MFE exceeds the TP ceiling = winners preserved, and the
    # fraction of winners that would be cut exactly where losers peak.
    tp_winner_preserved_fraction: float | None = None
    tp_winner_clipped_fraction: float | None = None
    if suggested_tp_magnitude is not None and not winners.empty:
        winner_mfe_series = winners["max_favorable_excursion_pct"].dropna()
        if not winner_mfe_series.empty:
            # Winners who ever reached TP level = their tail would be clipped
            # but their realised profit (at least = TP) is still positive.
            # Winners who never reached TP would not be affected by TP at all.
            reached_tp = (winner_mfe_series >= suggested_tp_magnitude).sum()
            tp_winner_preserved_fraction = reached_tp / len(winner_mfe_series)
            tp_winner_clipped_fraction = tp_winner_preserved_fraction  # alias

    # Peak separation: compare winner MFE p50 vs loser MFE p80.
    # If winners' median peak is clearly above where 80% of losers peak,
    # a TP placed at loser_mfe_p80 truncates only a small fraction of winners.
    peak_separation_ratio: float | None = None
    if winner_mfe_p50 is not None and loser_mfe_p80 is not None and loser_mfe_p80 > 0:
        peak_separation_ratio = winner_mfe_p50 / loser_mfe_p80

    print("  -- suggested thresholds --")
    if suggested_sl_magnitude is not None:
        print(f"    SL ~ {suggested_sl_magnitude * 100:.2f}%  (winner MAE p10/p20 floor)")
    else:
        print("    SL ~ n/a")
    if suggested_tp_magnitude is not None:
        print(f"    TP ~ {suggested_tp_magnitude * 100:.2f}%  (loser MFE p80/p90 ceiling)")
    else:
        print("    TP ~ n/a")

    if sl_separable_fraction is not None:
        marker = ""
        if sl_separable_fraction < 0.20:
            marker = "  [!! OVERLAP HEAVY — SL unlikely to help]"
        elif sl_separable_fraction < 0.40:
            marker = "  [!  overlap moderate]"
        print(
            f"    SL-separability: {sl_separable_fraction:.1%} of losers dip "
            f"deeper than winner MAE p20{marker}"
        )

    if tp_winner_clipped_fraction is not None:
        marker = ""
        if tp_winner_clipped_fraction > 0.60:
            marker = "  [!! TP will clip most winners' tails]"
        elif tp_winner_clipped_fraction > 0.35:
            marker = "  [!  TP clips a meaningful portion of winners]"
        print(
            f"    TP-clip: {tp_winner_clipped_fraction:.1%} of winners reached TP "
            f"level before exit (their upside tail would be truncated){marker}"
        )

    if peak_separation_ratio is not None:
        marker = ""
        if peak_separation_ratio < 0.8:
            marker = "  [!! winners median peak BELOW loser p80 — TP unreliable]"
        elif peak_separation_ratio < 1.2:
            marker = "  [!  winners and losers peak in the same zone]"
        elif peak_separation_ratio >= 1.5:
            marker = "  [OK winners peak well above loser zone]"
        print(
            f"    peak-separation: winner_MFE_p50 / loser_MFE_p80 = "
            f"{peak_separation_ratio:.2f}{marker}"
        )


def run(csv_path: Path) -> None:
    if not csv_path.exists():
        print(f"error: CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    trade_frame = pandas.read_csv(csv_path)
    required_columns = {
        "result",
        "percentage_change",
        "max_favorable_excursion_pct",
        "max_adverse_excursion_pct",
    }
    missing = required_columns - set(trade_frame.columns)
    if missing:
        print(f"error: CSV missing columns: {sorted(missing)}", file=sys.stderr)
        sys.exit(1)

    # Treat trades with missing result as excluded from analysis.
    trade_frame = trade_frame.dropna(subset=["result", "percentage_change"])
    # Focus on set A (the bucket analysis is per-strategy-set).
    if "set" in trade_frame.columns:
        set_values = sorted(trade_frame["set"].dropna().unique())
        print(f"# sets present: {set_values}")

    baseline_mean_pct = float(trade_frame["percentage_change"].mean())
    baseline_win_rate = float((trade_frame["result"] == "win").mean())
    print(f"# file: {csv_path}")
    print(f"# total trades: {len(trade_frame)}")
    print(f"# baseline win_rate={baseline_win_rate:.2%} mean_trade={_pct(baseline_mean_pct)}")

    for bucket in BUCKETS:
        analyse_bucket(trade_frame, bucket, baseline_mean_pct, baseline_win_rate)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_path = Path(sys.argv[1])
    else:
        latest = _latest_csv(DEFAULT_RESULT_DIRECTORY)
        if latest is None:
            print(
                f"error: no CSV found in {DEFAULT_RESULT_DIRECTORY}; "
                "pass a path explicitly.",
                file=sys.stderr,
            )
            sys.exit(1)
        target_path = latest
    run(target_path)
