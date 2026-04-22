"""Grid search over sell_window × exit_alpha_factor for B6 exit tuning.

Iterates over sell_window [3..10] × exit_alpha_factor [2..6] and reports
trade-level P&L statistics. No portfolio simulation — only individual
trade metrics.

Usage:
    python -m scripts.grid_search_exit_alpha
"""

import sys
import time
from pathlib import Path

import pandas

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stock_indicator import strategy  # noqa: E402

DATA_DIRECTORY = PROJECT_ROOT / "data"
STOCK_DATA_DIRECTORY = DATA_DIRECTORY / "stock_data"


def run_grid_search() -> None:
    data_directory = (
        STOCK_DATA_DIRECTORY if STOCK_DATA_DIRECTORY.exists() else DATA_DIRECTORY
    )
    start_date = pandas.Timestamp("2014-01-01")

    # B6 buy: window=4, above_pv_ratio 0.943-0.973, all other filters off
    buy_name = "ema_sma_cross_testing_4_-99_99_-99.0,99.0_0.943,0.973"

    results = []

    # Also run baseline (no exit_alpha_factor) for each sell window
    for sell_w in range(3, 11):
        for alpha_f in [None, 2, 3, 4, 5, 6]:
            sell_name = (
                f"ema_sma_cross_testing_{sell_w}"
                "_-0.01_65_-10.0,10.0_0.78,1.00"
            )

            t0 = time.time()
            try:
                artifacts = strategy._generate_strategy_evaluation_artifacts(
                    data_directory,
                    buy_name,
                    sell_name,
                    minimum_average_dollar_volume=None,
                    minimum_average_dollar_volume_ratio=0.0005,
                    top_dollar_volume_rank=200,
                    maximum_symbols_per_group=5,
                    start_date=start_date,
                    maximum_position_count=30,
                    use_confirmation_angle=False,
                    confirmation_entry_mode="market",
                    minimum_holding_bars=5,
                    exit_alpha_factor=float(alpha_f) if alpha_f is not None else None,
                )
            except Exception as exc:
                label = f"sell={sell_w} alpha={alpha_f or 'base'}"
                print(f"{label}: ERROR {exc}")
                continue

            all_trades = artifacts.trades
            elapsed = time.time() - t0

            label = f"sell={sell_w} alpha={alpha_f or 'base'}"
            if not all_trades:
                print(f"{label}: 0 trades ({elapsed:.0f}s)")
                continue

            wins = [t for t in all_trades if t.profit > 0]
            losses = [t for t in all_trades if t.profit <= 0]

            win_pcts = [
                (t.exit_price - t.entry_price) / t.entry_price for t in wins
            ]
            loss_pcts = [
                (t.exit_price - t.entry_price) / t.entry_price for t in losses
            ]

            n = len(all_trades)
            wr = len(wins) / n if n else 0
            mean_profit = sum(win_pcts) / len(win_pcts) if win_pcts else 0
            mean_loss = sum(loss_pcts) / len(loss_pcts) if loss_pcts else 0
            pl = abs(mean_profit / mean_loss) if mean_loss else 0
            avg_hold = sum(t.holding_period for t in all_trades) / n

            results.append({
                "sell_w": sell_w,
                "alpha_f": alpha_f,
                "trades": n,
                "wr": wr,
                "mean_profit": mean_profit,
                "mean_loss": mean_loss,
                "pl": pl,
                "avg_hold": avg_hold,
            })

            print(
                f"{label}: "
                f"n={n:4d} wr={wr:.1%} "
                f"mean_profit={mean_profit:+.2%} mean_loss={mean_loss:+.2%} "
                f"P/L={pl:.2f} "
                f"hold={avg_hold:.1f}d "
                f"({elapsed:.0f}s)",
                flush=True,
            )

    print("\n=== SORTED BY P/L RATIO ===")
    for r in sorted(results, key=lambda x: x["pl"], reverse=True):
        alpha_label = r["alpha_f"] or "base"
        print(
            f"  sell={r['sell_w']} alpha={alpha_label}: "
            f"n={r['trades']:4d} wr={r['wr']:.1%} "
            f"mean_profit={r['mean_profit']:+.2%} mean_loss={r['mean_loss']:+.2%} "
            f"P/L={r['pl']:.2f} hold={r['avg_hold']:.1f}d"
        )


if __name__ == "__main__":
    run_grid_search()
