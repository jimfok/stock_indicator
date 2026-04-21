"""Grid search over buy/sell EMA window sizes for above < 0.973 universe.

Searches for Buy3 complement strategies in the below-0.973 above_price_volume_ratio
space. Uses Top200 Pick5, no confirmation, no SL/TP.

Usage:
    python -m scripts.grid_search_window_below973
"""

import sys
import time
from pathlib import Path

import pandas

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stock_indicator import strategy  # noqa: E402

DATA_DIRECTORY = PROJECT_ROOT / "data"
STOCK_DATA_DIRECTORY = DATA_DIRECTORY / "stock_data_2014"


def run_grid_search() -> None:
    data_directory = (
        STOCK_DATA_DIRECTORY if STOCK_DATA_DIRECTORY.exists() else DATA_DIRECTORY
    )
    start_date = pandas.Timestamp("2014-01-01")

    results = []

    for buy_w in range(3, 11):
        for sell_w in range(3, 11):
            # above 0.0-0.973: complement to Buy3 (which owns 0.973-1.0)
            buy_name = f"ema_sma_cross_testing_{buy_w}_-99_99_-99.0,99.0_0.0,0.973"
            sell_name = f"ema_sma_cross_testing_{sell_w}_-99_99_-99.0,99.0_0.0,1.00"

            t0 = time.time()
            try:
                artifacts = strategy._generate_strategy_evaluation_artifacts(
                    data_directory,
                    buy_name,
                    sell_name,
                    minimum_average_dollar_volume=None,
                    minimum_average_dollar_volume_ratio=0.0005,  # 0.05%
                    top_dollar_volume_rank=200,
                    maximum_symbols_per_group=5,
                    start_date=start_date,
                    maximum_position_count=50,
                    use_confirmation_angle=False,
                    minimum_holding_bars=5,
                )
            except Exception as exc:
                print(f"buy={buy_w} sell={sell_w}: ERROR {exc}")
                continue

            all_trades = artifacts.trades
            elapsed = time.time() - t0

            if not all_trades:
                print(f"buy={buy_w} sell={sell_w}: 0 trades ({elapsed:.0f}s)")
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
            avg_hold = sum(t.holding_period for t in all_trades) / n

            results.append({
                "buy_w": buy_w,
                "sell_w": sell_w,
                "trades": n,
                "wr": wr,
                "mean_profit": mean_profit,
                "mean_loss": mean_loss,
                "profit_loss_ratio": abs(mean_profit / mean_loss) if mean_loss else 0,
                "avg_hold": avg_hold,
            })

            print(
                f"buy={buy_w} sell={sell_w}: "
                f"n={n:4d} wr={wr:.1%} "
                f"mean_profit={mean_profit:+.2%} mean_loss={mean_loss:+.2%} "
                f"P/L={abs(mean_profit / mean_loss) if mean_loss else 0:.2f} "
                f"hold={avg_hold:.1f}d "
                f"({elapsed:.0f}s)",
                flush=True,
            )

    print("\n=== SORTED BY P/L RATIO ===")
    for r in sorted(results, key=lambda x: x["profit_loss_ratio"], reverse=True):
        print(
            f"  buy={r['buy_w']} sell={r['sell_w']}: "
            f"n={r['trades']:4d} wr={r['wr']:.1%} "
            f"mean_profit={r['mean_profit']:+.2%} mean_loss={r['mean_loss']:+.2%} "
            f"P/L={r['profit_loss_ratio']:.2f} hold={r['avg_hold']:.1f}d"
        )

    print("\n=== SORTED BY WIN RATE ===")
    for r in sorted(results, key=lambda x: x["wr"], reverse=True):
        print(
            f"  buy={r['buy_w']} sell={r['sell_w']}: "
            f"n={r['trades']:4d} wr={r['wr']:.1%} "
            f"mean_profit={r['mean_profit']:+.2%} mean_loss={r['mean_loss']:+.2%} "
            f"P/L={r['profit_loss_ratio']:.2f} hold={r['avg_hold']:.1f}d"
        )


if __name__ == "__main__":
    run_grid_search()
