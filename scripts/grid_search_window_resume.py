"""Resume grid_search_window run by skipping (buy, sell) combos already in the log.

Reads logs/grid_search_results_v2.txt, parses completed (buy_w, sell_w) pairs,
runs only the remaining combinations from grid_search_window, and appends each
result line to the same log file.
"""

import re
import sys
import time
from pathlib import Path

import pandas

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stock_indicator import strategy  # noqa: E402

DATA_DIRECTORY = PROJECT_ROOT / "data"
STOCK_DATA_DIRECTORY = DATA_DIRECTORY / "stock_data"
LOG_PATH = PROJECT_ROOT / "logs" / "grid_search_results_v2.txt"

LINE_PATTERN = re.compile(r"^buy=(\d+) sell=(\d+):")


def load_done_combos(log_path: Path) -> set[tuple[int, int]]:
    done: set[tuple[int, int]] = set()
    if not log_path.exists():
        return done
    for raw_line in log_path.read_text().splitlines():
        match = LINE_PATTERN.match(raw_line)
        if match:
            done.add((int(match.group(1)), int(match.group(2))))
    return done


def run_resume() -> None:
    data_directory = (
        STOCK_DATA_DIRECTORY if STOCK_DATA_DIRECTORY.exists() else DATA_DIRECTORY
    )
    start_date = pandas.Timestamp("2014-01-01")
    done = load_done_combos(LOG_PATH)
    print(f"# resume: {len(done)} combos already done", flush=True)

    log_handle = LOG_PATH.open("a")

    for buy_w in range(3, 11):
        for sell_w in range(3, 11):
            if (buy_w, sell_w) in done:
                continue

            buy_name = f"ema_sma_cross_testing_{buy_w}_-99_99_-99.0,99.0_0.78,1.00"
            sell_name = f"ema_sma_cross_testing_{sell_w}_-0.01_65_-10.0,10.0_0.78,1.00"

            t0 = time.time()
            try:
                artifacts = strategy._generate_strategy_evaluation_artifacts(
                    data_directory,
                    buy_name,
                    sell_name,
                    minimum_average_dollar_volume=None,
                    minimum_average_dollar_volume_ratio=0.0005,
                    top_dollar_volume_rank=50,
                    maximum_symbols_per_group=2,
                    start_date=start_date,
                    maximum_position_count=3,
                    use_confirmation_angle=True,
                    confirmation_entry_mode="market",
                    minimum_holding_bars=5,
                )
            except Exception as exc:
                line = f"buy={buy_w} sell={sell_w}: ERROR {exc}"
                print(line, flush=True)
                log_handle.write(line + "\n")
                log_handle.flush()
                continue

            all_trades = artifacts.trades
            elapsed = time.time() - t0

            if not all_trades:
                line = f"buy={buy_w} sell={sell_w}: 0 trades ({elapsed:.0f}s)"
                print(line, flush=True)
                log_handle.write(line + "\n")
                log_handle.flush()
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
            pl = abs(mean_profit / mean_loss) if mean_loss else 0

            line = (
                f"buy={buy_w} sell={sell_w}: "
                f"n={n:4d} wr={wr:.1%} "
                f"mean_profit={mean_profit:+.2%} mean_loss={mean_loss:+.2%} "
                f"P/L={pl:.2f} hold={avg_hold:.1f}d ({elapsed:.0f}s)"
            )
            print(line, flush=True)
            log_handle.write(line + "\n")
            log_handle.flush()

    log_handle.close()


if __name__ == "__main__":
    run_resume()
