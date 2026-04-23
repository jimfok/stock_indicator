"""Place TP/SL orders for today's filled entry orders.

Run after market open (e.g. 5-10 minutes after 09:30 ET) to:
1. Query today's filled BUY orders from Futu
2. Read adaptive TP/SL percentages from latest log
3. Place TP limit sell + SL stop order for each fill

Usage:
    venv/bin/python -m stock_indicator.place_tp_sl [--dry-run]
"""

from __future__ import annotations

import json
import logging
import math
import re
import sys
from datetime import date
from pathlib import Path

LOGGER = logging.getLogger(__name__)

DATA_DIRECTORY = Path(__file__).resolve().parent.parent.parent / "data"
LOGS_DIRECTORY = Path(__file__).resolve().parent.parent.parent / "logs"

# Match dashboard.py settings
TRADING_ENV = "REAL"


def _get_tp_sl_from_log() -> tuple[float | None, float | None]:
    """Read TP/SL percentages from the latest daily log."""
    log_files = sorted(LOGS_DIRECTORY.glob("*.log"), reverse=True)
    for log_path in log_files:
        try:
            date.fromisoformat(log_path.stem)
        except ValueError:
            continue
        text = log_path.read_text(encoding="utf-8")
        tp_m = re.search(r"TP:\s*([\d.]+)%", text)
        sl_m = re.search(r"SL:\s*([\d.]+)%", text)
        tp = float(tp_m.group(1)) / 100 if tp_m else None
        sl = float(sl_m.group(1)) / 100 if sl_m else None
        if tp is not None:
            return tp, sl
    return None, None


def _log_order(order_data: dict) -> None:
    """Append order to today's order log."""
    today = date.today().isoformat()
    log_path = LOGS_DIRECTORY / f"{today}_orders.json"
    orders = []
    if log_path.exists():
        try:
            orders = json.loads(log_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    orders.append(order_data)
    log_path.write_text(json.dumps(orders, indent=2), encoding="utf-8")


def main() -> None:
    from datetime import datetime

    from futu import (
        OpenSecTradeContext,
        OrderType,
        SecurityFirm,
        TimeInForce,
        TrdEnv,
        TrdMarket,
        TrdSide,
    )

    dry_run = "--dry-run" in sys.argv
    trd_env = TrdEnv.REAL if TRADING_ENV == "REAL" else TrdEnv.SIMULATE

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
    )

    # Get TP/SL from log
    tp_pct, sl_pct = _get_tp_sl_from_log()
    if tp_pct is None:
        LOGGER.error("No TP/SL found in logs")
        return
    LOGGER.info("Adaptive TP: %.2f%%, SL: %.2f%%", tp_pct * 100, sl_pct * 100)

    # Connect to Futu
    trd_ctx = OpenSecTradeContext(
        host="127.0.0.1",
        port=11111,
        filter_trdmarket=TrdMarket.US,
        security_firm=SecurityFirm.FUTUSECURITIES,
    )

    # Query today's filled orders
    today_str = date.today().isoformat()
    ret, order_data = trd_ctx.history_order_list_query(
        trd_env=trd_env,
        start=today_str,
        end=today_str,
    )
    if ret != 0:
        LOGGER.error("Failed to query orders: %s", order_data)
        trd_ctx.close()
        return

    # Filter for filled BUY orders
    filled_buys = []
    if len(order_data) > 0:
        for _, row in order_data.iterrows():
            if (
                str(row.get("trd_side", "")) == "BUY"
                and str(row.get("order_status", "")) in (
                    "FILLED_ALL", "FILLED_PART",
                )
            ):
                filled_buys.append({
                    "code": str(row.get("code", "")),
                    "symbol": str(row.get("code", "")).replace("US.", ""),
                    "qty": int(row.get("dealt_qty", 0)),
                    "avg_price": float(row.get("dealt_avg_price", 0)),
                    "order_id": str(row.get("order_id", "")),
                })

    if not filled_buys:
        LOGGER.info("No filled BUY orders today")
        trd_ctx.close()
        return

    LOGGER.info("Found %d filled BUY orders", len(filled_buys))

    # Check existing orders to avoid duplicates
    ret_existing, existing_orders = trd_ctx.order_list_query(trd_env=trd_env)
    existing_sell_codes = set()
    if ret_existing == 0 and len(existing_orders) > 0:
        for _, row in existing_orders.iterrows():
            if (
                str(row.get("trd_side", "")) == "SELL"
                and str(row.get("order_status", "")) not in (
                    "CANCELLED_ALL", "FAILED", "DELETED",
                )
            ):
                existing_sell_codes.add(str(row.get("code", "")))

    # Place TP + SL for each fill
    for fill in filled_buys:
        code = fill["code"]
        symbol = fill["symbol"]
        qty = fill["qty"]
        entry_price = fill["avg_price"]

        if code in existing_sell_codes:
            LOGGER.info(
                "Skipping %s — sell order already exists", symbol
            )
            continue

        tp_price = round(entry_price * (1 + tp_pct), 2)

        LOGGER.info(
            "%s: entry=$%.2f qty=%d → TP=$%.2f (%.2f%%) [GTC limit sell]",
            symbol, entry_price, qty,
            tp_price, tp_pct * 100,
        )

        if dry_run:
            LOGGER.info("  [DRY RUN] skipping order placement")
            continue

        # Place TP limit sell (GTC)
        ret_tp, data_tp = trd_ctx.place_order(
            price=tp_price,
            qty=qty,
            code=code,
            trd_side=TrdSide.SELL,
            order_type=OrderType.NORMAL,
            trd_env=trd_env,
            time_in_force=TimeInForce.GTC,
        )
        tp_result = {
            "symbol": symbol,
            "side": "TP_SELL",
            "qty": qty,
            "entry_price": entry_price,
            "price": tp_price,
            "tp_pct": round(tp_pct * 100, 2),
            "status": "sent" if ret_tp == 0 else "failed",
            "order_id": (
                str(data_tp.iloc[0].get("order_id", ""))
                if ret_tp == 0 else None
            ),
            "error": str(data_tp) if ret_tp != 0 else None,
            "env": TRADING_ENV,
            "timestamp": datetime.now().isoformat(),
        }
        LOGGER.info("  TP: %s", tp_result["status"])
        _log_order(tp_result)

        # SL is NOT placed here.
        # min_hold=5 means SL should only activate after 5 bars.
        # SL will be placed by a separate process after min_hold expires.
        LOGGER.info("  SL: deferred (min_hold=5, entry=%s)", today_str)

    trd_ctx.close()
    LOGGER.info("Done")


if __name__ == "__main__":
    main()
