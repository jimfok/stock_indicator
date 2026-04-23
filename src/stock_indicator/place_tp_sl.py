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

import pandas

LOGGER = logging.getLogger(__name__)

DATA_DIRECTORY = Path(__file__).resolve().parent.parent.parent / "data"
LOGS_DIRECTORY = Path(__file__).resolve().parent.parent.parent / "logs"

# Match dashboard.py settings
TRADING_ENV = "REAL"
MIN_HOLD_BARS = 5


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

    if filled_buys:
        LOGGER.info("Found %d filled BUY orders", len(filled_buys))

    # Check existing orders to avoid duplicates (also used for SL)
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

        # SL deferred — will be placed when min_hold is met (see below).
        LOGGER.info("  SL: deferred (min_hold=%d)", MIN_HOLD_BARS)

    # ------------------------------------------------------------------
    # Place SL for positions that have passed min_hold
    # ------------------------------------------------------------------
    LOGGER.info("--- Checking SL for positions past min_hold ---")
    positions_path = DATA_DIRECTORY / "positions.json"
    positions: dict = {}
    if positions_path.exists():
        try:
            positions = json.loads(positions_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    buy3_positions = positions.get("buy3", [])
    if not buy3_positions:
        LOGGER.info("No positions in positions.json")
        trd_ctx.close()
        LOGGER.info("Done")
        return

    # Refresh existing sell orders (may have changed after TP placement above)
    ret_existing2, existing_orders2 = trd_ctx.order_list_query(trd_env=trd_env)
    existing_stop_codes = set()
    existing_limit_sell_codes = set()
    if ret_existing2 == 0 and len(existing_orders2) > 0:
        for _, row in existing_orders2.iterrows():
            if (
                str(row.get("trd_side", "")) == "SELL"
                and str(row.get("order_status", "")) not in (
                    "CANCELLED_ALL", "FAILED", "DELETED",
                )
            ):
                code_val = str(row.get("code", ""))
                order_type_val = str(row.get("order_type", ""))
                if order_type_val == "STOP":
                    existing_stop_codes.add(code_val)
                elif order_type_val == "NORMAL":
                    existing_limit_sell_codes.add(code_val)

    # Get actual positions from Futu (for cost_price)
    ret_pos, pos_data = trd_ctx.position_list_query(trd_env=trd_env)
    futu_positions: dict[str, dict] = {}
    if ret_pos == 0 and len(pos_data) > 0:
        for _, row in pos_data.iterrows():
            code_val = str(row.get("code", ""))
            futu_positions[code_val] = {
                "qty": int(row.get("qty", 0)),
                "cost_price": float(row.get("cost_price", 0)),
            }

    for pos in buy3_positions:
        symbol = pos.get("symbol", "")
        entry_date_str = pos.get("entry_date", "")
        code = f"US.{symbol}"

        if not entry_date_str:
            LOGGER.info("  %s: no entry_date, skipping SL", symbol)
            continue

        # Count trading bars since entry
        try:
            entry_ts = pandas.Timestamp(entry_date_str)
            today_ts = pandas.Timestamp(today_str)
            trading_days = pandas.bdate_range(entry_ts, today_ts)
            bars_held = max(0, len(trading_days) - 1)
        except Exception:
            bars_held = 0

        if bars_held < MIN_HOLD_BARS:
            LOGGER.info(
                "  %s: bars_held=%d < min_hold=%d, SL deferred",
                symbol, bars_held, MIN_HOLD_BARS,
            )
            continue

        # Already has a stop order?
        if code in existing_stop_codes:
            LOGGER.info("  %s: SL stop order already exists", symbol)
            continue

        # Get entry price from Futu position
        futu_pos = futu_positions.get(code)
        if not futu_pos or futu_pos["qty"] <= 0:
            LOGGER.info("  %s: no Futu position found, skipping SL", symbol)
            continue

        entry_price = futu_pos["cost_price"]
        qty = futu_pos["qty"]
        sl_price = round(entry_price * (1 - sl_pct), 2)

        LOGGER.info(
            "  %s: bars_held=%d >= min_hold=%d → SL=$%.2f (%.2f%%) [GTC stop]",
            symbol, bars_held, MIN_HOLD_BARS,
            sl_price, sl_pct * 100,
        )

        if dry_run:
            LOGGER.info("    [DRY RUN] skipping SL placement")
            continue

        ret_sl, data_sl = trd_ctx.place_order(
            price=sl_price,
            qty=qty,
            code=code,
            trd_side=TrdSide.SELL,
            order_type=OrderType.STOP,
            trd_env=trd_env,
            aux_price=sl_price,
            time_in_force=TimeInForce.GTC,
        )
        sl_result = {
            "symbol": symbol,
            "side": "SL_SELL",
            "qty": qty,
            "entry_price": entry_price,
            "price": sl_price,
            "sl_pct": round(sl_pct * 100, 2),
            "bars_held": bars_held,
            "status": "sent" if ret_sl == 0 else "failed",
            "order_id": (
                str(data_sl.iloc[0].get("order_id", ""))
                if ret_sl == 0 else None
            ),
            "error": str(data_sl) if ret_sl != 0 else None,
            "env": TRADING_ENV,
            "timestamp": datetime.now().isoformat(),
        }
        LOGGER.info("    SL: %s", sl_result["status"])
        _log_order(sl_result)

    trd_ctx.close()
    LOGGER.info("Done")


if __name__ == "__main__":
    main()
