# Stock Indicator — Engineering Pipeline Plan

## Current State (2026-04-22)

**Strategy: LOCKED.** Buy3 adaptive TP/SL, Calmar 1.03, 37yr zero losing years.

**Daily pipeline:**
```
cron (05:00 UTC+8 Tue-Sat)
  → update_all_data_from_yf       # download latest prices
  → find_history_signal            # detect entry/exit signals
  → compute_adaptive_tp_sl         # compute TP/SL from rolling stats
  → show_positions                 # display current holdings
  → tee to logs/{DATE}.log         # flat text log
```

**Log is the interface.** All downstream systems (web frontend, broker API) read from logs — they never call strategy code directly.

---

## Phase 1: Foundation — Futu integration + structured log + visualization

Three parallel workstreams, no dependencies between them.

### 1a. Futu API — real position data

**Goal:** Replace manual positions.json with real brokerage data.

**Futu OpenD capabilities (confirmed):**
- `position_list_query()` → holdings, cost_price, unrealized_pl, market_val (DataFrame)
- `request_history_kline()` → OHLCV daily data, 20yr history, 100 unique stocks/month (Basic tier)
- `place_order()` → market/limit/stop/trailing stop, paper trading via `trd_env=SIMULATE`
- Requires Futu OpenD gateway running locally on Mac

**Step 1: Read-only position sync**
```
futu_sync.py
  → connect to OpenD (127.0.0.1:11111)
  → position_list_query(trd_market=US)
  → write to data/positions_live.json
  → compare with data/positions.json (signal-based)
  → log discrepancies
```

Output: `data/positions_live.json`
```json
{
  "account": "real",
  "as_of": "2026-04-22T09:30:00",
  "positions": [
    {
      "symbol": "NVDA",
      "qty": 100,
      "cost_price": 166.97,
      "market_price": 199.98,
      "unrealized_pl": 3301.00,
      "unrealized_pl_pct": 0.1977
    }
  ],
  "total_market_value": 120000.00,
  "cash": 40000.00
}
```

**Step 2: Check if historical price API is accessible**
- Try `request_history_kline("US.AAPL", ...)` on your account
- If works: potential yfinance replacement (more reliable, no rate limit drama)
- If quota too small: keep yfinance for bulk download, Futu for live prices only

**Step 3: Closed-loop position tracking**
- On exit signal: read actual exit price from Futu instead of manual entry
- Auto-update `adaptive_state.json` with real raw_pct
- Cron becomes fully hands-free for position bookkeeping

**Prerequisites:**
- [ ] Install Futu OpenD on Mac
- [ ] `pip install futu-api`
- [ ] Confirm account tier and API questionnaire completed

### 1b. Structured JSON log

**Goal:** Machine-readable log for web dashboard and future automation.

Add TP/SL % to `find_history_signal` output (delayed rolling makes this possible at signal time). Write parallel JSON log: `logs/{DATE}.json`

```json
{
  "date": "2026-04-21",
  "strategy": "buy3",
  "adaptive_tp_pct": 0.0772,
  "adaptive_sl_pct": 0.03,
  "rolling_mp": 0.0506,
  "rolling_ml": -0.0584,
  "rolling_samples": 20,
  "entry_signals": ["NVDA", "TSLA"],
  "exit_signals": ["BA", "MSTR"],
  "hold_blocked": [],
  "positions_after": [
    {"symbol": "NVDA", "entry_date": "2026-04-21"},
    {"symbol": "TSLA", "entry_date": "2026-04-21"}
  ],
  "positions_count": 2,
  "max_positions": 6
}
```

Text log stays for human reading. JSON log is what web/Futu consumes.

### 1c. Web dashboard — signal + stats visualization

**Goal:** Browser dashboard showing today's signals, portfolio state, and rolling stats.

**Architecture:**
```
logs/{DATE}.json  ────────→  FastAPI server  ──→  browser
data/positions.json ───────┘
data/positions_live.json ──┘  (from Futu)
data/adaptive_state.json ──┘
```

No database. Server reads JSON files from disk.

**Views:**
1. **Today** — entry/exit signals, TP/SL %, action list
2. **Portfolio** — current positions (signal-based + Futu live), bars held, P/L
3. **Rolling stats** — MP, ML, sample count, TP/SL trend over time
4. **History** — past signals and outcomes (from `logs/*.json`)

**Tech:**
- Python FastAPI — same venv
- Minimal frontend: plain HTML + vanilla JS (or htmx)
- Local/LAN only initially

---

## Phase 2: Order automation (after Phase 1 is stable)

**Goal:** Semi-automated order placement via Futu OpenD.

### Order flow
1. Cron runs → `logs/{DATE}.json` written with entry signals + TP/SL %
2. `futu_order.py` reads JSON:
   - BUY: limit order near previous close → on fill, place TP limit sell + SL stop order
   - SELL (exit signal): market sell at open
3. Fill prices auto-update `positions.json` + `adaptive_state.json`

### Safety
- Paper trading (`trd_env=SIMULATE`) until confidence established
- Max position count enforced in code AND broker logic (double check)
- Order size = account_equity / max_positions (equal weight)
- Kill switch: `data/trading_paused.json` → skip all orders
- All orders logged to `logs/{DATE}_orders.json`
- Human confirmation mode: print order plan, wait for approval before sending

---

## Execution order

| Step | What | Effort | Blocks |
|------|------|--------|--------|
| 1a-prereq | Install OpenD, pip install futu-api, verify account | Setup | Everything Futu |
| 1a-step1 | futu_sync.py — read positions | Small | 1c portfolio view |
| 1a-step2 | Test historical kline API access | Small | yfinance replacement decision |
| 1b | JSON log + TP% in signal output | Small | 1c today view |
| 1c | Web dashboard | Medium | — |
| 1a-step3 | Auto position bookkeeping | Small | Phase 2 |
| 2 | Order automation (paper first) | Medium | — |
