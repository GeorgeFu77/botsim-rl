# BotSimRL

A machine-learning experiment that predicts whether Bitcoin will finish a 5-minute window up or down, then tests whether those predictions hold up under realistic, paper-only trading on Polymarket's 5-minute BTC markets. It runs next to [BotSim](https://github.com/GeorgeFu77/botsim), which supplies the live market data.

> **Paper only.** The code never connects a wallet, signs, authenticates or sends an order.

## In plain English

1. **Predict.** A small neural network, written from scratch in plain JavaScript with no ML libraries, reads 25 features of recent price action and estimates the probability that BTC closes the window higher. It trained on about 2.8 million examples built from six years of minute-by-minute Bitcoin history (2020–2026).
2. **Trade on paper.** A live agent decides whether each prediction is worth a simulated bet after fees, spread and a one-second delay. Simulated fills walk the real order book, and every entry has tight risk limits.
3. **Keep learning.** The model updates after every resolved market, but a new version only replaces the old one after it proves itself on at least 288 fresh markets over 24+ hours.
4. **Explore at scale.** 128 virtual "experimenters" try different choices on the same live feed and teach one shared reward model (a contextual bandit), while a central paper account trades with what it learns.

## Results so far

| Metric (historical test set: 414,200 predictions across 103,550 markets) | Result |
|---|---|
| Accuracy predicting Up/Down | **74.95%** |
| Accuracy 60 seconds into a window | 64.71% |
| Accuracy 240 seconds into a window | 85.06% |
| Brier score (lower is better) | 0.16568 |
| AUC | 0.8356 |

Predicting direction well is **not** the same as making money: the market price already reflects most of what the model knows, and fees and the spread take the rest. An audit of the first live agent also found real bugs, including a price reader that sometimes returned the wrong exchange's price and fills that were assumed to be better than the order book allowed. This version is built around fixing those. Whether any of it beats the market after costs is still an open question, and the notebook below tracks exactly what has and hasn't been shown. 29 automated tests cover the learning, restart, risk and settlement logic.

## Run it

No dependencies, just Node.js. The training data (monthly price archives) is downloaded, not committed.

```bash
npm run download     # download the price history
npm run episodes     # build training examples
npm run train        # train the direction model
npm run live         # start the live paper agent (needs BotSim's collectors running)
npm run status       # central paper account and learner health
npm test             # regression tests
```

## How I built it

I built this with AI coding agents as my coding partners. I asked the questions (can a model beat these markets, and how would I know if it was lying to me?), approved each experiment and fix, and ran it day to day. The AI agents wrote most of the code and the lab notebook below.

---

## Lab notebook


BTC five-minute probability prediction and an educational **paper-only learning lab**. A supervised MLP predicts direction; a shared contextual bandit learns one-step simulated action rewards. This is not full multi-step RL. The code never authenticates, signs, connects a wallet, or sends an order.

### September 20: 128 experimenters and one central account

The current experiment lives in `results/paper-lab-v1/`. `npm run status` and `npm run score` show the **big brain's own $80-start paper account**, not a leaderboard or pooled worker profits. The previous v4 account is preserved at its last balance; `live.paperEntries: false` stops further v4 entries while its forecaster continues learning and producing observations.

128 lightweight virtual workers share the existing feed, observations and reward learner in one Node process. Six exploration profiles are repeated with deterministic per-worker randomization. Workers use independently hypothetical account fills, not competing orders; summing their fills would overstate available real liquidity. Up, Down and abstain are the actions, with at most one filled entry per market/account. Every fill needs a fresh book after the simulated transit delay, respects available ask depth, includes fees, and enforces a slippage bound. No market close from the future is available to decisions.

The reward learner predicts net reward **per contract**, not a guaranteed probability, wallet return, or long-horizon optimal policy. Filled worker outcomes train it once per resolved market. Duplicated action/context/reward examples are collapsed (including floating-point near-duplicates); whole markets get equal learning weight. Replay is bounded to 512 markets with small clipped updates. The direction forecaster continues to learn from observed outcomes even if no worker enters. More workers explore alternatives; they do not multiply independent market evidence.

Workers can start another $80 episode only when their cash falls below the paper minimum and they have no outstanding positions. Deposited credits, cumulative losses, fees, wins and reset counts stay recorded. **The central account is never automatically reset or topped up.** It retains the earlier $0.40/0.5% entry, $0.80/1% exposure, $1.60 daily gross-loss and $8 persistent drawdown limits. Worker exploration does not stop at the central account's daily limit. Cash and realized P/L are reported separately from equity at entry cost; that equity is not a liquidation-price valuation.

Two additional hidden evaluation ledgers compare a frozen candidate against the frozen incumbent under identical execution rules, with fixed capped paper entries and no daily-loss cutoff. They are experimental comparisons, not the displayed central account. A candidate needs at least 288 future resolved markets and 24 hours, at least 20 completed entries on both sides, positive candidate P/L, at least $1 greater P/L than the incumbent, and no more than $0.40 extra realized drawdown. Only then can it replace the active decision policy. Otherwise the incumbent stays. Each new evaluation cycle resets only those evaluation ledgers, never the central account; the journal retains previous evaluations. These heuristic gates do not establish statistical significance or real-money suitability.

Warm-start replay uses **recorded fills**, not fabricated fills from BTC candles. Its 2,692 markets were split chronologically: 1,884 training, 403 validation, 405 test. Six training passes select an epoch on validation only. The selected reward model's held-out MSE was approximately 0.20199 versus baseline 0.20260. This measures predictions for the historically selected actions, not profitability of alternative policies. No historical replay dollars are credited to any new account. The larger candle dataset remains for direction forecasting, not simulated execution claims.

Persistence uses an append-first grouped journal plus atomic checkpoints of every account, policy, replay buffer and journal byte offset. Restart reads only the checkpoint and later journal tail, preserves pending/open positions and deduplicates already-resolved markets. Configuration or seed mismatches fail loudly instead of resetting an account. The existing `com.george.botsim.rl` service runs both the forecaster and lab; shared operational logs remain in `results/live-v4/live.log`.

- `state.json`: central-only account and learning-health view.
- `checkpoint.json`: private full experiment state, including workers and evaluator ledgers.
- `events.jsonl`: append-only decisions, grouped fills/cancellations, resolutions, evaluation cycles and worker resets.
- `bootstrap.json`: seed policy and historical split/evaluation report; replay refuses to overwrite a started experiment.
- `benchmark.json`: synthetic CPU-capacity measurements, excluding disk/network; not trading results.

The 32/64/128-worker benchmark selected 128 on this Mac. It measures account-cycle throughput in a batched engine, not 128 operating-system processes or GPU utilization. Tests cover isolated ledgers, duplicate-worker evidence, late-book fills, future-label rejection, worker-only resets, central loss pauses with ongoing learning, checkpoint-tail recovery, torn journal repair and three simulated days of deterministic restart equivalence. The simplification pass reduced the action-selection helper from 6 to 5 lines and reused one set of policy scores/IDs per observation; the same math and random draws are retained. Existing fill and journal helpers are shared rather than duplicated.

September 20 verification: **29 tests passed**. The optimized synthetic 128-worker cycle took about **0.37 ms**, excluding disk/network. A forced production-process crash restored the central position and worker positions under launchd. The first subsequent live settlement at 17:10 UTC generated a new reward-learning update from 115 filled worker accounts, collapsed to three distinct action/context/reward examples in one market. The central ledger and all 130 other internal ledgers reconciled independently; the central account still had exactly $80 in initial deposits and no resets. The first single-trade paper result is not evidence of profitability. Check `npm run status` for current numbers.

### September 18: continuous learning (v4)

The old live loop only predicted with frozen weights. V4 updates a separate learner after each observed market resolves, including markets with no paper entry. It bootstraps from the retained v3 experience once, then continues online. Each update mixes the newest resolved market with replay from a bounded 512-market buffer, weights markets equally, uses a small learning rate, and clips gradients. The input scaler stays frozen. The old Platt calibrator is folded into the learner's output layer so updates optimize the reported probability.

Learning and deployment are separate. A frozen candidate is evaluated using predictions logged **before** outcomes were known; later training cannot rewrite its scores. After at least 288 resolved markets spanning 24 hours, it replaces the active model only if its mean market-level Brier score improves by at least 0.002, its log loss is no worse, and its Brier score is no worse than the market baseline. Otherwise the active model stays put. A fresh candidate then starts a new evaluation cycle while the learner keeps training. These are experimental gates, not statistical proof of profitability or a guaranteed win rate.

`learning.json` atomically checkpoints weights, replay, processed market IDs and evaluation progress. Restarting replays unprocessed outcomes, not already-trained markets. Observations/outcomes are flushed before use; torn final journal records are backed up and repaired, while complete malformed records fail loudly. A missing outcome triggers bounded public read-only API requests with backoff, accepting only an exact matching, closed, resolved market with final 1/0 prices. Learning and resolution recovery run even while paper entries are paused. A stale learning timestamp is shown explicitly rather than treating an alive process as proof of learning.

The new **$80 educational paper account** is separate from v3 and from all wallets:

- Maximum entry cost: the smaller of 0.5% of available paper cash and $0.40, including simulated fees; confidence/edge can size it smaller.
- Maximum open plus pending cost: the smaller of 1% of paper equity-at-cost and $0.80.
- Daily budget: $1.60 in gross settled losses, reserving the full worst-case loss of open/pending entries. Profits do not replenish this budget. It resets at midnight UTC.
- Persistent drawdown budget: $8 below the paper high-water mark, initially a $72 floor. No overnight reset; learning continues after entries halt. Budgets below the $0.10 hypothetical minimum also stop entries.
- Fractional paper fills are educational, **not** a claim that a venue permits these order sizes. Confidence is an estimate of outcome probability, not the fraction of money to spend. The risk caps are simulation rules, not real-money protection.

The service starts at login and restarts after crashes. It needs the Mac powered on, awake, logged in, online, and BotSim's collectors working. It cannot learn through shutdown or recover feature observations never collected. Missing outcomes for recorded observations can be recovered later. No wallet connection, orders, power-setting changes, or external notification service were added.

### Running

```bash
npm run status       # central account and shared learner only
npm run score        # same central scoreboard; no worker leaderboard
npm run status:forecaster # underlying direction model and preserved v4 account
npm run score:forecaster  # direction scores, not trading returns
npm run lab:benchmark # synthetic batched 32/64/128 CPU benchmark
npm run lab:replay    # initial historical reward-model seed; refuses an existing run
npm test             # focused feed/feature/replay/fill regression checks
npm run live         # foreground; refuses a duplicate active process
```

The installed macOS service is `com.george.botsim.rl`. It starts at login and restarts after a crash. Commands on this machine:

```bash
launchctl print gui/$(id -u)/com.george.botsim.rl
launchctl bootout gui/$(id -u)/com.george.botsim.rl
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.george.botsim.rl.plist
```

Stop the service before foreground debugging. The PID file prevents a second agent from writing the same account. Graceful termination finishes the current pass; the journal restores account state after restarting.

### September 4 correction (feature version 3)

- Both the opening reference and current BTC price come from **Chainlink records selected by source timestamp**. The old `snapshot.market.openBtc` was blended spot; the old last-line reader could return `polybinance`.
- Books come directly from BotSim's existing `books.jsonl`. Both outcomes need fresh, uncrossed two-sided books; Up and Down asks are priced independently.
- A decision uses data already received, with a one-second visibility delay and bounded source skew. Paper orders then wait for a new book from after the simulated one-second transit delay.
- Fills walk available ask depth, include fees in the budget, allow partial fills, and stop before an expensive level eliminates the configured margin. These remain hypothetical fills, not proof of actual execution.
- Decisions occur near 60/120/180/240 seconds of source time, with one entry maximum per market. Every valid prediction is recorded, including skipped and already-held markets. Feed gaps can cause an offset to be missed.
- Coinbase history uses explicit bounds, caching, timeouts, and retries. Incomplete history is rejected. There are no overlapping async ticks.
- New settlements are copied into a local append-only file before BotSim's daily data clear can erase them. The feed cursor detects truncation even if the same file has already regrown.
- RSI, acceleration, replay mixture importance weights, and tied-prediction AUC are corrected. MLP scratch arrays are reused to reduce allocation churn.
- Training splits whole markets, uses separate epoch-selection and calibration blocks, and reports results by offset. A model/schema version mismatch stops startup.

All settings are in `config.js`. The original v3 used a 2% stake / 4% exposure cap; v4 uses the smaller paper limits above. The static simulated fee rate is 0.07; market-specific fee changes are not automatically discovered.

### Data and account history

The v4 experiment starts with $80 paper cash under `results/live-v4/`:

- `events.jsonl`: authoritative append-only observations, pending decisions, entries, cancellations, settlements. Includes model IDs, features, and source/receipt timestamps.
- `live-state.json`: atomically replaced account/health snapshot; reconstructed from events on startup.
- `resolutions.jsonl`: retained Chainlink outcomes copied from BotSim.
- `learning.json`: durable online learner, frozen active/candidate models, replay and evaluation progress.
- `live.log`: operational events and one status line per minute.

The $1,000-start v3 account and all older files remain preserved, including v3's historical unresolved positions; they are not rolled into or rewritten as the new $80 account. V4 recovers missing historical outcomes for learning without rewriting old account history. Source/model backups are in `results/archive/20260905T025530Z/` and `results/archive/20260919T025834Z/`.

**Settlement limitation:** ordinary labels still come from the upstream collector's Chainlink boundary prints, not independent venue confirmation. Missing labels now use the strictly validated public-API recovery path. If neither source supplies a valid outcome, the paper position stays open, appears overdue after ten minutes and blocks new exposure; no payout is invented. Forecast quality is not an after-fee trading win rate.

Tests cover three simulated days with repeated learner restarts, duplicate/future-label rejection, candidate promotion/rejection, bounded replay, paper caps including pending exposure, ten losing simulated days, journal repair, feed truncation, and exact resolution validation. A three-day simulated test is not three days of observed production uptime.

September 18 verification: 19 tests passed. An actual forced process crash was recovered by launchd with the same learner checkpoint and open paper position. That position subsequently settled and the newly resolved market produced another persisted training update (3,978 → 3,979), without a manual training command. Six missing historical outcomes were recovered, including all four that had blocked v3. No candidate has yet passed the new forward evaluation gate. The simplification review of the four core runtime files was 466 → 466 lines: no safe cuts were found; shared model restoration already replaces the former duplicate startup-loading block.

### Training

```bash
npm run download                 # monthly 5m archive
node src/data/download-1m.js     # matching 1m archive
npm run episodes                # rebuild after changing features
npm run train                   # writes model and regression metrics
node src/analyze.js
```

Version 3 trained on 2,761,341 observations after removing gapped histories. Its historical regression set has 414,200 observations in 103,550 markets: Brier **0.16568**, accuracy **74.95%**, AUC **0.8356**. Accuracy by offset: 60s **64.71%**, 120s **71.79%**, 180s **78.27%**, 240s **85.06%**. Later decisions are easier because more of the price move has happened.

These are historical prediction metrics, not evidence of profitable trading. The test period has already been inspected. Binance training labels and Coinbase/Chainlink live inputs still differ by venue. New-run scoring groups observations by market and, with at least seven days and 100 markets, resamples whole days for an uncertainty interval. A good overall Brier score neither proves nor rules out a profitable subset of trades.
