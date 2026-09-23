// BotSimRL config. Everything tunable lives here.
export const config = {
  symbol: 'BTCUSDT',
  interval: '5m',
  interval1m: '1m',        // finer grain: lets the model see INSIDE the 5-min window

  // How much history to pull. Binance publishes monthly zips from 2017-08.
  history: { from: '2020-01', to: null }, // to:null => up to last complete month

  // Chronological holdout. NEVER random — a random split leaks the future
  // backward and turns every metric into a lie.
  split: { train: 0.70, val: 0.15, test: 0.15 },

  // Replay buffer.
  replay: {
    capacity: 2_100_000,  // circular; must hold the whole train split so nothing is silently dropped
    recencyHalfLifeDays: 120, // older episodes stay, just quieter
    prioritized: true,
    alpha: 0.6,           // 0 = uniform, 1 = fully prioritized (Schaul et al.)
    beta: 0.4,            // importance-sampling correction; annealed to 1.0
    priorityCap: 4.0,     // stops PER obsessing over unlearnable noise
    uniformMix: 0.3,      // 30% of every batch drawn uniformly regardless
  },

  model: { hidden: 24, lr: 0.01, l2: 1e-5 },

  train: { epochs: 8, batch: 256, seed: 1337 },

  lab: {
    enabled: true,
    resultDir: 'results/paper-lab-v1',
    workers: 128,
    bankroll: 80,
    lr: 0.02,
    replayMarkets: 512,
    batchMarkets: 16,
    maxSlippage: 0.02,
    evaluationMarkets: 288,
    evaluationMs: 86_400_000,
    minimumEvaluationTrades: 20,
    minimumEvaluationGain: 1,
    allowedDrawdownDifference: 0.40,
    // Repeated randomized experiments, not 128 independent market outcomes.
    profiles: [
      { epsilon: 0.05, margin: 0.02, stake: 0.40 },
      { epsilon: 0.15, margin: 0.02, stake: 0.40 },
      { epsilon: 0.30, margin: 0.02, stake: 0.40 },
      { epsilon: 0.50, margin: 0.02, stake: 0.40 },
      { epsilon: 0.15, margin: 0.00, stake: 0.80 },
      { epsilon: 0.30, margin: 0.04, stake: 1.60 },
    ],
  },

  live: {
    paperEntries: false, // Preserve the old v4 account; paper-lab owns new entries.
    resultDir: 'results/live-v4',
    bootstrapDir: 'results/live-v3',
    botSimDir: '/Users/michelleai/Projects/BotSim',
    bankroll: 80, // Separate educational paper account. Never wallet funds.
    learning: { lr: 0.0003, replayMarkets: 512, batchMarkets: 16,
      evaluationMarkets: 288, evaluationMs: 86_400_000, minBrierGain: 0.002 },
    offsets: [60, 120, 180, 240],
    offsetToleranceSec: 10,
    tickMs: 1000,
    feedDelayMs: 1000,
    fillLatencyMs: 1000,
    maxFeedAgeMs: 10_000,
    maxSourceSkewMs: 5000,
    maxOpenAgeMs: 10_000,
    feeRate: 0.07,
    margin: 0.02,
    kellyFraction: 0.25,
    maxStake: 0.005,
    maxStakeDollars: 0.40,
    maxExposure: 0.01,
    maxExposureDollars: 0.80,
    minPaperSpend: 0.10, // Fractional hypothetical fills, not venue minimums.
    dailyLossLimit: 1.60, // Gross settled losses, plus worst-case open/pending loss.
    maxDrawdown: 8,      // Persistent high-water drawdown; never resets overnight.
    priceMin: 0.05,
    priceMax: 0.95,
  },
};
