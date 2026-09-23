// Shared causal features for historical training and live inference.
// Offline prices are Binance; live uses Coinbase history and Chainlink movement.
// FEATURE_VERSION must change whenever a feature's meaning changes.

export const FEATURES = [
  // --- intra-window: what the market can see and the old model could not
  'moveSoFar', 'moveSoFarZ', 'elapsedFrac', 'moveTimesLeft',
  // --- prior closed 5m candles
  'ret1', 'ret3', 'ret6', 'ret12', 'ret24', 'ret48',
  'vol6', 'vol24', 'vol96',
  'volRatio',
  'bodyRatio', 'upperWick', 'lowerWick',
  'streak', 'distHigh96', 'distLow96',
  'rsi14', 'accel',
  'hourSin', 'hourCos', 'dow',
];
export const DIM = FEATURES.length;
export const WARM = 97;
export const FEATURE_VERSION = 3;

const mean = (a) => a.reduce((s, x) => s + x, 0) / (a.length || 1);
const std = (a) => { const m = mean(a); return Math.sqrt(mean(a.map((x) => (x - m) ** 2))); };

// k    : ascending closed 5m candles; k[i-1] is the last one BEFORE this window
// i    : index such that k[i] is the window being predicted (only its open is used)
// open5: open price of the current window
// now  : BTC price at decision time
// elapsedSec: seconds since the window opened (0..300)
export function extract(k, i, open5, now, elapsedSec) {
  const p = k[i - 1];
  const t = k[i].t;
  const rets = [];
  for (let j = i - 96; j < i; j++) rets.push(Math.log(k[j].c / k[j - 1].c));

  const r = (n) => Math.log(p.c / k[i - 1 - n].c);
  const win = (n) => rets.slice(rets.length - n);
  const hi = Math.max(...k.slice(i - 96, i).map((x) => x.h));
  const lo = Math.min(...k.slice(i - 96, i).map((x) => x.l));

  let streak = 0;
  for (let j = i - 1; j >= 0 && Math.abs(streak) < 20; j--) {
    const up = k[j].c >= k[j].o;
    if (j === i - 1) streak = up ? 1 : -1;
    else if (up === (streak > 0)) streak += up ? 1 : -1;
    else break;
  }

  // Simple-window RSI: zero gains/losses still count in the 14-bar window.
  const gains = win(14).reduce((s, x) => s + Math.max(x, 0), 0);
  const losses = win(14).reduce((s, x) => s + Math.max(-x, 0), 0);
  const rsi = gains + losses > 0 ? gains / (gains + losses) : 0.5;
  const rng = (p.h - p.l) || 1e-9;
  const d = new Date(t);

  // Intra-window. moveSoFarZ scales the move by recent volatility, so "up $20"
  // means something different in a calm hour than a violent one. moveTimesLeft
  // is the interaction that matters most: a big move with 30s left is nearly
  // settled; the same move with 240s left is barely information.
  const vol5 = std(win(24)) || 1e-6;
  const move = Math.log(now / open5);
  const elapsedFrac = Math.min(1, Math.max(0, elapsedSec / 300));

  return [
    move, move / vol5, elapsedFrac, (move / vol5) * elapsedFrac,
    r(1), r(3), r(6), r(12), r(24), r(48),
    std(win(6)), vol5, std(win(96)),
    p.v / (mean(k.slice(i - 24, i).map((x) => x.v)) || 1e-9),
    (p.c - p.o) / rng, (p.h - Math.max(p.o, p.c)) / rng, (Math.min(p.o, p.c) - p.l) / rng,
    streak / 10,
    (hi - p.c) / p.c, (p.c - lo) / p.c,
    rsi - 0.5,
    2 * r(1) - r(2),
    Math.sin(2 * Math.PI * d.getUTCHours() / 24), Math.cos(2 * Math.PI * d.getUTCHours() / 24),
    d.getUTCDay() / 6 - 0.5,
  ];
}
