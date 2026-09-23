// Calibration — the only metric that matters for a betting system.
//
// Accuracy asks "was the direction right?". Calibration asks the harder question:
// "when it says 70%, does that happen 70% of the time?" A bet sized off an
// uncalibrated probability is sized wrong, every time, no matter how good the
// accuracy looks.

export function evaluate(model, X, Y, rows, cal = null) {
  const scratch = new Float64Array(model.d);
  const ps = new Float64Array(rows.length);
  let brier = 0, ll = 0, correct = 0, ups = 0;

  rows.forEach((i, k) => {
    let p = model.predict(X, i, scratch);
    if (cal) p = cal.apply(p);
    const y = Y[i];
    ps[k] = p;
    brier += (p - y) ** 2;
    ll += -(y * Math.log(p + 1e-12) + (1 - y) * Math.log(1 - p + 1e-12));
    if ((p >= 0.5 ? 1 : 0) === y) correct++;
    ups += y;
  });

  const n = rows.length;
  const base = ups / n;                                   // always-Up accuracy
  return {
    n,
    brier: brier / n,
    logloss: ll / n,
    acc: correct / n,
    baseRate: base,
    baseAcc: Math.max(base, 1 - base),                    // the dumb benchmark to beat
    auc: auc(ps, rows.map((i) => Y[i])),
    reliability: reliability(ps, rows.map((i) => Y[i])),
    spread: (() => { let lo = Infinity, hi = -Infinity; for (const v of ps) { if (v < lo) lo = v; if (v > hi) hi = v; } return [lo, hi]; })(),
  };
}

// Bucket predictions and compare predicted vs actual frequency.
function reliability(ps, ys, bins = 10) {
  const b = Array.from({ length: bins }, () => ({ n: 0, sp: 0, sy: 0 }));
  ps.forEach((p, k) => {
    const i = Math.min(bins - 1, Math.max(0, Math.floor(p * bins)));
    b[i].n++; b[i].sp += p; b[i].sy += ys[k];
  });
  return b.map((x, i) => ({
    bin: `${(i / bins).toFixed(1)}-${((i + 1) / bins).toFixed(1)}`,
    n: x.n,
    predicted: x.n ? x.sp / x.n : null,
    actual: x.n ? x.sy / x.n : null,
  }));
}

// Rank quality, independent of calibration. 0.5 = no signal.
function auc(ps, ys) {
  const pairs = Array.from(ps, (p, i) => [p, ys[i]]).sort((a, b) => a[0] - b[0]);
  let rankSum = 0, pos = 0;
  for (let i = 0; i < pairs.length;) {
    let j = i + 1;
    while (j < pairs.length && pairs[j][0] === pairs[i][0]) j++;
    const rank = (i + 1 + j) / 2;
    for (let k = i; k < j; k++) if (pairs[k][1] === 1) { rankSum += rank; pos++; }
    i = j;
  }
  const neg = pairs.length - pos;
  if (!pos || !neg) return 0.5;
  return (rankSum - (pos * (pos + 1)) / 2) / (pos * neg);
}

// Expected value per $1 staked if you bet into a market priced at `q`, given
// your probability `p`. This is the number that decides whether ANY of this is
// worth running — and it needs a real edge over the market price, not over 50%.
export function kelly(p, q) {
  const edge = p - q;
  if (edge <= 0) return { f: 0, ev: 0 };
  return { f: Math.max(0, Math.min(1, edge / (1 - q))), ev: edge };
}

export function report(name, m) {
  const flag = (v, good) => (v ? ' ✅' : ' ❌');
  console.log(`\n  ── ${name} ──  n=${m.n.toLocaleString()}`);
  console.log(`     Brier      ${m.brier.toFixed(5)}   (coin flip 0.25)${flag(m.brier < 0.25)}`);
  console.log(`     LogLoss    ${m.logloss.toFixed(5)}   (coin flip 0.69315)${flag(m.logloss < 0.69315)}`);
  console.log(`     Accuracy   ${(m.acc * 100).toFixed(2)}%   (always-Up ${(m.baseAcc * 100).toFixed(2)}%)${flag(m.acc > m.baseAcc)}`);
  console.log(`     AUC        ${m.auc.toFixed(4)}   (no signal 0.5)${flag(m.auc > 0.5)}`);
  console.log(`     p range    ${m.spread[0].toFixed(3)} .. ${m.spread[1].toFixed(3)}`);
}

export function reliabilityTable(m) {
  console.log(`\n     bin        n        predicted   actual    gap`);
  for (const b of m.reliability) {
    if (!b.n) continue;
    const gap = b.actual - b.predicted;
    console.log(`     ${b.bin}  ${String(b.n).padStart(7)}     ${b.predicted.toFixed(3)}      ${b.actual.toFixed(3)}   ${gap >= 0 ? '+' : ''}${gap.toFixed(3)}`);
  }
}
