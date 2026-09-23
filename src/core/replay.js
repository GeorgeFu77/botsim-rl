// Experience replay buffer — circular, recency-weighted, optionally prioritized.
//
// Three things this fixes vs a naive growing buffer:
//   1. CIRCULAR, not growing. A growing buffer sampled uniformly replays its
//      OLDEST episode ~ln(N) times more than its newest. Backwards for a market.
//   2. RECENCY WEIGHT, not deletion. Old regimes go stale, so they get quieter —
//      but they stay, because rare events (crashes, spikes) only live in old data.
//   3. CAPPED priorities + a uniform floor. Prioritized replay chases high error,
//      and in a noisy market high error usually means UNLEARNABLE, not informative.
//      Uncapped PER will drill the most random moments in the tape forever.
//
// Importance-sampling weights (beta) are applied, per Schaul et al. 2015 —
// without them, prioritized sampling biases the gradient and can lose to uniform.
//
// Sampling is backed by a SUM TREE: O(log n) to draw and O(log n) to reweight,
// instead of the O(n) scan a flat array needs. At 500k episodes that is the
// difference between minutes and hours.

const DAY = 86400_000;

class SumTree {
  constructor(cap) { this.cap = cap; this.t = new Float64Array(2 * cap); }
  set(i, v) {
    let x = i + this.cap;
    const d = v - this.t[x];
    if (d === 0) return;
    for (; x >= 1; x >>= 1) this.t[x] += d;
  }
  get total() { return this.t[1]; }
  // Walk down to the leaf whose cumulative range contains `r`.
  find(r) {
    let x = 1;
    while (x < this.cap) {
      const left = x << 1;
      x = r < this.t[left] ? left : (r -= this.t[left], left + 1);
    }
    return { idx: x - this.cap, w: this.t[x] };
  }
}

export class ReplayBuffer {
  constructor({ capacity, recencyHalfLifeDays, prioritized, alpha, beta, priorityCap, uniformMix }) {
    this.cap = capacity;
    this.halfLife = recencyHalfLifeDays * DAY;
    this.prioritized = prioritized;
    this.alpha = alpha; this.beta = beta;
    this.priorityCap = priorityCap; this.uniformMix = uniformMix;

    this.x = new Int32Array(capacity); this.y = new Uint8Array(capacity);
    this.t = new Float64Array(capacity);
    this.prio = new Float64Array(capacity);
    this.rec = new Float64Array(capacity);   // cached recency factor
    this.head = 0; this.n = 0; this.maxT = 0;
    this.tree = new SumTree(capacity);
  }

  _w(i) {
    const p = this.prioritized ? Math.pow(Math.min(this.prio[i], this.priorityCap), this.alpha) : 1;
    return p * this.rec[i];
  }

  push(x, y, t) {
    const i = this.n < this.cap ? this.n : this.head;
    this.x[i] = x; this.y[i] = y; this.t[i] = t;
    this.prio[i] = this.priorityCap;         // new episodes start max-priority: seen once, then ranked honestly
    if (t > this.maxT) this.maxT = t;
    this.rec[i] = 1;
    this.tree.set(i, this._w(i));
    if (this.n < this.cap) this.n++; else this.head = (this.head + 1) % this.cap;
  }

  // Exponential decay on age. Half-life in days: an episode one half-life old is
  // drawn half as often as a fresh one — quieter, never silent. Call after a load
  // or whenever maxT jumps; it is O(n), so not per-batch.
  refreshRecency() {
    for (let i = 0; i < this.n; i++) {
      this.rec[i] = Math.pow(0.5, (this.maxT - this.t[i]) / this.halfLife);
      this.tree.set(i, this._w(i));
    }
  }

  // Returns { idx, isw } — slot indices plus importance-sampling weights that
  // undo the sampling bias. Train with each sample's gradient scaled by isw.
  sample(batch, rng = Math.random) {
    if (!this.n || !Number.isInteger(batch) || batch < 1) throw new Error('Cannot sample empty replay or invalid batch');
    const idx = new Int32Array(batch);
    const isw = new Float64Array(batch);
    const nUniform = Math.round(batch * this.uniformMix);
    const total = this.tree.total;
    const uniform = nUniform / batch;
    if (!(total > 0)) throw new Error('Replay weights must have positive total');

    for (let b = 0; b < batch; b++) {
      let i, w;
      if (b < nUniform) {
        i = (rng() * this.n) | 0;            // uniform floor — always some plain coverage
        w = this._w(i);
      } else {
        ({ idx: i, w } = this.tree.find(rng() * total));
        if (i >= this.n) { i = (rng() * this.n) | 0; w = this._w(i); }
      }
      idx[b] = i;
      const pI = uniform / this.n + (1 - uniform) * w / total;
      isw[b] = pI > 0 ? Math.pow(1 / (this.n * pI), this.beta) : 1;
    }
    let mx = 0; for (const v of isw) if (v > mx) mx = v;
    if (mx > 0) for (let b = 0; b < batch; b++) isw[b] /= mx;   // normalize so max weight = 1
    return { idx, isw };
  }

  updatePriorities(idx, errs) {
    for (let b = 0; b < idx.length; b++) {
      const i = idx[b];
      this.prio[i] = Math.min(Math.abs(errs[b]) + 1e-4, this.priorityCap);
      this.tree.set(i, this._w(i));
    }
  }

  get size() { return this.n; }
}
