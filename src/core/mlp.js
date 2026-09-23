// Small MLP: input -> tanh hidden -> sigmoid. Same shape as BotSim's OnlineMLP,
// but minibatch-capable and it accepts per-sample importance weights (needed to
// undo prioritized-replay bias).
//
// Standardization is FIT ON TRAIN ONLY, then frozen. Fitting it across the whole
// dataset would leak test-set mean/variance backward into training — a quiet,
// very common way to make a model look better than it is.

const sigmoid = (z) => 1 / (1 + Math.exp(-z));

// Deterministic RNG so runs are reproducible.
export function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class MLP {
  constructor(dim, { hidden = 24, lr = 0.01, l2 = 1e-5, seed = 1, gradientClip = Infinity } = {}) {
    const rnd = mulberry32(seed);
    this.d = dim; this.h = hidden; this.lr = lr; this.l2 = l2; this.gradientClip = gradientClip;
    const s1 = Math.sqrt(2 / dim), s2 = Math.sqrt(2 / hidden);   // He-style init
    this.W1 = Float64Array.from({ length: dim * hidden }, () => (rnd() * 2 - 1) * s1);
    this.b1 = new Float64Array(hidden);
    this.W2 = Float64Array.from({ length: hidden }, () => (rnd() * 2 - 1) * s2);
    this.b2 = 0;
    this.mean = new Float64Array(dim);
    this.std = new Float64Array(dim).fill(1);
    this.hAct = new Float64Array(hidden);
    this.z = new Float64Array(dim);
    this.gW1 = new Float64Array(dim * hidden);
    this.gb1 = new Float64Array(hidden);
    this.gW2 = new Float64Array(hidden);
  }

  // Fit input standardization on training rows only, then freeze.
  fitScaler(X, dim, rows) {
    const n = rows.length;
    for (let j = 0; j < dim; j++) {
      let s = 0; for (const i of rows) s += X[i * dim + j];
      const m = s / n;
      let v = 0; for (const i of rows) v += (X[i * dim + j] - m) ** 2;
      this.mean[j] = m;
      this.std[j] = Math.sqrt(v / Math.max(n - 1, 1)) || 1;
    }
  }

  _z(X, i, out) {
    for (let j = 0; j < this.d; j++) out[j] = (X[i * this.d + j] - this.mean[j]) / this.std[j];
    return out;
  }

  _forward(z) {
    const hAct = this.hAct;
    for (let j = 0; j < this.h; j++) {
      let s = this.b1[j];
      for (let i = 0; i < this.d; i++) s += this.W1[i * this.h + j] * z[i];
      hAct[j] = Math.tanh(s);
    }
    let o = this.b2;
    for (let j = 0; j < this.h; j++) o += this.W2[j] * hAct[j];
    return { p: sigmoid(o), hAct };
  }

  predict(X, i, scratch = this.z) {
    return this._forward(this._z(X, i, scratch)).p;
  }

  // One minibatch of gradient descent. `isw` scales each sample's gradient.
  // Returns per-sample |p - y|, which feeds back as replay priorities.
  trainBatch(X, Y, idx, isw) {
    const B = idx.length;
    const gW1 = this.gW1.fill(0), gb1 = this.gb1.fill(0);
    const gW2 = this.gW2.fill(0); let gb2 = 0;
    const errs = new Float64Array(B);
    const z = this.z;

    for (let b = 0; b < B; b++) {
      const i = idx[b];
      this._z(X, i, z);
      const { p, hAct } = this._forward(z);
      const y = Y[i];
      errs[b] = Math.abs(p - y);
      const g = (p - y) * (isw ? isw[b] : 1);           // dLoss/dOutput for sigmoid + log-loss
      for (let j = 0; j < this.h; j++) {
        gW2[j] += g * hAct[j];
        const dh = g * this.W2[j] * (1 - hAct[j] * hAct[j]);
        gb1[j] += dh;
        for (let k = 0; k < this.d; k++) gW1[k * this.h + j] += dh * z[k];
      }
      gb2 += g;
    }

    const norm = Math.sqrt([...gW1, ...gb1, ...gW2, gb2].reduce((s, g) => s + g * g, 0)) / B;
    if (!Number.isFinite(norm)) throw new Error('Non-finite MLP gradient');
    const lr = this.lr / B * Math.min(1, this.gradientClip / Math.max(norm, 1e-12));
    for (let j = 0; j < this.h; j++) {
      this.W2[j] -= lr * gW2[j] + this.lr * this.l2 * this.W2[j];
      this.b1[j] -= lr * gb1[j];
      for (let k = 0; k < this.d; k++) {
        const t = k * this.h + j;
        this.W1[t] -= lr * gW1[t] + this.lr * this.l2 * this.W1[t];
      }
    }
    this.b2 -= lr * gb2;
    return errs;
  }

  toJSON() {
    return { d: this.d, h: this.h, lr: this.lr, l2: this.l2,
      W1: [...this.W1], b1: [...this.b1], W2: [...this.W2], b2: this.b2,
      mean: [...this.mean], std: [...this.std] };
  }
}
