// Platt scaling — fixes overconfidence without retraining the model.
//
// The network's RANKING was already fine (AUC above chance); only its numbers
// were wrong — it said 0.72 where reality was 0.49. Platt fits a 1-D logistic
// on top of the raw output: p_cal = sigmoid(a * logit(p_raw) + b), learned on
// the VALIDATION split only. Never on test, never on train — train is the split
// the model already overfit, so its probabilities look better there than they are.
//
// This matters because bet sizing (Kelly) consumes the probability directly.
// A model that is right about direction but wrong about confidence bets the
// wrong amount every single time.

const sigmoid = (z) => 1 / (1 + Math.exp(-z));
const logit = (p) => Math.log(Math.max(p, 1e-9) / Math.max(1 - p, 1e-9));

export class Platt {
  constructor() { this.a = 1; this.b = 0; }

  fit(ps, ys, { iters = 400, lr = 0.1 } = {}) {
    const z = ps.map(logit);
    for (let it = 0; it < iters; it++) {
      let ga = 0, gb = 0;
      for (let i = 0; i < z.length; i++) {
        const g = sigmoid(this.a * z[i] + this.b) - ys[i];
        ga += g * z[i]; gb += g;
      }
      this.a -= lr * ga / z.length;
      this.b -= lr * gb / z.length;
    }
    return this;
  }

  apply(p) { return sigmoid(this.a * logit(p) + this.b); }
  toJSON() { return { a: this.a, b: this.b }; }
  static fromJSON(o) { const p = new Platt(); p.a = o.a; p.b = o.b; return p; }
}
