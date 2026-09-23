// Educational one-step reward learner: Up / Down / abstain. Not a sequential
// trading policy, not an LLM, and never connected to a brokerage or wallet.
import { createHash } from 'node:crypto';
import { feePerShare, validProbability } from './live-data.js';
import { mulberry32 } from './mlp.js';

export const BANDIT_DIM = 8;
const clip = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
export const policyId = (w) => createHash('sha256').update(JSON.stringify(w)).digest('hex').slice(0, 12);
export const initialPolicy = () => [[0, 1, -1, 0, 0, 0, 0, 0], [1, -1, 0, -1, 0, 0, 0, 0]];
export function validatePolicy(w) {
  if (!Array.isArray(w) || w.length !== 2 || w.some((row) => row.length !== BANDIT_DIM || !row.every(Number.isFinite))) {
    throw new Error('Invalid bandit policy');
  }
  return w;
}
export function decisionFeatures(o, feeRate) {
  if (![o.pUp, o.askUp, o.askDown].every(validProbability) || o.features?.length !== 25 ||
      !o.features.every(Number.isFinite) || !Number.isFinite(o.elapsedSec)) throw new Error('Invalid lab observation');
  return [1, o.pUp, o.askUp + feePerShare(o.askUp, feeRate), o.askDown + feePerShare(o.askDown, feeRate),
    clip(o.elapsedSec / 300, 0, 1), clip(o.features[1] / 5, -1, 1),
    clip(o.features[4] * 100, -1, 1), clip(o.features[11] * 100, 0, 1)];
}
export function actionValues(w, x) {
  return [0, ...w.map((row) => clip(row.reduce((s, a, i) => s + a * x[i], 0), -1, 1))];
}
export function chooseAction(values, random, epsilon = 0, margin = 0.02) {
  if (random() < epsilon) return Math.floor(random() * 3);
  const best = values[1] >= values[2] ? 1 : 2;
  return values[best] > margin ? best : 0;
}
export function validExperience(row) {
  return [1, 2].includes(row.action) && row.x?.length === BANDIT_DIM && row.x.every(Number.isFinite) &&
    Number.isFinite(row.reward) && row.reward >= -1 && row.reward <= 1;
}
// One gradient step per unique market. Within a market duplicated worker actions
// are collapsed; more virtual wallets must not multiply evidence by their count.
export function trainReward(w, groups, lr = 0.02) {
  const grad = [new Array(BANDIT_DIM).fill(0), new Array(BANDIT_DIM).fill(0)];
  const usable = groups.filter((g) => g.rows.length);
  for (const g of usable) for (const row of g.rows) {
    if (!validExperience(row)) throw new Error('Invalid reward experience');
    const prediction = w[row.action - 1].reduce((s, a, i) => s + a * row.x[i], 0);
    const error = clip(prediction - row.reward, -1, 1) / (usable.length * g.rows.length);
    for (let i = 0; i < BANDIT_DIM; i++) grad[row.action - 1][i] += error * row.x[i];
  }
  const norm = Math.sqrt(grad.flat().reduce((s, v) => s + v * v, 0));
  for (let a = 0; a < 2; a++) for (let i = 0; i < BANDIT_DIM; i++) w[a][i] -= lr * grad[a][i] / Math.max(1, norm);
  validatePolicy(w);
}
export function replayUpdate(state, group, cfg) {
  const rounded = (v) => Math.round(v * 1e9) / 1e9;
  const unique = new Map(group.rows.map((r) => [JSON.stringify([r.action, r.x.map(rounded), rounded(r.reward)]), r]));
  const current = { slug: group.slug, rows: [...unique.values()] };
  if (!current.rows.length) return;
  const random = mulberry32(9001 + state.updates), batch = [current];
  for (let i = 1; i < cfg.batchMarkets && state.replay.length; i++) batch.push(state.replay[Math.floor(random() * state.replay.length)]);
  trainReward(state.weights, batch, cfg.lr);
  state.replay.push(current);
  if (state.replay.length > cfg.replayMarkets) state.replay.shift();
  state.updates++; state.experiences += current.rows.length;
}
