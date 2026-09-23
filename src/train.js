// Pretrain the direction predictor on Binance history, with experience replay
// and a CHRONOLOGICAL holdout.
//
// The split is by time, never random. A random split lets the model train on
// next week while being tested on last week — the future leaks backward and
// every number it prints becomes a lie.
//
//   npm run train
import fs from 'node:fs';
import { config } from '../config.js';
import { MLP, mulberry32 } from './core/mlp.js';
import { ReplayBuffer } from './core/replay.js';
import { evaluate, report, reliabilityTable } from './core/calibration.js';
import { Platt } from './core/platt.js';
import { FEATURE_VERSION, FEATURES } from './core/features.js';

const meta = JSON.parse(fs.readFileSync('data/meta.json', 'utf8'));
const { dim, n } = meta;
if (meta.featureVersion !== FEATURE_VERSION || JSON.stringify(meta.features) !== JSON.stringify(FEATURES)) {
  throw new Error('Feature schema changed. Run npm run episodes before training.');
}
function tensor(file, Type, count) {
  const b = fs.readFileSync(file);
  if (b.length !== count * Type.BYTES_PER_ELEMENT) throw new Error(`Invalid tensor size: ${file}`);
  return new Type(b.buffer, b.byteOffset, count);
}
const X = tensor('data/X.bin', Float32Array, n * dim);
const Y = tensor('data/Y.bin', Uint8Array, n);
const T = tensor('data/T.bin', Float64Array, n);

// ---- chronological split -----------------------------------------------------
// Never split the four observations of one market across partitions.
const boundary = (fraction) => {
  let i = Math.floor(n * fraction);
  while (i < n && T[i] === T[i - 1]) i++;
  return i;
};
const iTrain = boundary(config.split.train);
const iCal = boundary(config.split.train + config.split.val / 2);
const iVal = boundary(config.split.train + config.split.val);
const trainRows = Array.from({ length: iTrain }, (_, i) => i);
const valRows = Array.from({ length: iCal - iTrain }, (_, i) => iTrain + i);
const calRows = Array.from({ length: iVal - iCal }, (_, i) => iCal + i);
const testRows = Array.from({ length: n - iVal }, (_, i) => iVal + i);
const day = (i) => new Date(T[i]).toISOString().slice(0, 10);

console.log(`\n  BotSimRL — pretraining on ${n.toLocaleString()} episodes, dim=${dim}`);
console.log(`  base rate (Up): ${(meta.baseRate * 100).toFixed(2)}%\n`);
console.log(`  train  ${day(0)} -> ${day(iTrain - 1)}   ${iTrain.toLocaleString()}`);
console.log(`  select ${day(iTrain)} -> ${day(iCal - 1)}   ${valRows.length.toLocaleString()}`);
console.log(`  calibrate ${day(iCal)} -> ${day(iVal - 1)}   ${calRows.length.toLocaleString()}`);
console.log(`  test   ${day(iVal)} -> ${day(n - 1)}   ${testRows.length.toLocaleString()}   [historical regression set, previously inspected]`);

// ---- model + replay ----------------------------------------------------------
const model = new MLP(dim, { ...config.model, seed: config.train.seed });
model.fitScaler(X, dim, trainRows);              // train rows only — no leakage

const buf = new ReplayBuffer(config.replay);
if (iTrain > buf.cap) throw new Error('Replay capacity is smaller than the training split');
for (const i of trainRows) buf.push(i, Y[i], T[i]);
buf.refreshRecency();
console.log(`\n  replay buffer: ${buf.size.toLocaleString()} episodes (cap ${config.replay.capacity.toLocaleString()})`);
console.log(`  recency half-life: ${config.replay.recencyHalfLifeDays}d · prioritized=${config.replay.prioritized} · uniform floor=${config.replay.uniformMix * 100}%\n`);

// ReplayBuffer stores row indices as its "x"; map sampled slots back to rows.
const rowOf = (slot) => buf.x[slot];
const rng = mulberry32(config.train.seed);
const { epochs, batch } = config.train;
const stepsPerEpoch = Math.floor(buf.size / batch);

let best = { brier: Infinity, epoch: -1, snap: null };
const trainSample = trainRows.filter((_, k) => k % 20 === 0);
const history = [];
for (let e = 1; e <= epochs; e++) {
  // Anneal importance-sampling beta toward 1.0, as the PER paper prescribes.
  buf.beta = config.replay.beta + (1 - config.replay.beta) * ((e - 1) / Math.max(epochs - 1, 1));

  for (let s = 0; s < stepsPerEpoch; s++) {
    const { idx, isw } = buf.sample(batch, rng);
    const rows = Int32Array.from(idx, rowOf);
    const errs = model.trainBatch(X, Y, rows, isw);
    buf.updatePriorities(idx, errs);
  }

  const tr = evaluate(model, X, Y, trainSample);
  const va = evaluate(model, X, Y, valRows);
  const mark = va.brier < best.brier ? '  <- best' : '';
  if (va.brier < best.brier) best = { brier: va.brier, epoch: e, snap: model.toJSON() };
  history.push({ epoch: e, trainBrier: tr.brier, valBrier: va.brier });
  console.log(`  epoch ${String(e).padStart(2)}  train brier ${tr.brier.toFixed(5)}   val brier ${va.brier.toFixed(5)}   val acc ${(va.acc * 100).toFixed(2)}%   auc ${va.auc.toFixed(4)}${mark}`);
}

// ---- historical regression report ------------------------------------------
console.log(`\n  best epoch: ${best.epoch} (val brier ${best.brier.toFixed(5)})`);
if (!best.snap) throw new Error('Training failed to produce finite validation metrics');
console.log('\n  Historical regression evaluation — this set is not a fresh forward test.');

Object.assign(model, {
  W1: Float64Array.from(best.snap.W1), b1: Float64Array.from(best.snap.b1),
  W2: Float64Array.from(best.snap.W2), b2: best.snap.b2,
});

// Calibrate on the separate later block, not on epoch-selection data or test.
const scratch = new Float64Array(model.d);
const valP = calRows.map((i) => model.predict(X, i, scratch));
const valY = calRows.map((i) => Y[i]);
const platt = new Platt().fit(valP, valY);
console.log(`\n  Platt fitted on separate calibration block: a=${platt.a.toFixed(4)} b=${platt.b.toFixed(4)}`);

const teRaw = evaluate(model, X, Y, testRows);
const te = evaluate(model, X, Y, testRows, platt);
report('TEST — raw (uncalibrated)', teRaw);
report('TEST — calibrated', te);
reliabilityTable(te);
te.byOffset = {};
for (const off of meta.offsets) {
  const rows = testRows.filter((i) => Math.round(X[i * dim + 2] * 300) === off);
  te.byOffset[off] = evaluate(model, X, Y, rows, platt);
  console.log(`  offset ${off}s: n=${rows.length} accuracy=${(te.byOffset[off].acc * 100).toFixed(2)}% brier=${te.byOffset[off].brier.toFixed(5)}`);
}
te.uniqueMarkets = new Set(testRows.map((i) => T[i])).size;
te.training = { featureVersion: FEATURE_VERSION, createdAt: new Date().toISOString(), bestEpoch: best.epoch,
  history, split: { trainEnd: T[iTrain - 1], selectEnd: T[iCal - 1], calibrationEnd: T[iVal - 1] } };

fs.mkdirSync('results', { recursive: true });
fs.writeFileSync('results/model.json.tmp', JSON.stringify({ ...best.snap, platt: platt.toJSON(),
  features: FEATURES, featureVersion: FEATURE_VERSION, training: te.training }));
fs.renameSync('results/model.json.tmp', 'results/model.json');
fs.writeFileSync('results/test-metrics.json', JSON.stringify(te, null, 2));

console.log(`\n  ${'-'.repeat(66)}`);
console.log('  These are BTC direction metrics, not evidence of a profitable trading edge.');
console.log('  Compare against executable market prices on new live observations.');
console.log(`  Brier improvement from calibration: ${(teRaw.brier - te.brier).toFixed(5)}`);
console.log(`  ${'-'.repeat(66)}\n`);
