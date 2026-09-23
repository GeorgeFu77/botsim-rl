import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { JsonlCursor, priceAt, bookAt, paperFill } from './live-data.js';
import { ReplayBuffer } from './replay.js';
import { extract, WARM } from './features.js';
import { evaluate } from './calibration.js';
import { MLP } from './mlp.js';
import { config } from '../../config.js';

test('Chainlink selection excludes other sources, future data and stale prices', () => {
  const records = [
    { src: 'chainlink', price: 100, exchTs: 1000, recvTs: 1100 },
    { src: 'polybinance', price: 999, exchTs: 1100, recvTs: 1200 },
    { src: 'chainlink', price: 200, exchTs: 2000, recvTs: 2100 },
    { src: 'chainlink', price: 300, exchTs: 900, recvTs: 5000 },
  ];
  assert.equal(priceAt(records, 1500, 1500, 1000).price, 100);
  assert.equal(priceAt(records, 1500, 1500, 100), null);
  assert.equal(priceAt(records, 1000, 1500, 1000).price, 100);
});

test('book selection rejects stale, crossed and invalid books without reviving an older book', () => {
  const cfg = { ...config.live, feedDelayMs: 100, maxFeedAgeMs: 1000 };
  const book = { slug: 's', outcome: 'Up', exchTs: 1000, recvTs: 1100,
    bids: [{ price: 0.4, size: 10 }], asks: [{ price: 0.5, size: 10 }] };
  assert.equal(bookAt([book], 's', 'Up', 1500, cfg).mid, 0.45);
  assert.equal(bookAt([book], 's', 'Up', 1150, cfg), null);
  assert.equal(bookAt([book], 's', 'Up', 2500, cfg), null);
  assert.equal(bookAt([book, { ...book, exchTs: 1200, bids: [{ price: 0.6, size: 1 }] }], 's', 'Up', 1500, cfg), null);
});

test('paper fills cap spending including fees, respect depth and stop at unprofitable prices', () => {
  const book = { asks: [{ price: 0.5, size: 2 }, { price: 0.9, size: 100 }] };
  const fill = paperFill(book, 20, 0.7, config.live);
  assert.equal(fill.shares, 2);
  assert.ok(Math.abs(fill.cost - 1.035) < 1e-12);
  assert.equal(fill.q, 0.5);
  const limited = paperFill({ asks: [{ price: 0.5, size: 100 }] }, 2, 0.7, config.live);
  assert.ok(Math.abs(limited.cost - 2) < 1e-12);
  assert.equal(paperFill(book, 20, 0.51, config.live), null);
});

test('JSONL cursor handles partial writes and same-inode truncation followed by regrowth', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'botsimrl-test-'));
  const file = path.join(dir, 'feed.jsonl');
  try {
    fs.writeFileSync(file, '{"n":1}\n{"n":');
    const c = new JsonlCursor(file);
    assert.deepEqual(c.read(), [{ n: 1 }]);
    fs.appendFileSync(file, '2}\n');
    assert.deepEqual(c.read(), [{ n: 2 }]);
    assert.deepEqual(c.read(), []);
    fs.writeFileSync(file, '{"n":300}\n{"n":400}\n{"n":500}\n');
    assert.deepEqual(c.read(), [{ n: 300 }, { n: 400 }, { n: 500 }]);
  } finally { fs.unlinkSync(file); fs.rmdirSync(dir); }
});

test('importance correction uses the uniform/prioritized mixture probability', () => {
  const b = new ReplayBuffer({ ...config.replay, capacity: 2, prioritized: false,
    recencyHalfLifeDays: 1, beta: 1, uniformMix: 0.5 });
  b.push(0, 0, 0); b.push(1, 1, 3 * 86400_000); b.refreshRecency();
  const draws = [0, 0.99, 0, 0.99];
  const { idx, isw } = b.sample(4, () => draws.shift());
  assert.deepEqual([...idx], [0, 1, 0, 1]);
  const expected = (0.25 + 0.5 / 9) / (0.25 + 0.5 * 8 / 9);
  assert.ok(Math.abs(isw[1] - expected) < 1e-12);
  assert.equal(isw[0], 1);
});

test('features have neutral RSI on flat history and real return acceleration', () => {
  const k = Array.from({ length: WARM + 1 }, (_, i) => ({ t: i * 300_000, o: 100, h: 100, l: 100, c: 100, v: 1 }));
  const flat = extract(k, WARM, 100, 100, 60);
  assert.equal(flat[20], 0); assert.equal(flat[21], 0);
  k[WARM - 1].c = 102;
  const changed = extract(k, WARM, 100, 101, 120);
  assert.ok(Math.abs(changed[21] - Math.log(1.02)) < 1e-12);
  assert.equal(changed[20], 0.5);
  assert.ok(changed.every(Number.isFinite));
});

test('equal predictions produce AUC 0.5 regardless of label order', () => {
  const model = { d: 1, predict: () => 0.5 };
  assert.equal(evaluate(model, [0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 2, 3]).auc, 0.5);
});

test('reused MLP scratch is deterministic across predictions and training batches', () => {
  const a = new MLP(2), b = new MLP(2), X = [1, -1, -1, 1], Y = [1, 0];
  const p = a.predict(X, 0); a.predict(X, 1);
  assert.equal(a.predict(X, 0), p);
  for (let i = 0; i < 5; i++) {
    a.trainBatch(X, Y, [0, 1]); b.trainBatch(X, Y, [0, 1]);
    a.predict(X, 1);
  }
  assert.deepEqual(a.toJSON(), b.toJSON());
});
