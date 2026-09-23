// Synthetic compute benchmark, never a performance/backtest result. No network,
// no production ledger writes, and no claims of independent market evidence.
import fs from 'node:fs';
import path from 'node:path';
import { performance } from 'node:perf_hooks';
import { config } from '../config.js';
import { PaperLab } from './core/paper-lab.js';
import { atomicJSON } from './core/live-data.js';

const runs = [];
for (const workers of [32, 64, 128]) {
  const samples = [];
  for (let repeat = 0; repeat < 3; repeat++) {
    const lab = new PaperLab({ ...config.lab, workers }, config.live);
    const start = performance.now();
    for (let i = 0; i < 500; i++) {
      const t = 1_789_002_000 + i * 300, slug = `btc-updown-5m-${t}`, at = (t + 60) * 1000;
      lab.observe({ slug, key: `${slug}:60`, offset: 60, at, pUp: i % 2 ? 0.70 : 0.30,
        askUp: 0.49, askDown: 0.53, elapsedSec: 60, features: new Array(25).fill(0.01) }, at);
      const books = ['Up', 'Down'].map((outcome) => ({ slug, outcome, exchTs: at + 1100, recvTs: at + 1100,
        bids: [{ price: 0.45, size: 10 }], asks: [{ price: outcome === 'Up' ? 0.49 : 0.53, size: 100 }] }));
      lab.fill(books, at + 2200);
      lab.resolve(new Map([[slug, { slug, periodEnd: t + 300, outcome: i % 3 ? 'Up' : 'Down', source: 'synthetic benchmark' }]]), (t + 301) * 1000);
    }
    samples.push((performance.now() - start) / 500);
  }
  samples.sort((a, b) => a - b);
  runs.push({ workers, medianMsPerMarketCycle: samples[1], accountCyclesPerSecond: workers * 1000 / samples[1] });
}
const best = runs.filter((r) => r.medianMsPerMarketCycle < 50).sort((a, b) => b.accountCyclesPerSecond - a.accountCyclesPerSecond)[0];
const report = { generatedAt: new Date().toISOString(), kind: 'synthetic batched CPU benchmark; excludes disk/network; not investment results',
  productionProcessCount: 1, runs, recommendedWorkers: best?.workers ?? 32, rssMB: process.memoryUsage().rss / 1024 ** 2 };
fs.mkdirSync(config.lab.resultDir, { recursive: true });
atomicJSON(path.join(config.lab.resultDir, 'benchmark.json'), report);
console.log(JSON.stringify(report, null, 2));
