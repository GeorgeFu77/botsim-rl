import fs from 'node:fs';
import path from 'node:path';
import { config } from '../config.js';

const file = path.join(config.lab.resultDir, 'state.json');
if (!fs.existsSync(file)) { console.log('Paper learning lab has not started.'); process.exit(1); }
const s = JSON.parse(fs.readFileSync(file, 'utf8')), b = s.brain, l = s.learning;
let running = false;
try { process.kill(s.pid, 0); running = true; } catch (e) { if (e.code !== 'ESRCH') throw e; }
const fresh = Date.now() - s.heartbeatAt < 45_000;
const atCost = s.risk.equity;
console.log(`BIG BRAIN · PAPER ONLY · ${running && fresh ? 'RUNNING' : 'STOPPED / STALE'}`);
console.log(`Starting cash $80.00 · cash $${b.bankroll.toFixed(2)} · realized P/L $${b.pnl.toFixed(2)} · equity at entry cost $${atCost.toFixed(2)}`);
console.log(`Closed ${b.settled} · wins ${b.wins} · losses ${b.settled - b.wins} · win rate ${b.settled ? (100 * b.wins / b.settled).toFixed(1) + '%' : 'n/a'}`);
console.log(`Open ${Object.keys(b.open).length} · pending ${Object.keys(b.pending).length} · realized drawdown $${b.maxDrawdown.toFixed(2)} · fees $${b.fees.toFixed(2)}`);
console.log(`Entries: ${s.risk.reason ?? 'enabled'} · data: ${s.feedStatus} · heartbeat ${((Date.now() - s.heartbeatAt) / 1000).toFixed(1)}s ago`);
console.log(`Shared learner: ${s.workers} virtual workers in one process · ${l.updates} live market updates · ${l.uniqueExperiences} distinct filled-action examples`);
console.log(`Latest resolved market: ${l.lastMarketEnd ? new Date(l.lastMarketEnd).toISOString() : 'waiting for first outcome'} · last weight update ${l.lastLearningAt ? new Date(l.lastLearningAt).toISOString() : 'waiting for worker outcomes'}`);
console.log(`Active ${l.activeId} · candidate gate ${l.evaluatedMarkets}/${l.requiredMarkets} markets, also needs 24h · promotions ${l.promotions}`);
console.log(`Replay seed: ${l.seed?.trainMarkets ?? 0} historical filled markets · never counted as new live trades or wallet profits`);
if (l.lastEvaluation) console.log(`Last candidate check: ${l.lastEvaluation.promoted ? 'accepted' : 'not accepted'} · reference P/L $${l.lastEvaluation.reference.pnl.toFixed(2)} · candidate P/L $${l.lastEvaluation.candidate.pnl.toFixed(2)}`);
console.log('Only the central account is shown. No real wallets, credentials, orders or pooled worker profits.');
if (s.lastError) console.log(`Source errors this process: ${s.sourceErrors} · latest: ${s.lastError}`);
if (!running || !fresh) process.exitCode = 1;
else if (l.stale && Date.now() - (s.startedAt ?? s.serviceStartedAt ?? s.heartbeatAt) > 1_800_000) process.exitCode = 2;
