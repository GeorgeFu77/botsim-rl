import fs from 'node:fs';
import path from 'node:path';
import { config } from '../config.js';

const file = path.join(config.live.resultDir, 'live-state.json');
if (!fs.existsSync(file)) { console.log('Paper agent has not started.'); process.exit(1); }
const s = JSON.parse(fs.readFileSync(file, 'utf8'));
let running = false;
try { process.kill(s.pid, 0); running = true; } catch (e) { if (e.code !== 'ESRCH') throw e; }
const age = Date.now() - s.heartbeatAt;
console.log(`Paper agent: ${running && age < 45_000 ? 'RUNNING' : 'STOPPED / STALE'} · PID ${s.pid} · heartbeat ${(age / 1000).toFixed(1)}s ago`);
console.log(`Feed status: ${s.status} · errors ${s.errors} · overdue settlements ${s.overdue.length}`);
console.log(`Cash $${s.bankroll.toFixed(2)} · realized P/L $${s.pnl.toFixed(2)} · settled ${s.settled} · observations ${s.seen}`);
console.log(`Open ${Object.keys(s.open).length} · pending ${Object.keys(s.pending).length} · model ${s.modelId}`);
if (s.learning) {
  const l = s.learning;
  console.log(`Learning: ${l.stale ? 'WAITING / STALE' : 'ACTIVE'} · ${l.updates} market updates · last ${l.lastUpdateAt ? new Date(l.lastUpdateAt).toISOString() : 'never'}`);
  console.log(`Newest learned outcome: ${l.lastMarketEnd ? new Date(l.lastMarketEnd).toISOString() : 'none'} · unresolved markets ${l.waitingMarkets}`);
  console.log(`Learner ${l.learnerId} · candidate gate ${l.evaluatedMarkets}/${l.requiredMarkets} markets (also needs 24h) · promotions ${l.promotions}`);
}
if (s.risk) console.log(`Paper entries: ${s.risk.reason ?? 'enabled'} · daily gross losses $${s.risk.dailyLosses.toFixed(2)} · reserved $${s.risk.reserved.toFixed(2)} · drawdown floor $${s.risk.floor.toFixed(2)}`);
if (s.recoveryError) console.log(`Settlement recovery retrying: ${s.recoveryError}`);
console.log('PAPER ONLY — no wallet connection; forecast accuracy is not a trade win rate.');
if (s.lastError) console.log(`Last error: ${s.lastError}`);
if (!running || age >= 45_000) process.exitCode = 1;
else if (s.learning?.stale) process.exitCode = 2;
