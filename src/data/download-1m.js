// Pull 1-minute klines — the same archive, finer grain. These let the model see
// what BTC has done INSIDE the 5-min window, which is the single thing the
// market prices and a candle-open-only model is blind to.
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { config } from '../../config.js';

const OUT = 'data/klines1m';
const base = 'https://data.binance.vision/data/spot/monthly/klines';
fs.mkdirSync(OUT, { recursive: true });

const have = fs.readdirSync('data/klines').filter((f) => f.endsWith('.csv'))
  .map((f) => f.match(/(\d{4}-\d{2})\.csv$/)?.[1]).filter(Boolean).sort();

let got = 0, skipped = 0, missing = 0, bytes = 0;
for (const ym of have) {
  const name = `${config.symbol}-1m-${ym}`;
  const csv = path.join(OUT, `${name}.csv`);
  if (fs.existsSync(csv) && fs.statSync(csv).size > 0) { skipped++; continue; }
  const zip = path.join(OUT, `${name}.zip`);
  try {
    execFileSync('curl', ['-sfL', '--max-time', '180', '-o', zip, `${base}/${config.symbol}/1m/${name}.zip`]);
    execFileSync('unzip', ['-oq', zip, '-d', OUT]);
    bytes += fs.statSync(zip).size;
    fs.unlinkSync(zip); got++;
    process.stdout.write(`\r  ${got}/${have.length}  ${ym}   `);
  } catch { missing++; if (fs.existsSync(zip)) fs.unlinkSync(zip); }
}
console.log(`\n  1m months: ${got} new, ${skipped} cached, ${missing} unavailable · ${(bytes / 1048576).toFixed(0)} MB`);
