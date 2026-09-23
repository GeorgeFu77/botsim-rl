// Pull monthly 5m klines from Binance's public data archive.
// Public, free, no API key. ~478 KB/month for BTCUSDT 5m.
//   npm run download
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { config } from '../../config.js';

const OUT = 'data/klines';
const base = 'https://data.binance.vision/data/spot/monthly/klines';

// "2020-01" -> ["2020-01","2020-02",...] up to `to` (default: last complete month)
function months(from, to) {
  const [fy, fm] = from.split('-').map(Number);
  const end = to ? to.split('-').map(Number) : (() => {
    const d = new Date(); d.setUTCDate(1); d.setUTCMonth(d.getUTCMonth() - 1);
    return [d.getUTCFullYear(), d.getUTCMonth() + 1];
  })();
  const out = [];
  for (let y = fy, m = fm; y < end[0] || (y === end[0] && m <= end[1]); m++) {
    if (m > 12) { m = 1; y++; if (y > end[0]) break; }
    out.push(`${y}-${String(m).padStart(2, '0')}`);
  }
  return out;
}

fs.mkdirSync(OUT, { recursive: true });
const { symbol, interval } = config;
const want = months(config.history.from, config.history.to);

let got = 0, skipped = 0, missing = 0, bytes = 0;
for (const ym of want) {
  const name = `${symbol}-${interval}-${ym}`;
  const csv = path.join(OUT, `${name}.csv`);
  if (fs.existsSync(csv) && fs.statSync(csv).size > 0) { skipped++; continue; }

  const url = `${base}/${symbol}/${interval}/${name}.zip`;
  const zip = path.join(OUT, `${name}.zip`);
  try {
    execFileSync('curl', ['-sfL', '--max-time', '120', '-o', zip, url]);
    execFileSync('unzip', ['-oq', zip, '-d', OUT]);
    bytes += fs.statSync(zip).size;
    fs.unlinkSync(zip);
    got++;
    process.stdout.write(`\r  downloaded ${got}  (${ym})   `);
  } catch {
    missing++;                                   // month not published yet
    if (fs.existsSync(zip)) fs.unlinkSync(zip);
  }
}

const rows = fs.readdirSync(OUT).filter((f) => f.endsWith('.csv'))
  .reduce((n, f) => n + (fs.statSync(path.join(OUT, f)).size / 90 | 0), 0);
console.log(`\n\n  months: ${got} new, ${skipped} cached, ${missing} unavailable`);
console.log(`  downloaded: ${(bytes / 1048576).toFixed(1)} MB`);
console.log(`  ~${rows.toLocaleString()} candles on disk -> ${OUT}/`);
