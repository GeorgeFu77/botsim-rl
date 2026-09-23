import fs from 'node:fs';
const m = JSON.parse(fs.readFileSync('results/test-metrics.json', 'utf8'));
console.log(`Historical evaluation: ${m.n.toLocaleString()} observations in ${m.uniqueMarkets?.toLocaleString() ?? 'unknown'} markets.`);
console.log(`Brier ${m.brier.toFixed(5)} · accuracy ${(m.acc * 100).toFixed(2)}% · AUC ${m.auc.toFixed(4)}`);
for (const [offset, v] of Object.entries(m.byOffset ?? {})) {
  console.log(`${offset}s: Brier ${v.brier.toFixed(5)} · accuracy ${(v.acc * 100).toFixed(2)}% · n=${v.n}`);
}
console.log('The historical set has been inspected before. Offsets share outcomes and are not independent samples.');
console.log('Train labels are Binance; live inputs use Coinbase history plus Chainlink prices. Venue drift remains.');
console.log('Use npm run score for a forward comparison against the market; offline accuracy is not trading edge.');
