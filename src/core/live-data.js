import fs from 'node:fs';

export const validProbability = (p) => Number.isFinite(p) && p > 0 && p < 1;
export const feePerShare = (price, rate) => rate * price * (1 - price);

export function atomicJSON(file, value) {
  fs.writeFileSync(`${file}.tmp`, JSON.stringify(value), { flush: true });
  fs.renameSync(`${file}.tmp`, file);
}

export function appendJSON(file, value) {
  fs.appendFileSync(file, JSON.stringify(value) + '\n', { flush: true });
}

// Only the single writer calls this at startup. Preserve a torn final record for
// inspection; complete malformed records still fail loudly instead of vanishing.
export function readJournal(file, offset = 0) {
  if (!fs.existsSync(file)) return [];
  const fd = fs.openSync(file, 'r');
  let text;
  try {
    const size = fs.fstatSync(fd).size;
    if (!Number.isSafeInteger(offset) || offset < 0 || offset > size) throw new Error('Invalid journal checkpoint offset');
    const buffer = Buffer.alloc(size - offset);
    let read = 0;
    while (read < buffer.length) {
      const n = fs.readSync(fd, buffer, read, buffer.length - read, offset + read);
      if (!n) throw new Error('Journal shortened while reading');
      read += n;
    }
    text = buffer.toString('utf8');
  } finally { fs.closeSync(fd); }
  if (text && !text.endsWith('\n')) {
    const end = text.lastIndexOf('\n') + 1, tail = text.slice(end);
    let complete = false;
    try { JSON.parse(tail); complete = true; } catch {}
    if (complete) { fs.appendFileSync(file, '\n', { flush: true }); text += '\n'; }
    else {
      fs.writeFileSync(`${file}.partial-${Date.now()}`, tail, { flag: 'wx', flush: true });
      fs.truncateSync(file, offset + Buffer.byteLength(text.slice(0, end)));
      console.warn(`event=repaired_partial_journal file=${file}`);
      text = text.slice(0, end);
    }
  }
  return text.split('\n').filter(Boolean).map(JSON.parse);
}

// Read only new bytes; an anchor also detects truncate-and-regrow between polls.
export class JsonlCursor {
  constructor(file, tailBytes = Infinity) {
    this.file = file; this.tailBytes = tailBytes;
    this.offset = null; this.partial = ''; this.anchor = Buffer.alloc(0);
  }

  read() {
    let fd;
    try { fd = fs.openSync(this.file, 'r'); }
    catch (e) { if (e.code === 'ENOENT') return []; throw e; }
    try {
      const { size, ino } = fs.fstatSync(fd);
      let discardFirst = false;
      if (this.offset === null) {
        this.offset = Math.max(0, size - this.tailBytes);
        discardFirst = this.offset > 0;
      } else {
        const anchor = Buffer.alloc(this.anchor.length);
        if (size >= this.offset) fs.readSync(fd, anchor, 0, anchor.length, this.offset - anchor.length);
        if (ino !== this.ino || size < this.offset || !anchor.equals(this.anchor)) {
          this.offset = 0; this.partial = '';
        }
      }
      this.ino = ino;
      const out = [];
      while (this.offset < size) {
        const b = Buffer.alloc(Math.min(1024 * 1024, size - this.offset));
        const got = fs.readSync(fd, b, 0, b.length, this.offset);
        if (!got) break;
        this.offset += got;
        const lines = (this.partial + b.toString('utf8', 0, got)).split('\n');
        this.partial = lines.pop();
        if (discardFirst && lines.length) { lines.shift(); discardFirst = false; }
        for (const line of lines) {
          if (!line.trim()) continue;
          try { out.push(JSON.parse(line)); }
          catch { console.warn(`event=invalid_jsonl file=${this.file}`); }
        }
      }
      this.anchor = Buffer.alloc(Math.min(64, this.offset));
      fs.readSync(fd, this.anchor, 0, this.anchor.length, this.offset - this.anchor.length);
      return out;
    } finally { fs.closeSync(fd); }
  }
}

export function priceAt(records, sourceTime, receivedBy, maxAge) {
  let best = null;
  for (const r of records) {
    if (r.src !== 'chainlink' || !(r.price > 0) || !Number.isFinite(r.price) ||
        !Number.isFinite(r.exchTs) || !Number.isFinite(r.recvTs) ||
        r.recvTs > receivedBy || r.exchTs > sourceTime || sourceTime - r.exchTs > maxAge) continue;
    if (!best || r.exchTs > best.exchTs) best = r;
  }
  return best;
}

export function bookAt(records, slug, outcome, at, cfg) {
  let best = null;
  for (const r of records) {
    if (r.slug !== slug || r.outcome !== outcome || !Number.isFinite(r.exchTs) || !Number.isFinite(r.recvTs) ||
        r.recvTs + cfg.feedDelayMs > at || r.exchTs > at || at - r.recvTs > cfg.maxFeedAgeMs ||
        at - r.exchTs > cfg.maxFeedAgeMs) continue;
    if (!best || r.exchTs > best.exchTs || (r.exchTs === best.exchTs && r.recvTs > best.recvTs)) best = r;
  }
  if (!best || !Array.isArray(best.bids) || !Array.isArray(best.asks) || !best.bids.length || !best.asks.length) return null;
  const valid = (l) => validProbability(l.price) && Number.isFinite(l.size) && l.size > 0;
  if (!best.bids.every(valid) || !best.asks.every(valid)) return null;
  const bids = [...best.bids].sort((a, b) => b.price - a.price);
  const asks = [...best.asks].sort((a, b) => a.price - b.price);
  if (bids[0].price >= asks[0].price) return null;
  return { ...best, bids, asks, mid: (bids[0].price + asks[0].price) / 2 };
}

// Spend includes fees. Never cross a level that would erase the required edge.
export function paperFill(book, budget, p, cfg) {
  if (!book?.asks?.length || !(budget > 0) || !validProbability(p)) return null;
  return fillAsks(book, budget, cfg, (price, fee) => p - price - fee > cfg.margin);
}

// Shared hypothetical depth execution. Each experimental account is a separate
// counterfactual world; account fills cannot be summed into executable liquidity.
export function fillAsks(book, budget, cfg, accept = () => true) {
  if (!book?.asks?.length || !Number.isFinite(budget) || budget <= 0) return null;
  if (!book.asks.every((l) => validProbability(l.price) && Number.isFinite(l.size) && l.size > 0)) return null;
  let shares = 0, cost = 0, fees = 0, notional = 0;
  for (const { price, size } of book.asks) {
    const fee = feePerShare(price, cfg.feeRate);
    if (price < cfg.priceMin || price > cfg.priceMax || !accept(price, fee)) break;
    const take = Math.min(size, (budget - cost) / (price + fee));
    if (!(take > 0)) break;
    shares += take; fees += take * fee; notional += take * price; cost += take * (price + fee);
  }
  return cost >= (cfg.minPaperSpend ?? 1) ? { shares, cost, fees, q: notional / shares } : null;
}
