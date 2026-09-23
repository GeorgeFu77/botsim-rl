export function marketEnd(slug) {
  const match = /^btc-updown-5m-(\d+)$/.exec(slug);
  const start = Number(match?.[1]);
  return Number.isSafeInteger(start) && start > 0 && start % 300 === 0 ? start + 300 : null;
}

// Never infer settlement from a near-1 price, closed flag alone, or another slug.
export function confirmedResolution(body, slug, now = Date.now()) {
  const end = marketEnd(slug);
  if (!end || end * 1000 > now || body.slug !== slug || body.closed !== true ||
      body.umaResolutionStatus !== 'resolved' || Date.parse(body.endDate) !== end * 1000) return null;
  const outcomes = typeof body.outcomes === 'string' ? JSON.parse(body.outcomes) : body.outcomes;
  const prices = typeof body.outcomePrices === 'string' ? JSON.parse(body.outcomePrices) : body.outcomePrices;
  if (!Array.isArray(outcomes) || !Array.isArray(prices) || outcomes.length !== 2 || prices.length !== 2 ||
      new Set(outcomes).size !== 2 || !outcomes.every((x) => ['Up', 'Down'].includes(x))) return null;
  const values = prices.map((p) => typeof p === 'string' && /^[01]$/.test(p) ? Number(p) : p);
  if (!values.includes(1) || !values.includes(0)) return null;
  return { type: 'resolution', slug, periodStart: end - 300, periodEnd: end,
    outcome: outcomes[values.indexOf(1)], source: 'gamma confirmed', receivedAt: now };
}

export async function fetchResolution(slug) {
  const response = await fetch(`https://gamma-api.polymarket.com/markets/slug/${encodeURIComponent(slug)}`,
    { signal: AbortSignal.timeout(8000) });
  if (response.status === 404) return null;
  if (!response.ok) throw new Error(`Resolution HTTP ${response.status}: ${slug}`);
  return confirmedResolution(await response.json(), slug);
}
