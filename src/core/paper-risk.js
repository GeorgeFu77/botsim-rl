// Loss limits for the educational simulator, not a real-money risk guarantee.
export function paperRisk(state, cfg, now, excludePending = null) {
  const day = new Date(now).toISOString().slice(0, 10);
  const open = Object.values(state.open).reduce((s, p) => s + p.cost, 0);
  const pending = Object.entries(state.pending).reduce((s, [slug, p]) => s + (slug === excludePending ? 0 : p.budget), 0);
  const reserved = open + pending;
  const equity = state.bankroll + open;
  const losses = state.lossDay === day ? state.dailyLosses : 0;
  const floor = Math.max(cfg.bankroll, state.peakEquity) - cfg.maxDrawdown;
  const capacity = Math.max(0, Math.min(state.bankroll - pending,
    equity * cfg.maxExposure - reserved, cfg.maxExposureDollars - reserved,
    cfg.dailyLossLimit - losses - reserved, equity - floor - reserved));
  const overdue = Object.values(state.open).some((p) => now - p.periodEnd * 1000 > 600_000);
  const reason = state.drawdownHalted || equity - floor - reserved < cfg.minPaperSpend ? 'drawdown_halt'
    : overdue ? 'waiting_overdue_settlement'
    : cfg.dailyLossLimit - losses - reserved < cfg.minPaperSpend ? 'daily_loss_limit'
    : capacity < cfg.minPaperSpend ? 'exposure_limit' : null;
  return { reason, capacity: reason ? 0 : capacity, dailyLosses: losses, reserved, equity, floor };
}

export function paperBudget(state, choice, cfg, now, excludePending = null) {
  const risk = paperRisk(state, cfg, now, excludePending);
  if (!choice || choice.edge <= cfg.margin || choice.cost >= 1) return 0;
  const kelly = Math.max(0, choice.edge / (1 - choice.cost));
  return Math.max(0, Math.min(risk.capacity, state.bankroll * cfg.maxStake,
    cfg.maxStakeDollars, state.bankroll * kelly * cfg.kellyFraction));
}
