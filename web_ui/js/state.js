// state.js — the one piece of state shared across pages: the alert threshold
// (flag a customer when their estimated churn probability is at least this).
const listeners = new Set();
let threshold = 0.265;
let defaultThreshold = 0.265;
let touched = false; // has the user moved it themselves?

export const getThreshold = () => threshold;
export const getDefaultThreshold = () => defaultThreshold;
export const onThresholdChange = (fn) => listeners.add(fn);

export function setThreshold(v, { fromUser = true } = {}) {
  threshold = Math.round(v * 1000) / 1000;
  if (fromUser) touched = true;
  listeners.forEach((fn) => fn(threshold));
}

// Called once the API tells us its default; never overrides a user's choice.
export function initDefaultThreshold(v) {
  defaultThreshold = v;
  if (!touched) setThreshold(v, { fromUser: false });
}

export const pct = (x, digits = 1) => `${(x * 100).toFixed(digits)}%`;
