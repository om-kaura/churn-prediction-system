// api.js — every network call the frontend makes, in one place.
// The API address is derived from wherever the page itself was loaded from, on
// port 8000: open the dashboard at localhost and it talks to localhost:8000; open
// it at http://<server-ip>:8080 and it talks to <server-ip>:8000. This matters
// because this JavaScript runs in the *visitor's* browser, so a hardcoded
// "localhost" would point at the visitor's own computer once deployed.
// To point somewhere else (e.g. a domain), set window.CHURN_API_BASE before
// this module loads.
export const API_BASE =
  window.CHURN_API_BASE || `${window.location.protocol}//${window.location.hostname}:8000`;

async function asJson(res) {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const detail = Array.isArray(body.detail) ? body.detail.map((d) => d.msg).join('; ') : body.detail;
    const err = new Error(detail || `Request failed (${res.status})`);
    err.status = res.status;
    throw err;
  }
  return res.json();
}

const get = (path) => fetch(`${API_BASE}${path}`, { cache: 'no-store' }).then(asJson);

export const getHealth = () => get('/health');
export const getModels = () => get('/models');
export const getRocCurve = () => get('/models/roc-curve');
export const getThresholds = () => get('/models/thresholds');
export const getDatasetSummary = () => get('/dataset/summary');

export function predictCompare(payload, threshold) {
  const qs = threshold == null ? '' : `?threshold=${encodeURIComponent(threshold)}`;
  return fetch(`${API_BASE}/predict/compare${qs}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  }).then(asJson);
}
