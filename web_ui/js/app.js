import { API_BASE, getHealth, getModels, getRocCurve, getThresholds, getDatasetSummary, predictCompare } from './api.js';
import { renderShapChart, clearShapChart, renderRocChart, renderMetricsChart } from './charts.js';
import { createWizard } from './wizard.js';
import { initExplorer } from './explorer.js';
import { renderDatasetSummary } from './dataset.js';
import { getThreshold, initDefaultThreshold, onThresholdChange, pct } from './state.js';

const $ = (id) => document.getElementById(id);

// ---------- Pages ----------
const PAGES = ['predict', 'graphs', 'about'];

function switchPage(pageId) {
  PAGES.forEach((id) => {
    $(`page-${id}`).classList.toggle('hidden', id !== pageId);
    $(`tab-${id}`).classList.toggle('tab-active', id === pageId);
    $(`tab-${id}`).setAttribute('aria-selected', String(id === pageId));
  });
  // Charts drawn while their page was hidden were measured at 0x0; make sure they fit now.
  if (pageId === 'graphs' && window.Chart) {
    ['metricsChart', 'rocChart', 'prChart'].forEach((id) => { const c = window.Chart.getChart($(id)); if (c) c.resize(); });
  }
}

// ---------- Health ----------
async function checkHealth() {
  const dot = $('statusDot');
  const label = $('statusLabel');
  try {
    const health = await getHealth();
    if (health.default_threshold != null) initDefaultThreshold(health.default_threshold);
    dot.className = 'status-dot online';
    label.textContent = 'API connected';
    return true;
  } catch {
    dot.className = 'status-dot offline';
    label.textContent = 'API offline';
    return false;
  }
}

// The API can take a couple of seconds to load its models and SHAP explainers,
// and the static dashboard server answers instantly, so the first fetch after a
// page load can beat the API to the punch. Retry a few times before giving up.
async function withRetry(fn, { attempts = 5, delayMs = 1500 } = {}) {
  let lastErr;
  for (let i = 0; i < attempts; i++) {
    try { return await fn(); } catch (err) {
      lastErr = err;
      if (i < attempts - 1) await new Promise((r) => setTimeout(r, delayMs));
    }
  }
  throw lastErr;
}

// ---------- Predict page ----------
function showResultError(message) {
  $('resultEmpty').classList.add('hidden');
  $('resultContent').classList.add('hidden');
  const err = $('resultError');
  err.classList.remove('hidden');
  err.textContent = /Failed to fetch|NetworkError|Load failed/.test(message)
    ? `Can\u2019t reach the prediction API at ${API_BASE} \u2014 make sure the backend is running and that port 8000 is open, then reload.`
    : message;
}

function resetResultPanels() {
  $('resultError').classList.add('hidden');
  $('resultContent').classList.add('hidden');
  $('resultEmpty').classList.remove('hidden');
  // Hide the *wrapper*, never the canvas: Chart.js writes an inline `display: block` onto
  // the canvas, which beats any CSS class (Tailwind's `hidden` included), so hiding the
  // canvas directly silently fails once a chart exists.
  clearShapChart();
  $('shapWrap').classList.add('hidden');
  $('shapEmpty').classList.remove('hidden');
  $('compareList').textContent = 'Run a prediction to compare models.';
}

function renderPrediction(best, thresholdUsed) {
  $('resultEmpty').classList.add('hidden');
  $('resultError').classList.add('hidden');
  $('resultContent').classList.remove('hidden');

  const flagged = best.churn_prediction === 'Yes';

  const badge = $('resultBadge');
  badge.textContent = flagged ? 'Flagged as at-risk' : 'Not flagged';
  badge.className = 'inline-block px-3 py-1 rounded-full text-xs font-semibold ' + (flagged ? 'badge-high' : 'badge-low');

  const prob = $('resultProb');
  prob.textContent = pct(best.churn_probability);
  prob.style.color = flagged ? 'var(--risk-high)' : 'var(--risk-low)';

  $('resultSummary').textContent = flagged
    ? `About ${pct(best.churn_probability, 0)} of customers like this one churn \u2014 above your ${pct(thresholdUsed)} alert threshold.`
    : `About ${pct(best.churn_probability, 0)} of customers like this one churn \u2014 below your ${pct(thresholdUsed)} alert threshold.`;
  $('resultMeta').textContent = `Scored by ${best.model}`;

  // Show the container first so Chart.js measures a real size, then draw.
  $('shapEmpty').classList.add('hidden');
  $('shapWrap').classList.remove('hidden');
  renderShapChart($('shapChart'), best.top_risk_factors);
}

function renderComparison(compareData) {
  const wrap = $('compareList');
  wrap.replaceChildren(...compareData.results.map((r) => {
    const flagged = r.churn_prediction === 'Yes';
    const color = flagged ? 'var(--risk-high)' : 'var(--risk-low)';

    const row = document.createElement('div');
    row.className = 'flex items-center justify-between py-2 border-b border-[rgba(30,27,58,0.06)] last:border-0';

    const left = document.createElement('div');
    left.className = 'flex items-center gap-2';
    const dot = document.createElement('span');
    dot.className = `w-1.5 h-1.5 rounded-full ${flagged ? 'dot-high' : 'dot-low'}`;
    const name = document.createElement('span');
    name.className = 'text-sm font-medium';
    name.textContent = r.model;
    left.append(dot, name);
    if (r.is_best) {
      const tag = document.createElement('span');
      tag.className = 'text-[10px] px-1.5 py-0.5 rounded bg-[var(--accent)] text-white';
      tag.textContent = 'BEST';
      left.append(tag);
    }

    const value = document.createElement('span');
    value.className = 'text-sm tabular-nums font-semibold';
    value.style.color = color;
    value.textContent = pct(r.churn_probability);

    row.append(left, value);
    return row;
  }));
}

async function handleComplete(payload) {
  const nextBtn = $('nextBtn');
  nextBtn.disabled = true;
  const original = nextBtn.textContent;
  nextBtn.textContent = 'Predicting\u2026';
  try {
    // One request scores all three models; the headline result is the best one's
    // entry. (Used to be two requests, which logged the best model twice and
    // spent twice the rate-limit budget per click.)
    const compare = await predictCompare(payload, getThreshold());
    const best = compare.results.find((r) => r.is_best) || compare.results[0];
    renderPrediction(best, compare.threshold_used);
    renderComparison(compare);
  } catch (err) {
    showResultError(err.message);
  } finally {
    nextBtn.disabled = false;
    nextBtn.textContent = original;
    checkHealth();
  }
}

// ---------- Model Performance page ----------
function setPerfStatus(text, isError) {
  const note = $('bestModelNote');
  note.textContent = text;
  note.classList.toggle('text-[var(--risk-high)]', !!isError);
}

async function loadMetricsChart() {
  const models = await withRetry(getModels);
  renderMetricsChart($('metricsChart'), models.models);
  return models;
}

async function loadRocChart() {
  renderRocChart($('rocChart'), await withRetry(getRocCurve));
}

async function loadExplorer(modelsPromise) {
  const [data, models] = await Promise.all([withRetry(getThresholds), modelsPromise.catch(() => null)]);
  initExplorer(data, models && models.best_model);
}

async function loadModelPerformance() {
  setPerfStatus('Loading\u2026', false);
  $('explorerStatus').textContent = 'Loading\u2026';
  // Each part loads (and retries) independently, so a hiccup in one doesn't block the others.
  const modelsPromise = loadMetricsChart();
  const results = await Promise.allSettled([modelsPromise, loadRocChart(), loadExplorer(modelsPromise)]);
  const failed = results.filter((r) => r.status === 'rejected').length;

  if (failed === 0) {
    setPerfStatus(`Best model: ${results[0].value.best_model} \u2014 chosen by 5-fold cross-validated ROC-AUC, not by test score.`, false);
  } else if (failed === results.length) {
    setPerfStatus('Model metrics unavailable \u2014 is the API running? Click to retry.', true);
    $('explorerStatus').textContent = 'Unavailable \u2014 click to retry.';
  } else {
    setPerfStatus('Partially loaded \u2014 some charts failed. Click to retry.', true);
    if (results[2].status === 'rejected') $('explorerStatus').textContent = 'Unavailable \u2014 click to retry.';
  }
}

// ---------- About page ----------
async function loadDataset() {
  const status = $('datasetStatus');
  status.textContent = 'Loading\u2026';
  status.classList.remove('hidden');
  try {
    renderDatasetSummary(await withRetry(getDatasetSummary));
  } catch {
    status.textContent = 'Dataset summary unavailable \u2014 is the API running? Click to retry.';
  }
}

// ---------- Wire everything up ----------
document.addEventListener('DOMContentLoaded', () => {
  $('docsLink').href = `${API_BASE}/docs`;
  $('downloadLog').href = `${API_BASE}/predictions/export.csv`;

  checkHealth();
  loadModelPerformance();
  loadDataset();
  $('bestModelNote').addEventListener('click', loadModelPerformance);
  $('explorerStatus').addEventListener('click', loadModelPerformance);
  $('datasetStatus').addEventListener('click', loadDataset);

  PAGES.forEach((id) => $(`tab-${id}`).addEventListener('click', () => switchPage(id)));
  switchPage('predict');
  $('goThreshold').addEventListener('click', () => switchPage('graphs'));

  $('thresholdNote').textContent = pct(getThreshold());
  onThresholdChange((t) => { $('thresholdNote').textContent = pct(t); });

  const wizard = createWizard($('wizard'), { onComplete: handleComplete });
  $('resetBtn').addEventListener('click', () => {
    wizard.reset();
    resetResultPanels();
  });
});

// Exported so the show/hide behaviour can be tested without a browser.
export { renderPrediction, resetResultPanels };
