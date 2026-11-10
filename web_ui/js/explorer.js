// explorer.js — the threshold explorer: slide the alert threshold and watch the
// confusion matrix and precision/recall change. This is the part most dashboards
// hide inside a model: *where you cut the line* decides accuracy vs. churners caught.
import { renderPrChart, updatePrMarker } from './charts.js';
import { getThreshold, getDefaultThreshold, setThreshold, onThresholdChange, pct } from './state.js';

const STEP = 0.005; // must match THRESHOLDS in train_model.py

// rows are [threshold, tp, fp, tn, fn] at 0.005, 0.010, ... 0.995
export function rowAt(rows, t) {
  const i = Math.min(rows.length - 1, Math.max(0, Math.round(t / STEP) - 1));
  return rows[i];
}

export function metricsFrom([threshold, tp, fp, tn, fn]) {
  const n = tp + fp + tn + fn;
  const precision = tp + fp ? tp / (tp + fp) : 0;
  const recall = tp + fn ? tp / (tp + fn) : 0;
  return {
    threshold, tp, fp, tn, fn, precision, recall,
    accuracy: n ? (tp + tn) / n : 0,
    f1: precision + recall ? (2 * precision * recall) / (precision + recall) : 0,
  };
}

export function explain(m, modelName) {
  const churners = m.tp + m.fn;
  if (m.tp + m.fp === 0) return `At ${pct(m.threshold)} the ${modelName} model flags nobody, so it misses all ${churners} customers who left.`;
  return `At ${pct(m.threshold)} the ${modelName} model catches ${m.tp} of the ${churners} customers who left, `
    + `but also flags ${m.fp} loyal customers who would have stayed.`;
}

const $ = (id) => document.getElementById(id);
const ctx = { data: null, current: null, wired: false };

function currentPoint() {
  const m = metricsFrom(rowAt(ctx.data.models[ctx.current], getThreshold()));
  return { x: m.recall, y: m.precision, t: m.threshold };
}

function render() {
  const t = getThreshold();
  const m = metricsFrom(rowAt(ctx.data.models[ctx.current], t));
  $('thresholdSlider').value = t;
  $('thresholdValue').textContent = pct(t);
  $('cm-tn').textContent = m.tn; $('cm-fp').textContent = m.fp;
  $('cm-fn').textContent = m.fn; $('cm-tp').textContent = m.tp;
  $('mx-recall').textContent = pct(m.recall, 0);
  $('mx-precision').textContent = pct(m.precision, 0);
  $('mx-f1').textContent = m.f1.toFixed(2);
  $('mx-accuracy').textContent = pct(m.accuracy, 0);
  $('explorerSentence').textContent = explain(m, ctx.current);
  updatePrMarker({ x: m.recall, y: m.precision, t: m.threshold });
}

export function initExplorer(data, bestModel) {
  ctx.data = data;
  const names = Object.keys(data.models);
  ctx.current = names.includes(ctx.current) ? ctx.current : (names.includes(bestModel) ? bestModel : names[0]);

  const select = $('explorerModel');
  select.replaceChildren(...names.map((n) => {
    const o = document.createElement('option');
    o.value = n; o.textContent = n; o.selected = n === ctx.current;
    return o;
  }));

  $('explorerStatus').classList.add('hidden');
  $('explorerBody').classList.remove('hidden');
  renderPrChart($('prChart'), data.models[ctx.current], currentPoint());
  render();

  if (ctx.wired) return; // listeners must only be attached once, even if data is reloaded
  ctx.wired = true;
  $('thresholdSlider').addEventListener('input', (e) => setThreshold(parseFloat(e.target.value)));
  $('balancedBtn').addEventListener('click', () => setThreshold(getDefaultThreshold(), { fromUser: false }));
  select.addEventListener('change', () => {
    ctx.current = select.value;
    renderPrChart($('prChart'), ctx.data.models[ctx.current], currentPoint());
    render();
  });
  onThresholdChange(() => ctx.data && render());
}
