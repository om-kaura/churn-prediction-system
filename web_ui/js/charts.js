// charts.js — Chart.js rendering. Assumes the Chart.js UMD build is already
// loaded as a global (see index.html) — this module just consumes `window.Chart`.

const FEATURE_LABELS = {
  gender: 'Gender', SeniorCitizen: 'Senior Citizen', Partner: 'Partner', Dependents: 'Dependents',
  tenure: 'Tenure', PhoneService: 'Phone Service', MultipleLines: 'Multiple Lines',
  InternetService: 'Internet Service', OnlineSecurity: 'Online Security', OnlineBackup: 'Online Backup',
  DeviceProtection: 'Device Protection', TechSupport: 'Tech Support', StreamingTV: 'Streaming TV',
  StreamingMovies: 'Streaming Movies', Contract: 'Contract Type', PaperlessBilling: 'Paperless Billing',
  PaymentMethod: 'Payment Method', MonthlyCharges: 'Monthly Charges', TotalCharges: 'Total Charges',
};

let shapChart = null;
let rocChart = null;
let metricsChart = null;

export function renderShapChart(canvas, factors) {
  const sorted = [...factors].sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact));
  const labels = sorted.map((f) => FEATURE_LABELS[f.feature] || f.feature);
  const values = sorted.map((f) => f.impact);
  const colors = values.map((v) => (v >= 0 ? '#F2557B' : '#22C58B'));

  if (shapChart) shapChart.destroy();
  shapChart = new Chart(canvas.getContext('2d'), {
    type: 'bar',
    data: { labels, datasets: [{ data: values, backgroundColor: colors, borderRadius: 6 }] },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: (ctx) => (ctx.parsed.x >= 0 ? '+' : '') + ctx.parsed.x.toFixed(3) + ' SHAP value' } },
      },
      scales: {
        x: { grid: { color: 'rgba(30,27,58,0.06)' }, title: { display: true, text: 'Impact on prediction' } },
        y: { grid: { display: false } },
      },
    },
  });
  return shapChart;
}

export function renderRocChart(canvas, rocData) {
  const palette = { 'Logistic Regression': '#7C6FE8', 'Random Forest': '#22D3C5', XGBoost: '#F2557B' };
  const datasets = Object.entries(rocData).map(([name, d]) => ({
    label: `${name} (AUC ${d.auc.toFixed(3)})`,
    data: d.fpr.map((x, i) => ({ x, y: d.tpr[i] })),
    borderColor: palette[name] || '#7C6FE8',
    backgroundColor: 'transparent',
    borderWidth: 2,
    pointRadius: 0,
    tension: 0.15,
  }));
  datasets.push({
    label: 'Random guess',
    data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
    borderColor: 'rgba(30,27,58,0.25)',
    borderDash: [4, 4],
    borderWidth: 1,
    pointRadius: 0,
  });

  if (rocChart) rocChart.destroy();
  rocChart = new Chart(canvas.getContext('2d'), {
    type: 'line',
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      parsing: false,
      plugins: { legend: { position: 'bottom', labels: { boxWidth: 10, font: { size: 11 } } } },
      scales: {
        x: { type: 'linear', min: 0, max: 1, title: { display: true, text: 'False Positive Rate' }, grid: { color: 'rgba(30,27,58,0.06)' } },
        y: { type: 'linear', min: 0, max: 1, title: { display: true, text: 'True Positive Rate' }, grid: { color: 'rgba(30,27,58,0.06)' } },
      },
    },
  });
  return rocChart;
}

export function renderMetricsChart(canvas, models) {
  const labels = models.map((m) => m.name);
  const metric = (key) => models.map((m) => m.metrics[key]);

  if (metricsChart) metricsChart.destroy();
  metricsChart = new Chart(canvas.getContext('2d'), {
    type: 'bar',
    data: {
      labels,
      datasets: [
        { label: 'ROC-AUC', data: metric('ROC_AUC'), backgroundColor: '#7C6FE8', borderRadius: 5 },
        { label: 'F1', data: metric('F1'), backgroundColor: '#22D3C5', borderRadius: 5 },
        { label: 'Recall', data: metric('Recall'), backgroundColor: '#F2557B', borderRadius: 5 },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { position: 'bottom', labels: { boxWidth: 10, font: { size: 11 } } } },
      scales: {
        y: { min: 0, max: 1, grid: { color: 'rgba(30,27,58,0.06)' } },
        x: { grid: { display: false } },
      },
    },
  });
  return metricsChart;
}

let prChart = null;

// Precision-vs-recall at every threshold, with a marker for the selected one.
// rows: [threshold, tp, fp, tn, fn][]; current: {x: recall, y: precision, t}
export function renderPrChart(canvas, rows, current) {
  const pts = rows.map(([t, tp, fp, , fn]) => ({
    x: tp + fn ? tp / (tp + fn) : 0,
    y: tp + fp ? tp / (tp + fp) : 1,
    t,
  }));
  if (prChart) prChart.destroy();
  prChart = new Chart(canvas.getContext('2d'), {
    type: 'line',
    data: {
      datasets: [
        { label: 'All thresholds', data: pts, borderColor: '#7C6FE8', borderWidth: 2, pointRadius: 0, tension: 0.1 },
        { label: 'Selected', data: [current], showLine: false, pointRadius: 7, pointBackgroundColor: '#F2557B', pointBorderColor: '#fff', pointBorderWidth: 2 },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      parsing: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (c) => `threshold ${(c.raw.t * 100).toFixed(1)}%: catches ${(c.raw.x * 100).toFixed(0)}% of churners, ${(c.raw.y * 100).toFixed(0)}% of flags correct`,
          },
        },
      },
      scales: {
        x: { type: 'linear', min: 0, max: 1, title: { display: true, text: 'Recall — share of churners caught' }, grid: { color: 'rgba(30,27,58,0.06)' } },
        y: { type: 'linear', min: 0, max: 1, title: { display: true, text: 'Precision — share of flags that are right' }, grid: { color: 'rgba(30,27,58,0.06)' } },
      },
    },
  });
  return prChart;
}

export function updatePrMarker(point) {
  if (!prChart) return;
  prChart.data.datasets[1].data = [point];
  prChart.update('none');
}

export function clearShapChart() {
  if (shapChart) { shapChart.destroy(); shapChart = null; }
}
