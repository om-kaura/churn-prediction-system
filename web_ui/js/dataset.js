// dataset.js — renders the training-data summary on the About page, so anyone can
// see exactly what the models learned from. Built with textContent (never innerHTML)
// because the content comes from an API response.
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};

export function renderDatasetSummary(s) {
  document.getElementById('datasetStatus').classList.add('hidden');
  document.getElementById('datasetBody').classList.remove('hidden');

  const stats = document.getElementById('datasetStats');
  stats.replaceChildren(...[
    ['Customers', s.rows.toLocaleString()],
    ['Columns', s.column_count],
    ['Churn rate', `${(s.churn_rate * 100).toFixed(1)}%`],
    ['Churned / stayed', `${s.churned.toLocaleString()} / ${s.retained.toLocaleString()}`],
  ].map(([label, value]) => {
    const box = el('div', 'stat-chip');
    box.append(el('div', 'text-lg font-bold display-num', String(value)), el('div', 'text-xs text-muted', label));
    return box;
  }));
  document.getElementById('datasetSource').textContent = `${s.source} \u2014 ${s.file}`;

  document.getElementById('datasetColumns').replaceChildren(...s.columns.map((c) => {
    const tr = el('tr');
    [c.name, c.kind, c.unique, c.missing, c.examples.join(', ')].forEach((v) => tr.append(el('td', '', String(v))));
    return tr;
  }));

  const keys = Object.keys(s.sample[0] || {});
  document.getElementById('datasetSampleHead').replaceChildren(el('tr', '', undefined));
  document.querySelector('#datasetSampleHead tr').append(...keys.map((k) => el('th', '', k)));
  document.getElementById('datasetSampleBody').replaceChildren(...s.sample.map((row) => {
    const tr = el('tr');
    keys.forEach((k) => tr.append(el('td', '', String(row[k]))));
    return tr;
  }));
}
