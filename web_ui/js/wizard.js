// wizard.js — renders one step of questions at a time instead of a 19-field wall.

const STEPS = [
  {
    title: 'Demographics',
    fields: [
      { id: 'gender', label: 'Gender', type: 'select', options: ['Female', 'Male'], default: 'Female' },
      { id: 'SeniorCitizen', label: 'Senior citizen', type: 'toggle', default: false },
      { id: 'Partner', label: 'Has a partner', type: 'toggle', default: true },
      { id: 'Dependents', label: 'Has dependents', type: 'toggle', default: false },
    ],
  },
  {
    title: 'Phone & Internet',
    fields: [
      { id: 'PhoneService', label: 'Phone service', type: 'toggle', default: true },
      { id: 'MultipleLines', label: 'Multiple lines', type: 'select', options: ['No', 'Yes', 'No phone service'], default: 'No' },
      { id: 'InternetService', label: 'Internet service', type: 'select', options: ['DSL', 'Fiber optic', 'No'], default: 'DSL' },
    ],
  },
  {
    title: 'Internet Add-ons',
    fields: [
      { id: 'OnlineSecurity', label: 'Online security', type: 'select', options: ['No', 'Yes', 'No internet service'], default: 'No' },
      { id: 'OnlineBackup', label: 'Online backup', type: 'select', options: ['No', 'Yes', 'No internet service'], default: 'Yes' },
      { id: 'DeviceProtection', label: 'Device protection', type: 'select', options: ['No', 'Yes', 'No internet service'], default: 'No' },
      { id: 'TechSupport', label: 'Tech support', type: 'select', options: ['No', 'Yes', 'No internet service'], default: 'No' },
    ],
  },
  {
    title: 'Streaming',
    fields: [
      { id: 'StreamingTV', label: 'Streaming TV', type: 'select', options: ['No', 'Yes', 'No internet service'], default: 'No' },
      { id: 'StreamingMovies', label: 'Streaming movies', type: 'select', options: ['No', 'Yes', 'No internet service'], default: 'No' },
    ],
  },
  {
    title: 'Contract & Billing',
    fields: [
      { id: 'Contract', label: 'Contract', type: 'select', options: ['Month-to-month', 'One year', 'Two year'], default: 'Month-to-month' },
      { id: 'PaperlessBilling', label: 'Paperless billing', type: 'toggle', default: true },
      { id: 'PaymentMethod', label: 'Payment method', type: 'select', options: ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], default: 'Electronic check' },
    ],
  },
  {
    title: 'Tenure & Charges',
    fields: [
      { id: 'tenure', label: 'Tenure (months)', type: 'number', min: 0, max: 72, default: 12 },
      { id: 'MonthlyCharges', label: 'Monthly charges ($)', type: 'number', min: 0, step: 0.01, default: 70.35 },
      { id: 'TotalCharges', label: 'Total charges ($)', type: 'number', min: 0, step: 0.01, default: 844.20 },
    ],
  },
];

const INTERNET_ADDONS = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'];

export function createWizard(container, { onComplete }) {
  let state = {};
  const resetState = () => { STEPS.forEach((step) => step.fields.forEach((f) => { state[f.id] = f.default; })); };
  resetState();
  let stepIndex = 0;
  let totalChargesTouched = false;

  function fieldEl(field) {
    const wrap = document.createElement('div');
    wrap.className = field.type === 'toggle'
      ? 'flex items-center justify-between py-2.5'
      : 'flex flex-col gap-1.5 py-2';

    const label = document.createElement('label');
    label.textContent = field.label;
    label.className = 'text-sm font-medium text-[var(--ink)]';
    label.htmlFor = field.id;

    let control;
    if (field.type === 'toggle') {
      control = document.createElement('label');
      control.className = 'toggle';
      control.innerHTML = `<input type="checkbox" id="${field.id}"><span class="toggle-track"><span class="toggle-thumb"></span></span>`;
      const input = control.querySelector('input');
      input.checked = !!state[field.id];
      input.addEventListener('change', () => {
        state[field.id] = input.checked;
        if (field.id === 'PhoneService') applyPhoneDependency();
      });
    } else if (field.type === 'select') {
      control = document.createElement('select');
      control.id = field.id;
      control.className = 'glass-input w-full px-3 py-2 text-sm';
      field.options.forEach((opt) => {
        const o = document.createElement('option');
        o.value = opt; o.textContent = opt;
        if (opt === state[field.id]) o.selected = true;
        control.appendChild(o);
      });
      control.addEventListener('change', () => {
        state[field.id] = control.value;
        if (field.id === 'InternetService') applyInternetDependency();
      });
    } else {
      control = document.createElement('input');
      control.type = 'number';
      control.id = field.id;
      control.className = 'glass-input w-full px-3 py-2 text-sm tabular-nums';
      if (field.min !== undefined) control.min = field.min;
      if (field.max !== undefined) control.max = field.max;
      if (field.step !== undefined) control.step = field.step;
      control.value = state[field.id];
      control.addEventListener('input', () => {
        state[field.id] = parseFloat(control.value) || 0;
        if (field.id === 'TotalCharges') totalChargesTouched = true;
        if ((field.id === 'tenure' || field.id === 'MonthlyCharges') && !totalChargesTouched) {
          state.TotalCharges = Math.round(state.tenure * state.MonthlyCharges * 100) / 100;
          const tc = container.querySelector('#TotalCharges');
          if (tc) tc.value = state.TotalCharges;
        }
      });
    }

    if (field.type === 'toggle') { wrap.append(label, control); } else { wrap.append(label, control); }
    return wrap;
  }

  function applyPhoneDependency() {
    const sel = container.querySelector('#MultipleLines');
    if (!sel) return;
    sel.disabled = !state.PhoneService;
    sel.value = state.PhoneService ? (state.MultipleLines === 'No phone service' ? 'No' : state.MultipleLines) : 'No phone service';
    state.MultipleLines = sel.value;
  }

  function applyInternetDependency() {
    const disabled = state.InternetService === 'No';
    INTERNET_ADDONS.forEach((id) => {
      const sel = container.querySelector(`#${id}`);
      if (!sel) return; // only present when its step is on screen
      sel.disabled = disabled;
      sel.value = disabled ? 'No internet service' : (state[id] === 'No internet service' ? 'No' : state[id]);
      state[id] = sel.value;
    });
  }

  function renderProgress() {
    const track = container.querySelector('#stepTrack');
    track.innerHTML = '';
    for (let i = 0; i <= STEPS.length; i++) {
      const dot = document.createElement('div');
      dot.className = 'step-dot' + (i < stepIndex ? ' done' : i === stepIndex ? ' active' : '');
      track.appendChild(dot);
    }
    container.querySelector('#stepLabel').textContent =
      stepIndex < STEPS.length ? `Step ${stepIndex + 1} of ${STEPS.length + 1} — ${STEPS[stepIndex].title}` : `Step ${STEPS.length + 1} of ${STEPS.length + 1} — Review`;
  }

  function renderStep() {
    const body = container.querySelector('#wizardBody');
    body.innerHTML = '';

    if (stepIndex < STEPS.length) {
      const step = STEPS[stepIndex];
      step.fields.forEach((f) => body.appendChild(fieldEl(f)));
      if (step.fields.some((f) => f.id === 'MultipleLines')) applyPhoneDependency();
      if (step.fields.some((f) => INTERNET_ADDONS.includes(f.id))) applyInternetDependency();
    } else {
      const summary = document.createElement('div');
      summary.className = 'text-sm space-y-1 max-h-64 overflow-y-auto pr-1';
      STEPS.flatMap((s) => s.fields).forEach((f) => {
        const row = document.createElement('div');
        row.className = 'flex justify-between border-b border-[rgba(30,27,58,0.06)] py-1.5';
        const val = f.type === 'toggle' ? (state[f.id] ? 'Yes' : 'No') : state[f.id];
        row.innerHTML = `<span class="text-[var(--ink-muted)]">${f.label}</span><span class="font-medium">${val}</span>`;
        summary.appendChild(row);
      });
      body.appendChild(summary);
    }

    container.querySelector('#backBtn').disabled = stepIndex === 0;
    const nextBtn = container.querySelector('#nextBtn');
    nextBtn.textContent = stepIndex === STEPS.length ? 'Predict Churn' : 'Next';
    renderProgress();
  }

  container.querySelector('#backBtn').addEventListener('click', () => {
    if (stepIndex > 0) { stepIndex -= 1; renderStep(); }
  });
  container.querySelector('#nextBtn').addEventListener('click', () => {
    if (stepIndex < STEPS.length) { stepIndex += 1; renderStep(); }
    else onComplete(buildPayload());
  });

  function buildPayload() {
    return {
      gender: state.gender,
      SeniorCitizen: state.SeniorCitizen ? 1 : 0,
      Partner: state.Partner ? 'Yes' : 'No',
      Dependents: state.Dependents ? 'Yes' : 'No',
      tenure: Math.round(state.tenure),
      PhoneService: state.PhoneService ? 'Yes' : 'No',
      MultipleLines: state.MultipleLines,
      InternetService: state.InternetService,
      OnlineSecurity: state.OnlineSecurity,
      OnlineBackup: state.OnlineBackup,
      DeviceProtection: state.DeviceProtection,
      TechSupport: state.TechSupport,
      StreamingTV: state.StreamingTV,
      StreamingMovies: state.StreamingMovies,
      Contract: state.Contract,
      PaperlessBilling: state.PaperlessBilling ? 'Yes' : 'No',
      PaymentMethod: state.PaymentMethod,
      MonthlyCharges: state.MonthlyCharges,
      TotalCharges: state.TotalCharges,
    };
  }

  renderStep();
  return {
    reset() {
      resetState();
      stepIndex = 0;
      totalChargesTouched = false;
      renderStep();
    },
  };
}
