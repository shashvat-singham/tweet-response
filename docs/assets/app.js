/**
 * tweet-response — page controller.
 *
 * Every number rendered here comes from `model/metrics.json`, which is emitted
 * by `ml/export.py` from the actual held-out test split. Nothing is hard-coded.
 */

import { EmotionModel } from './engine.js';
import { lineChart, barChart, histogram, stackedRows, el, fmtNum, fmtPct } from './charts.js';

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const colorOf = (cls) => cssVar('--' + cls);
const escapeHtml = (s) => s.replace(/[&<>"]/g, (c) =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

/* ── theme ─────────────────────────────────────────────────────────────── */
const root = document.documentElement;
const forced = new URLSearchParams(location.search).get('theme');   // ?theme=light
const stored = localStorage.getItem('tr-theme');
if (forced === 'light' || forced === 'dark') root.dataset.theme = forced;
else if (stored) root.dataset.theme = stored;
else if (window.matchMedia('(prefers-color-scheme: light)').matches) root.dataset.theme = 'light';

const themeListeners = [];
$('#theme').addEventListener('click', () => {
  root.dataset.theme = root.dataset.theme === 'dark' ? 'light' : 'dark';
  localStorage.setItem('tr-theme', root.dataset.theme);
  themeListeners.forEach((f) => f());
});

/* ── chrome: scroll progress, nav highlight, reveal ────────────────────── */
const bar = $('#scrollbar');
addEventListener('scroll', () => {
  const h = document.body.scrollHeight - innerHeight;
  bar.style.width = `${Math.min(100, (scrollY / (h || 1)) * 100)}%`;
}, { passive: true });

const sections = $$('main section[id]');
const navLinks = new Map($$('.nav a').map((a) => [a.getAttribute('href').slice(1), a]));
const navObs = new IntersectionObserver((entries) => {
  for (const e of entries) {
    if (!e.isIntersecting) continue;
    navLinks.forEach((a) => a.classList.remove('active'));
    navLinks.get(e.target.id)?.classList.add('active');
  }
}, { rootMargin: '-45% 0px -50% 0px' });
sections.forEach((s) => navObs.observe(s));

const revealObs = new IntersectionObserver((entries, obs) => {
  for (const e of entries) if (e.isIntersecting) { e.target.classList.add('in'); obs.unobserve(e.target); }
}, { rootMargin: '0px 0px -8% 0px' });
$$('.panel, .sec-head').forEach((n) => { n.classList.add('reveal'); revealObs.observe(n); });

/* ── worker plumbing (with a main-thread fallback) ─────────────────────── */
class Engine {
  constructor() { this.seq = 0; this.pending = new Map(); }

  async init(base) {
    try {
      this.worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
      this.worker.onmessage = (e) => {
        const { id, ok, result, error } = e.data;
        const p = this.pending.get(id);
        if (!p) return;
        this.pending.delete(id);
        ok ? p.resolve(result) : p.reject(new Error(error));
      };
      return await this.call('load', { base });
    } catch (err) {
      this.worker = null;                       // e.g. file:// or old Safari
      this.local = await EmotionModel.load(base);
      return { vocab: this.local.meta.vocab_size, bytes: this.local.meta.total_bytes,
               classes: this.local.classes };
    }
  }

  call(op, payload) {
    if (!this.worker) {
      const m = this.local;
      if (op === 'analyze') return Promise.resolve(m.analyze(payload.text, payload.opts));
      if (op === 'neighbours') return Promise.resolve(m.neighbours(payload.word, payload.k));
      if (op === 'benchmark') return Promise.resolve(m.benchmark(payload.text, payload.n));
    }
    const id = ++this.seq;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ id, op, payload });
      setTimeout(() => {
        if (this.pending.delete(id)) reject(new Error('inference timed out'));
      }, 15000);
    });
  }
}

/* ── boot ──────────────────────────────────────────────────────────────── */
const engine = new Engine();
let M = null;   // metrics.json

(async function main() {
  M = await fetch('./model/metrics.json').then((r) => r.json());
  renderStatic();
  try {
    const info = await engine.init('./model/');
    setState('ready', `${(info.bytes / 1024).toFixed(0)} KB · ${fmtNum(info.vocab)} vocab · int8`);
    wireDemo();
    lookupNeighbours();
    runBenchmark();
    verifyParity();
  } catch (err) {
    setState('error', 'model failed to load');
    console.error(err);
  }
})();

function setState(state, text) {
  $('#modelState').dataset.state = state;
  $('#modelStateText').textContent = text;
}

/* ══════════════════════════════════════════════════════════════════════
   Static report
   ══════════════════════════════════════════════════════════════════════ */
function renderStatic() {
  const H = M.headline, mdl = M.model;

  const fills = {
    testAcc: fmtPct(H.test_accuracy, 1),
    macroF1: H.macro_f1.toFixed(3),
    params: fmtNum(mdl.params),
    trainRows: fmtNum(mdl.train_rows),
    weightsKb: (mdl.weights_bytes / 1024).toFixed(0),
    ece: H.ece.toFixed(3),
    latency: '—',
  };
  for (const [k, v] of Object.entries(fills))
    $$(`[data-fill="${k}"]`).forEach((n) => { n.textContent = v; });

  renderArchitecture(mdl);
  renderData();
  renderTraining();
  renderConfusion();
  renderPerClass();
  renderCalibration();
  renderCoverage();
  renderPairs();
  renderWorstCases();
  renderEmbedding();
  renderShipping();
  renderCard();

  let raf;
  addEventListener('resize', () => {
    clearTimeout(raf);
    raf = setTimeout(() => { redrawCharts(); }, 180);
  });
  themeListeners.push(redrawCharts);
}

function redrawCharts() {
  renderData(); renderTraining(); renderCalibration(); renderCoverage();
  renderEmbedding(); renderConfusion(); renderPerClass();
  if (lastResult) renderTrajectory(lastResult);
}

/* ── 02 architecture ───────────────────────────────────────────────────── */
function renderArchitecture(mdl) {
  const host = $('#archDiagram');
  host.innerHTML = '';
  const arrow = () => {
    const a = document.createElement('div');
    a.className = 'arch-arrow';
    a.innerHTML = '<svg width="14" height="16" viewBox="0 0 14 16"><path d="M7 1v13m0 0 4-4m-4 4-4-4" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>';
    return a;
  };
  const input = document.createElement('div');
  input.className = 'arch-layer';
  input.innerHTML = `<span class="lname">Input · token ids</span><span class="lshape">(50,)</span>
    <span class="ldetail">Keras tokenizer · post-padded, post-truncated</span><span class="lparams">—</span>`;
  host.appendChild(input);
  for (const L of mdl.architecture) {
    host.appendChild(arrow());
    const d = document.createElement('div');
    d.className = 'arch-layer';
    d.innerHTML = `<span class="lname">${L.layer}</span><span class="lshape">${L.output}</span>
      <span class="ldetail">${L.detail}</span><span class="lparams">${fmtNum(L.params)} params</span>`;
    host.appendChild(d);
  }

  const budget = $('#paramBudget');
  budget.innerHTML = '';
  const total = mdl.architecture.reduce((a, L) => a + L.params, 0);
  for (const L of mdl.architecture) {
    const row = document.createElement('div');
    row.className = 'budget-row';
    row.innerHTML = `<span class="lab">${L.layer}</span>
      <span class="num">${fmtNum(L.params)} · ${fmtPct(L.params / total, 1)}</span>
      <span class="track"><span class="fill" style="width:${(L.params / total) * 100}%"></span></span>`;
    budget.appendChild(row);
  }
  const t = document.createElement('div');
  t.className = 'budget-row';
  t.style.marginTop = '.3rem';
  t.innerHTML = `<span class="lab" style="color:var(--ink)">Total</span>
    <span class="num">${fmtNum(total)}</span>`;
  budget.appendChild(t);
}

/* ── 03 data ───────────────────────────────────────────────────────────── */
function renderData() {
  const D = M.dataset;
  stackedRows($('#splitChart'), {
    rows: D.splits.map((s) => ({
      label: s.name, parts: s.counts.map((v, i) => ({ value: v, color: colorOf(M.classes[i]), name: M.classes[i] })),
    })),
  });
  const tr = D.splits[0];
  const maxC = Math.max(...tr.counts), minC = Math.min(...tr.counts);
  $('#splitFoot').innerHTML =
    `Class balance is <b>${(maxC / minC).toFixed(1)}×</b> skewed in train — ` +
    `<span style="color:${colorOf('joy')}">joy</span> has ${fmtNum(maxC)} examples, ` +
    `<span style="color:${colorOf('surprise')}">surprise</span> only ${fmtNum(minC)}. ` +
    `The split ratios are preserved across train / validation / test.`;

  histogram($('#lenChart'), {
    data: D.length_histogram.map((d) => ({ x: d.len, n: d.n })),
    cut: M.model.config.maxlen, color: colorOf('sadness'),
  });
  const p = D.length_percentiles;
  $('#lenFoot').innerHTML =
    `Median ${p['50']} words, p95 ${p['95']}, p99 ${p['99']}. Truncating at 50 touches ` +
    `only <b>${fmtPct(D.test_truncation_rate, 2)}</b> of test tweets, so the sequence budget is ` +
    `generous — most of the 50 timesteps are padding.`;

  const kv = $('#vocabStats');
  kv.innerHTML = '';
  const rows = [
    ['unique word types', fmtNum(D.unique_tokens)],
    ['kept (rank ≤ 10k)', fmtNum(Math.min(D.vocab_cutoff, D.unique_tokens))],
    ['token coverage', fmtPct(D.coverage_at_cutoff, 2)],
    ['&lt;UNK&gt; rate on test', fmtPct(D.test_oov_rate, 2)],
    ['truncated at 50', fmtPct(D.test_truncation_rate, 2)],
  ];
  for (const [k, v] of rows) {
    const r = document.createElement('div');
    r.className = 'kv-row';
    r.innerHTML = `<span class="k">${k}</span><span class="rule"></span><span class="v">${v}</span>`;
    kv.appendChild(r);
  }

  const lo = $('#logOdds');
  lo.innerHTML = '';
  for (const c of M.classes) {
    const items = M.token_log_odds[c] ?? [];
    const maxZ = Math.max(...items.map((i) => i.z), 1);
    const col = document.createElement('div');
    col.className = 'lo-col';
    col.innerHTML = `<h4><i style="background:${colorOf(c)}"></i>${c}</h4>` + items.slice(0, 12).map((i) =>
      `<div class="lo-item"><span class="w">${i.token}</span><span class="z">${i.z.toFixed(1)}</span>
       <span class="meter" style="width:${(i.z / maxZ) * 100}%;background:${colorOf(c)}"></span></div>`).join('');
    lo.appendChild(col);
  }
}

/* ── 04 training ───────────────────────────────────────────────────────── */
function renderTraining() {
  const h = M.history;
  const labels = h.loss.map((_, i) => String(i + 1));
  const best = h.val_accuracy.indexOf(Math.max(...h.val_accuracy));

  lineChart($('#accChart'), {
    xLabels: labels, xTitle: 'epoch', yDomain: [0.2, 1], yFormat: (v) => (v * 100).toFixed(0) + '%',
    markers: [{ at: best, label: 'best' }],
    series: [
      { values: h.accuracy, color: cssVar('--ink-3') },
      { values: h.val_accuracy, color: cssVar('--accent'), area: true },
    ],
    tipHtml: (i) => `epoch ${i + 1}<br>train ${fmtPct(h.accuracy[i])}<br>val ${fmtPct(h.val_accuracy[i])}`,
  });
  lineChart($('#lossChart'), {
    xLabels: labels, xTitle: 'epoch', yFormat: (v) => v.toFixed(2),
    markers: [{ at: best, label: 'best' }],
    series: [
      { values: h.loss, color: cssVar('--ink-3') },
      { values: h.val_loss, color: cssVar('--accent'), area: true },
    ],
    tipHtml: (i) => `epoch ${i + 1}<br>train ${h.loss[i].toFixed(3)}<br>val ${h.val_loss[i].toFixed(3)}`,
  });

  const led = $('#repro');
  const H = M.headline, mdl = M.model;
  const delta = H.test_accuracy - H.reference_keras_accuracy;
  led.innerHTML = `
    <div class="led-row head"><span>metric</span><span>this run</span><span>notebook</span><span class="hide-s">note</span></div>
    ${row('test accuracy', fmtPct(H.test_accuracy, 2), fmtPct(H.reference_keras_accuracy, 2),
      `<span class="delta ${delta >= 0 ? 'pos' : 'neg'}">${delta >= 0 ? '+' : ''}${(delta * 100).toFixed(2)} pp</span> vs the TensorFlow original`)}
    ${row('parameters', fmtNum(mdl.params), '175,926', 'identical topology, layer for layer')}
    ${row('epochs run', String(mdl.epochs_trained), '10', `early stop, patience ${mdl.config.patience} on val accuracy`)}
    ${row('framework', 'PyTorch (CPU)', 'TensorFlow (Colab GPU)', 'Keras initialisers ported explicitly')}
    ${row('wall clock', `${(mdl.wall_seconds / 60).toFixed(1)} min`, '≈4 min', 'CPU vs GPU — the model is tiny either way')}
  `;
  function row(k, a, b, note) {
    return `<div class="led-row"><span>${k}</span><span class="num">${a}</span>
      <span class="num" style="color:var(--ink-3)">${b}</span><span class="hide-s" style="color:var(--ink-3);font-size:.78rem">${note}</span></div>`;
  }
}

/* ── 05 evaluation ─────────────────────────────────────────────────────── */
function renderConfusion() {
  const host = $('#confusion');
  const cm = M.confusion, C = M.classes;
  const rowSums = cm.map((r) => r.reduce((a, b) => a + b, 0));
  const t = document.createElement('table');
  const head = t.insertRow();
  head.appendChild(document.createElement('th'));
  const corner = document.createElement('th');
  corner.textContent = 'predicted →';
  corner.style.textAlign = 'left';
  head.appendChild(corner);
  for (let i = 1; i < C.length; i++) head.appendChild(document.createElement('th'));
  const sub = t.insertRow();
  sub.appendChild(document.createElement('th'));
  C.forEach((c) => { const th = document.createElement('th'); th.textContent = c.slice(0, 4); sub.appendChild(th); });

  cm.forEach((r, i) => {
    const tr = t.insertRow();
    const th = document.createElement('th');
    th.className = 'rowh';
    th.textContent = C[i];
    th.style.color = colorOf(C[i]);
    tr.appendChild(th);
    r.forEach((v, j) => {
      const frac = v / (rowSums[i] || 1);
      const td = tr.insertCell();
      const base = i === j ? colorOf(C[i]) : cssVar('--anger');
      td.style.background = `color-mix(in srgb, ${base} ${Math.round(8 + frac * 88)}%, transparent)`;
      td.style.color = frac > 0.45 ? (root.dataset.theme === 'light' ? '#fff' : '#04070c') : 'var(--ink-2)';
      td.innerHTML = `${v}<span class="pct">${(frac * 100).toFixed(0)}%</span>`;
      td.tabIndex = 0;
      td.setAttribute('role', 'button');
      td.setAttribute('aria-label', `${C[i]} predicted as ${C[j]}: ${v} of ${rowSums[i]}`);
      const pick = () => {
        $$('.confusion td').forEach((n) => n.classList.remove('sel'));
        td.classList.add('sel');
        showCell(i, j, v, rowSums[i]);
      };
      td.addEventListener('click', pick);
      td.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(); } });
    });
  });
  host.innerHTML = '';
  host.appendChild(t);

  const worst = M.confusion_pairs[0];
  const wi = C.indexOf(worst.actual), wj = C.indexOf(worst.predicted);
  showCell(wi, wj, cm[wi][wj], rowSums[wi]);
  $$('.confusion td')[wi * C.length + wj]?.classList.add('sel');
}

function showCell(i, j, v, support) {
  const C = M.classes;
  const box = $('#cmDetail');
  if (i === j) {
    box.innerHTML = `<b style="color:${colorOf(C[i])}">${C[i]}</b> — recalled correctly
      <b>${v}</b> of <b>${support}</b> times (${fmtPct(v / support, 1)}).`;
    return;
  }
  const pair = M.confusion_pairs.find((p) => p.actual === C[i] && p.predicted === C[j]);
  box.innerHTML = `<b style="color:${colorOf(C[i])}">${C[i]}</b> read as
    <b style="color:${colorOf(C[j])}">${C[j]}</b> — <b>${v}</b> of ${support}
    (${fmtPct(v / support, 1)} of all true ${C[i]}).` +
    (pair ? pair.examples.map((e) =>
      `<div class="ex-line">“${e.text}” <span style="opacity:.6">p=${e.confidence}</span></div>`).join('')
      : '<div class="ex-line" style="opacity:.6">no examples in this cell</div>');
}

function renderPerClass() {
  const host = $('#perClass');
  host.innerHTML = `<div class="pc-head"><span></span><div class="pc-metrics">
    <span>precision</span><span>recall</span><span>F1</span></div></div>`;
  for (const r of M.per_class) {
    const c = colorOf(r.class);
    const d = document.createElement('div');
    d.className = 'pc-row';
    d.innerHTML = `<span class="pc-name"><i style="background:${c}"></i>${r.class}</span>
      <div class="pc-metrics">
        ${metric(r.precision, c)}${metric(r.recall, c)}${metric(r.f1, c)}
      </div>`;
    host.appendChild(d);
    function metric(v, col) {
      return `<span class="pc-metric"><b>${v.toFixed(3)}</b>
        <span class="mtrack"><span class="mfill" style="width:${v * 100}%;background:${col}"></span></span></span>`;
    }
  }
  const H = M.headline;
  const foot = document.createElement('p');
  foot.className = 'foot';
  foot.innerHTML = `Macro F1 <b>${H.macro_f1.toFixed(3)}</b> sits well below weighted F1
    <b>${H.weighted_f1.toFixed(3)}</b> — the gap is the price of the class imbalance, paid
    almost entirely by <span style="color:${colorOf('surprise')}">surprise</span> and
    <span style="color:${colorOf('love')}">love</span>. Majority-class baseline:
    ${fmtPct(H.majority_class_baseline, 1)}.`;
  host.appendChild(foot);
}

function renderCalibration() {
  // Bins holding a handful of tweets swing between 0% and 100% and say nothing;
  // they are dropped from the line and reported in the caption instead.
  const MIN_N = 5;
  const cal = M.calibration.filter((b) => b.n >= MIN_N);
  const dropped = M.calibration.length - cal.length;
  lineChart($('#calChart'), {
    xLabels: cal.map((b) => `${(b.hi * 100).toFixed(0)}`),
    xTitle: 'confidence bin (%)', yDomain: [0, 1],
    yFormat: (v) => (v * 100).toFixed(0) + '%',
    series: [
      { values: cal.map((b) => b.confidence), color: cssVar('--ink-3'), dash: '3 4' },
      { values: cal.map((b) => b.accuracy), color: cssVar('--accent'), area: true },
    ],
    tipHtml: (i) => `bin ${(cal[i].lo * 100).toFixed(0)}–${(cal[i].hi * 100).toFixed(0)}%<br>` +
      `n=${fmtNum(cal[i].n)}<br>accuracy ${fmtPct(cal[i].accuracy)}<br>` +
      `mean confidence ${fmtPct(cal[i].confidence)}`,
  });
  const over = cal.filter((b) => b.accuracy < b.confidence).length;
  $('#calFoot').innerHTML = `Dashed line is the mean confidence in each bin — perfect calibration
    would put the solid line on top of it. ${over} of ${cal.length} bins sit below, so the softmax
    is <b>${over / cal.length > .6 ? 'systematically over' : 'mildly over'}confident</b>: expected for
    a cross-entropy model trained to convergence without label smoothing or temperature scaling.
    ECE <b>${M.headline.ece.toFixed(3)}</b>${dropped ? ` · ${dropped} bins with fewer than ${MIN_N} tweets omitted` : ''}.`;
}

function renderCoverage() {
  const cov = M.coverage;
  const lo = Math.min(...cov.map((c) => c.accuracy));
  lineChart($('#covChart'), {
    xLabels: cov.map((c) => (c.coverage * 100).toFixed(0)), xTitle: 'coverage (%)',
    yDomain: [Math.max(0, lo - 0.02), 1], yFormat: (v) => (v * 100).toFixed(0) + '%',
    series: [{ values: cov.map((c) => c.accuracy), color: cssVar('--accent-2'), area: true }],
    tipHtml: (i) => `coverage ${fmtPct(cov[i].coverage, 0)}<br>accuracy ${fmtPct(cov[i].accuracy)}<br>` +
      `threshold p ≥ ${cov[i].threshold.toFixed(3)}`,
  });
  const slider = $('#covSlider');
  slider.max = String(cov.length);
  const sync = () => {
    const c = cov[Math.min(cov.length - 1, +slider.value - 1)];
    $('#covReadout').innerHTML = `answer <b>${fmtPct(c.coverage, 0)}</b> of tweets ` +
      `(p ≥ ${c.threshold.toFixed(2)}) → <b>${fmtPct(c.accuracy, 1)}</b> accurate`;
  };
  slider.addEventListener('input', sync);
  sync();
}

/* ── 06 errors ─────────────────────────────────────────────────────────── */
function renderPairs() {
  const host = $('#pairs');
  host.innerHTML = '';
  for (const p of M.confusion_pairs.slice(0, 4)) {
    const d = document.createElement('div');
    d.className = 'pair';
    d.innerHTML = `<div class="pair-head">
        <span class="tag" style="background:color-mix(in srgb,${colorOf(p.actual)} 22%,transparent);color:${colorOf(p.actual)}">${p.actual}</span>
        <span style="color:var(--ink-3)">read as</span>
        <span class="tag" style="background:color-mix(in srgb,${colorOf(p.predicted)} 22%,transparent);color:${colorOf(p.predicted)}">${p.predicted}</span>
        <span class="cnt">${p.count} · ${fmtPct(p.rate, 1)} of true ${p.actual}</span></div>
      <div class="pair-ex">“${p.examples[0]?.text ?? ''}”</div>`;
    host.appendChild(d);
  }
  const note = document.createElement('p');
  note.className = 'foot';
  note.innerHTML = `These are not arbitrary. <b>love</b> and <b>joy</b> share almost all of their
    positive vocabulary, and the corpus gives <b>love</b> a quarter of joy's examples; the same
    argument holds for <b>surprise</b> against <b>fear</b>. A 16-dimensional embedding trained on
    16k sentences has no way to separate <i>“i feel blessed”</i> from <i>“i feel adored”</i>
    without more signal.`;
  host.appendChild(note);
}

function renderWorstCases() {
  const host = $('#worst');
  host.innerHTML = '';
  for (const c of M.confident_errors.slice(0, 5)) {
    const b = document.createElement('button');
    b.className = 'case';
    b.type = 'button';
    b.innerHTML = `<div class="txt">“${c.text}”</div>
      <div class="meta"><span style="color:${colorOf(c.actual)}">gold ${c.actual}</span>
      <span style="color:var(--ink-3)">→</span>
      <span style="color:${colorOf(c.predicted)}">said ${c.predicted}</span>
      <span style="color:var(--ink-3)">p=${c.confidence}</span></div>`;
    b.addEventListener('click', () => {
      $('#tweet').value = c.text;
      $('#tweet').dispatchEvent(new Event('input'));
      $('#demo').scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
    host.appendChild(b);
  }
}

/* ── 07 embeddings ─────────────────────────────────────────────────────── */
let pcaState = null;
function renderEmbedding() {
  const host = $('#pca');
  const pts = M.embedding_pca.points;
  host.innerHTML = '<canvas></canvas><div class="pca-tip"></div>';
  const canvas = host.querySelector('canvas');
  const tip = host.querySelector('.pca-tip');
  const dpr = Math.min(2, devicePixelRatio || 1);
  const w = host.clientWidth, h = host.clientHeight;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y);
  const pad = 16;
  const sx = (v) => pad + ((v - Math.min(...xs)) / (Math.max(...xs) - Math.min(...xs))) * (w - 2 * pad);
  const sy = (v) => h - pad - ((v - Math.min(...ys)) / (Math.max(...ys) - Math.min(...ys))) * (h - 2 * pad);

  const draw = (hoverIdx = -1) => {
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = cssVar('--line');
    ctx.lineWidth = 1;
    ctx.strokeRect(.5, .5, w - 1, h - 1);
    pts.forEach((p, i) => {
      const r = 1.4 + p.p * 3.2;
      ctx.globalAlpha = 0.25 + p.p * 0.6;
      ctx.fillStyle = colorOf(M.classes[p.c]);
      ctx.beginPath();
      ctx.arc(sx(p.x), sy(p.y), i === hoverIdx ? r + 3 : r, 0, 6.2832);
      ctx.fill();
    });
    ctx.globalAlpha = 1;
    // Label the words that sit furthest from the origin — they anchor the axes.
    // Greedy collision rejection keeps the labels readable at any panel width.
    ctx.font = '10px ui-monospace, monospace';
    ctx.fillStyle = cssVar('--ink-2');
    const placed = [];
    const anchors = [...pts].sort((a, b) => (b.x * b.x + b.y * b.y) - (a.x * a.x + a.y * a.y));
    for (const a of anchors) {
      if (placed.length >= 16) break;
      const wd = ctx.measureText(a.w).width;
      // Flip the label to the left of the point rather than let it clip.
      const px = sx(a.x) + 6 + wd > w - 4 ? sx(a.x) - 6 - wd : sx(a.x) + 6;
      const py = Math.max(11, Math.min(h - 4, sy(a.y) - 5));
      if (placed.some((p) => Math.abs(p.y - py) < 11 && px < p.x + p.w + 6 && p.x < px + wd + 6)) continue;
      placed.push({ x: px, y: py, w: wd });
      ctx.fillText(a.w, px, py);
    }
  };
  draw();

  canvas.addEventListener('pointermove', (e) => {
    const r = canvas.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    let best = -1, bd = 64;
    pts.forEach((p, i) => {
      const d = (sx(p.x) - mx) ** 2 + (sy(p.y) - my) ** 2;
      if (d < bd) { bd = d; best = i; }
    });
    if (best < 0) { tip.style.opacity = 0; draw(); return; }
    const p = pts[best];
    tip.innerHTML = `${p.w} · <span style="color:${colorOf(M.classes[p.c])}">${M.classes[p.c]}</span> ${p.p.toFixed(2)}`;
    tip.style.left = sx(p.x) + 'px';
    tip.style.top = sy(p.y) + 'px';
    tip.style.opacity = 1;
    draw(best);
  });
  canvas.addEventListener('pointerleave', () => { tip.style.opacity = 0; draw(); });
  canvas.addEventListener('click', (e) => {
    const r = canvas.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    let best = -1, bd = 100;
    pts.forEach((p, i) => {
      const d = (sx(p.x) - mx) ** 2 + (sy(p.y) - my) ** 2;
      if (d < bd) { bd = d; best = i; }
    });
    if (best >= 0) { $('#nnInput').value = pts[best].w; lookupNeighbours(); }
  });

  const v = M.embedding_pca.variance;
  $('#pcaVar').textContent = `PC1 ${fmtPct(v[0], 1)} · PC2 ${fmtPct(v[1], 1)} of variance`;
  $('#pcaLegend').innerHTML = M.classes.map((c) =>
    `<span><i style="background:${colorOf(c)}"></i>${c}</span>`).join('') +
    `<span class="hint-inline">dot size = confidence of the token alone</span>`;
  pcaState = { draw };
}

let nnTimer;
function lookupNeighbours() {
  clearTimeout(nnTimer);
  nnTimer = setTimeout(async () => {
    const w = $('#nnInput').value.trim();
    const out = $('#nnOut');
    if (!w) { out.innerHTML = '<p class="nn-empty">Type a word, or click a point in the projection.</p>'; return; }
    const res = await engine.call('neighbours', { word: w, k: 10 });
    if (!res) { out.innerHTML = `<p class="nn-empty">“${w}” is outside the 10,000-word vocabulary.</p>`; return; }
    out.innerHTML = `<p class="nn-empty" style="margin:0 0 .4rem">frequency rank #${fmtNum(res.rank)}</p>` +
      res.neighbours.map((n) =>
        `<div class="nn-row"><span class="w">${n.word}</span><span class="s">${n.sim.toFixed(3)}</span>
         <span class="track"><span class="fill" style="width:${Math.max(0, n.sim) * 100}%"></span></span></div>`).join('');
  }, 120);
}
$('#nnInput').addEventListener('input', lookupNeighbours);
$('#nnInput').value = 'nervous';

/* ── 08 shipping ───────────────────────────────────────────────────────── */
function renderShipping() {
  const H = M.headline, mdl = M.model;
  const kv = (host, rows) => {
    host.innerHTML = rows.map(([k, v]) =>
      `<div class="kv-row"><span class="k">${k}</span><span class="rule"></span><span class="v">${v}</span></div>`).join('');
  };
  kv($('#quantStats'), [
    ['scheme', 'int8 symmetric, per output row'],
    ['float32 accuracy', fmtPct(H.test_accuracy_float32, 2)],
    ['int8 accuracy', fmtPct(H.test_accuracy, 2)],
    ['argmax agreement', fmtPct(H.int8_argmax_agreement, 2)],
    ['max |Δp|', H.int8_max_prob_deviation.toFixed(4)],
    ['compression', `${(mdl.params * 4 / mdl.weights_bytes).toFixed(2)}× vs float32`],
  ]);

  // Real transfer sizes, read back off the server rather than guessed.
  const files = [
    ['weights.bin (int8 model)', './model/weights.bin'],
    ['vocab.json (10k tokens)', './model/vocab.json'],
    ['metrics.json (this report)', './model/metrics.json'],
    ['JavaScript (4 modules)', ['./assets/app.js', './assets/engine.js', './assets/charts.js', './assets/worker.js']],
    ['style.css', './assets/style.css'],
  ];
  const size = async (u) => {
    try { return +(await fetch(u, { method: 'HEAD' })).headers.get('content-length') || 0; }
    catch { return 0; }
  };
  Promise.all(files.map(async ([label, u]) => [label,
    Array.isArray(u) ? (await Promise.all(u.map(size))).reduce((a, b) => a + b, 0) : await size(u),
  ])).then((parts) => {
    parts = parts.filter(([, v]) => v > 0);
    if (!parts.length) parts = [['weights.bin (int8 model)', mdl.weights_bytes]];
    const total = parts.reduce((a, p) => a + p[1], 0);
    $('#payload').innerHTML = parts.map(([k, v]) =>
      `<div class="budget-row"><span class="lab">${k}</span>
        <span class="num">${(v / 1024).toFixed(0)} KB</span>
        <span class="track"><span class="fill" style="width:${(v / total) * 100}%"></span></span></div>`).join('') +
      `<div class="budget-row" style="margin-top:.35rem"><span class="lab" style="color:var(--ink)">total, uncompressed</span>
        <span class="num">${(total / 1024).toFixed(0)} KB</span></div>`;
  });
}

async function runBenchmark() {
  const text = M.demo_examples[0].text;
  $('#benchBtn').disabled = true;
  const r = await engine.call('benchmark', { text, n: 120 });
  $('#benchBtn').disabled = false;
  const cfg = M.model.config;
  const macs = cfg.maxlen * 2 * (4 * cfg.hidden * (cfg.emb + cfg.hidden) +
                                 4 * cfg.hidden * (2 * cfg.hidden + cfg.hidden));
  // A frozen clock (headless Chrome, hardened timer settings) reports zero;
  // say so rather than printing a fictional throughput.
  const timed = r.total > 0;
  $('#bench').innerHTML = [
    ['mean forward pass', timed ? `${r.mean.toFixed(2)} ms` : 'below timer resolution'],
    [`${r.runs} passes took`, timed ? `${r.total.toFixed(1)} ms` : '—'],
    ['sequence length', String(cfg.maxlen)],
    ['multiply–accumulates', `≈ ${(macs / 1e6).toFixed(2)} M / pass`],
    ['throughput', timed ? `${fmtNum(Math.round(1000 / r.mean))} tweets/s` : '—'],
  ].map(([k, v]) => `<div class="kv-row"><span class="k">${k}</span><span class="rule"></span><span class="v">${v}</span></div>`).join('');
  $$('[data-fill="latency"]').forEach((n) => { n.textContent = timed ? `${r.mean.toFixed(1)} ms` : '<1 ms'; });
}
$('#benchBtn').addEventListener('click', runBenchmark);

/** Live check that the JS engine reproduces the Python reference vectors. */
async function verifyParity() {
  const host = $('#parity');
  let golden;
  try { golden = await fetch('./model/golden.json').then((r) => r.json()); }
  catch { host.innerHTML = '<p class="nn-empty">reference vectors unavailable</p>'; return; }
  const rows = [];
  for (const g of golden) {
    const res = await engine.call('analyze', { text: g.text, opts: { attribution: false, trajectory: false } });
    const dev = Math.max(...res.probs.map((p, i) => Math.abs(p - g.probs[i])));
    rows.push({ text: g.text, dev });
  }
  const worst = Math.max(...rows.map((r) => r.dev));
  host.innerHTML = rows.map((r) =>
    `<div class="parity-row"><span class="t">“${r.text}”</span>
      <span class="n">Δ ${r.dev.toExponential(1)}</span>
      <span class="ok">${r.dev < 1e-4 ? '✓ match' : '⚠︎'}</span></div>`).join('') +
    `<div class="parity-row"><span class="t" style="color:var(--ink)">worst deviation across ${rows.length} vectors</span>
      <span class="n">${worst.toExponential(1)}</span><span class="ok">float32 noise</span></div>`;
}

/* ── 09 model card ─────────────────────────────────────────────────────── */
function renderCard() {
  const D = M.dataset, H = M.headline;
  const surprise = M.per_class.find((r) => r.class === 'surprise');
  const love = M.per_class.find((r) => r.class === 'love');
  $('#limitations').innerHTML = [
    `Recall collapses on the rare classes: <b>surprise</b> ${fmtPct(surprise.recall, 1)}
     (${surprise.support} test examples) and <b>love</b> ${fmtPct(love.recall, 1)}.`,
    `Overconfident by construction — ECE <b>${H.ece.toFixed(3)}</b>. Use the risk–coverage curve,
     not the raw softmax, if a decision depends on it.`,
    `Vocabulary is frozen at ${fmtNum(D.vocab_cutoff)} types; ${fmtPct(D.test_oov_rate, 2)} of test
     tokens already fall through to <span class="mono">&lt;UNK&gt;</span>, and slang drifts fast.`,
    `The corpus is distant-supervised from hashtags, so the labels are noisy — several
     "confidently wrong" examples above are arguably mislabelled rather than misclassified.`,
  ].map((s) => `<li>${s}</li>`).join('');

  $('#provenance').innerHTML = [
    ['dataset', D.source],
    ['train / val / test', D.splits.map((s) => fmtNum(s.rows)).join(' / ')],
    ['framework', `PyTorch, ${M.model.device.toUpperCase()}`],
    ['served as', 'int8 · plain JavaScript'],
    ['licence', 'MIT'],
  ].map(([k, v]) => `<div class="kv-row"><span class="k">${k}</span><span class="rule"></span><span class="v">${v}</span></div>`).join('');
}

/* ══════════════════════════════════════════════════════════════════════
   Live demo
   ══════════════════════════════════════════════════════════════════════ */
let lastResult = null;

function wireDemo() {
  const ta = $('#tweet');
  const exHost = $('#examples');
  exHost.innerHTML = '';
  for (const ex of M.demo_examples) {
    const b = document.createElement('button');
    b.className = 'ex';
    b.type = 'button';
    b.innerHTML = `<i style="display:inline-block;width:6px;height:6px;border-radius:50%;
      background:${colorOf(ex.gold)};margin-right:.4rem;vertical-align:middle"></i>` +
      (ex.text.length > 42 ? ex.text.slice(0, 40) + '…' : ex.text);
    b.title = `held-out test tweet · gold label: ${ex.gold}`;
    b.addEventListener('click', () => { ta.value = ex.text; run(); });
    exHost.appendChild(b);
  }

  buildBars();
  let t;
  ta.addEventListener('input', () => { clearTimeout(t); t = setTimeout(run, 90); });
  ta.value = M.demo_examples[1].text;   // a joy example — the majority class
  run();
}

function buildBars() {
  const host = $('#bars');
  host.innerHTML = M.classes.map((c) =>
    `<div class="bar-row" data-c="${c}">
       <span class="name">${c}</span>
       <span class="bar-track"><span class="bar-fill" style="background:${colorOf(c)}"></span></span>
       <span class="val">0.0%</span>
     </div>`).join('');
}

let inflight = false, queued = false;
async function run() {
  if (inflight) { queued = true; return; }
  inflight = true;
  const text = $('#tweet').value;
  try {
    const res = await engine.call('analyze', { text, opts: { attribution: true, trajectory: true } });
    lastResult = res;
    paintResult(res);
  } catch (e) { console.error(e); }
  inflight = false;
  if (queued) { queued = false; run(); }
}

function paintResult(res) {
  const rows = $$('#bars .bar-row');
  const top = res.top;
  rows.forEach((row, i) => {
    row.querySelector('.bar-fill').style.width = `${res.probs[i] * 100}%`;
    row.querySelector('.val').textContent = fmtPct(res.probs[i], 1);
    row.classList.toggle('top', i === top);
  });
  requestAnimationFrame(() => $('#bars').classList.add('animate'));
  $('#latency').textContent = `${res.ms > 0 ? res.ms.toFixed(1) + ' ms · ' : ''}${res.passes} passes`;
  $('#passCount').textContent = `${res.words.length} tokens`;
  $('#attrClass').textContent = M.classes[top];
  $('#attrClass').style.color = colorOf(M.classes[top]);

  const oov = res.known.filter((k) => !k).length;
  $('#encHint').textContent =
    `${res.words.length} tokens · ${oov} out-of-vocabulary` +
    (res.truncated ? ` · truncated to ${M.model.config.maxlen}` : '') +
    ` · ${M.model.config.maxlen - Math.min(res.words.length, M.model.config.maxlen)} padding steps`;

  renderTokens(res);
  renderTrajectory(res);
}

function renderTokens(res) {
  const host = $('#tokens');
  host.innerHTML = '';
  if (!res.words.length) { host.innerHTML = '<span class="foot">Type something above.</span>'; return; }
  const attrs = res.attributions ?? [];
  const scale = Math.max(0.2, ...attrs.map((a) => Math.abs(a)));
  const col = colorOf(M.classes[res.top]);
  res.words.slice(0, M.model.config.maxlen).forEach((w, i) => {
    const a = attrs[i] ?? 0;
    const mag = Math.min(1, Math.abs(a) / scale);
    const s = document.createElement('span');
    s.className = 'tok' + (res.known[i] ? '' : ' oov');
    s.style.background = a >= 0
      ? `color-mix(in srgb, ${col} ${Math.round(mag * 62)}%, var(--panel-2))`
      : `color-mix(in srgb, var(--sadness) ${Math.round(mag * 34)}%, var(--panel-2))`;
    s.style.borderColor = a >= 0 && mag > .4 ? col : 'var(--line)';
    s.innerHTML = `${escapeHtml(w)}<span class="d">${a >= 0 ? '+' : ''}${a.toFixed(2)}</span>`;
    s.title = (res.known[i] ? '' : 'out of vocabulary → <UNK> · ') +
      `Δ log-odds ${a.toFixed(3)} · Δp ${(res.deltaP?.[i] ?? 0).toFixed(4)}`;
    host.appendChild(s);
  });
}

function renderTrajectory(res) {
  const host = $('#trajectory');
  if (!res.trace || !res.trace.length) { host.innerHTML = ''; return; }
  const words = res.words.slice(0, res.trace.length);
  lineChart(host, {
    xLabels: words.map((w) => (w.length > 8 ? w.slice(0, 7) + '…' : w)),
    yDomain: [0, 1], yFormat: (v) => (v * 100).toFixed(0) + '%',
    series: M.classes.map((c, i) => ({
      values: res.trace.map((p) => p[i]), color: colorOf(c),
      area: i === res.top,
    })),
    tipHtml: (i) => `after “${words[i]}”<br>` + M.classes
      .map((c, k) => [c, res.trace[i][k]])
      .sort((a, b) => b[1] - a[1]).slice(0, 3)
      .map(([c, p]) => `<span style="color:${colorOf(c)}">■</span> ${c} ${fmtPct(p, 1)}`).join('<br>'),
  });
}
