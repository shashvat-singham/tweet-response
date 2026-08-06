/**
 * Minimal SVG chart toolkit — enough for this report, nothing more.
 * Everything is theme-aware through CSS custom properties, so a theme flip
 * needs no redraw beyond re-reading colour variables for canvas.
 */

const NS = 'http://www.w3.org/2000/svg';

export const el = (tag, attrs = {}, parent = null) => {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
};

export const fmtPct = (v, d = 1) => (v * 100).toFixed(d) + '%';
export const fmtNum = (v) => v.toLocaleString('en-US');

function frame(host, { pad = [14, 12, 22, 34] } = {}) {
  host.innerHTML = '';
  const wrap = document.createElement('div');
  wrap.className = 'chart-wrap';
  wrap.style.height = '100%';
  host.appendChild(wrap);
  const W = host.clientWidth || 420;
  const H = host.clientHeight || 180;
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: 'none' }, wrap);
  svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
  const [pt, pr, pb, pl] = pad;
  return {
    svg, wrap, W, H, pt, pr, pb, pl,
    iw: W - pl - pr, ih: H - pt - pb,
    x: (t) => pl + t * (W - pl - pr),
    y: (t) => pt + (1 - t) * (H - pt - pb),
  };
}

function tooltip(wrap) {
  const tip = document.createElement('div');
  tip.className = 'chart-tip';
  wrap.appendChild(tip);
  return {
    show(x, y, html) { tip.innerHTML = html; tip.style.left = x + 'px'; tip.style.top = y + 'px'; tip.style.opacity = 1; },
    hide() { tip.style.opacity = 0; },
  };
}

/** Multi-series line chart with a shared vertical cursor. */
export function lineChart(host, {
  series, xLabels, yDomain, yTicks = 4, yFormat = (v) => v.toFixed(2),
  xTitle = '', markers = [], tipHtml,
}) {
  const f = frame(host, { pad: [14, 14, xTitle ? 38 : 26, 38] });
  const n = xLabels.length;
  const all = series.flatMap((s) => s.values);
  const lo = yDomain ? yDomain[0] : Math.min(...all) * 0.96;
  const hi = yDomain ? yDomain[1] : Math.max(...all) * 1.04;
  const ty = (v) => (v - lo) / (hi - lo || 1);
  const tx = (i) => (n === 1 ? 0.5 : i / (n - 1));

  for (let t = 0; t <= yTicks; t++) {
    const v = lo + (hi - lo) * (t / yTicks);
    const y = f.y(t / yTicks);
    el('line', { class: 'grid-line', x1: f.pl, x2: f.W - f.pr, y1: y, y2: y }, f.svg);
    el('text', { class: 'ax-txt', x: f.pl - 6, y: y + 3, 'text-anchor': 'end' }, f.svg)
      .textContent = yFormat(v);
  }
  const tickRow = xTitle ? f.H - 20 : f.H - 8;
  const step = Math.max(1, Math.ceil(n / 8));
  xLabels.forEach((lab, i) => {
    if (i % step && i !== n - 1) return;
    const anchor = i === 0 ? 'start' : i === n - 1 ? 'end' : 'middle';
    el('text', { class: 'ax-txt', x: f.x(tx(i)), y: tickRow, 'text-anchor': anchor }, f.svg)
      .textContent = lab;
  });
  if (xTitle) el('text', { class: 'ax-txt', x: f.pl + f.iw / 2, y: f.H - 5,
                           'text-anchor': 'middle', opacity: .8 }, f.svg).textContent = xTitle;

  for (const m of markers) {
    const x = f.x(tx(m.at));
    el('line', { class: 'hoverline', x1: x, x2: x, y1: f.pt, y2: f.H - f.pb,
                 'stroke-dasharray': '3 3' }, f.svg);
    el('text', { class: 'ax-txt', x: x - 4, y: f.pt + 9, 'text-anchor': 'end' }, f.svg)
      .textContent = m.label;
  }

  for (const s of series) {
    const d = s.values.map((v, i) => `${i ? 'L' : 'M'}${f.x(tx(i))},${f.y(ty(v))}`).join(' ');
    if (s.area) {
      el('path', { d: `${d} L${f.x(tx(n - 1))},${f.y(0)} L${f.x(0)},${f.y(0)} Z`,
                   fill: s.color, opacity: .10, stroke: 'none' }, f.svg);
    }
    el('path', { class: 'series', d, stroke: s.color,
                 'stroke-dasharray': s.dash || 'none' }, f.svg);
    if (n <= 20) s.values.forEach((v, i) =>
      el('circle', { class: 'marker', cx: f.x(tx(i)), cy: f.y(ty(v)), r: 2.6, stroke: s.color }, f.svg));
  }

  const tip = tooltip(f.wrap);
  const cursor = el('line', { class: 'hoverline', y1: f.pt, y2: f.H - f.pb, opacity: 0 }, f.svg);
  const hit = el('rect', { x: 0, y: 0, width: f.W, height: f.H, fill: 'transparent' }, f.svg);
  hit.style.cursor = 'crosshair';
  hit.addEventListener('pointermove', (e) => {
    const r = f.svg.getBoundingClientRect();
    const rel = ((e.clientX - r.left) / r.width * f.W - f.pl) / f.iw;
    const i = Math.max(0, Math.min(n - 1, Math.round(rel * (n - 1))));
    const px = f.x(tx(i));
    cursor.setAttribute('x1', px); cursor.setAttribute('x2', px); cursor.setAttribute('opacity', 1);
    tip.show(px / f.W * r.width, f.y(ty(series[0].values[i])) / f.H * r.height,
      tipHtml ? tipHtml(i) : `${xLabels[i]}<br>` +
        series.map((s) => `<span style="color:${s.color}">■</span> ${yFormat(s.values[i])}`).join('<br>'));
  });
  hit.addEventListener('pointerleave', () => { cursor.setAttribute('opacity', 0); tip.hide(); });
  return f;
}

/** Vertical bars; `bars` = [{label, value, color, sub}] */
export function barChart(host, { bars, yFormat = fmtNum, max }) {
  const f = frame(host, { pad: [12, 8, 24, 34] });
  const hi = max ?? Math.max(...bars.map((b) => b.value)) * 1.08;
  const bw = f.iw / bars.length;
  const tip = tooltip(f.wrap);
  bars.forEach((b, i) => {
    const h = (b.value / hi) * f.ih;
    const x = f.pl + i * bw + bw * 0.16;
    const w = bw * 0.68;
    const r = el('rect', { x, y: f.pt + f.ih - h, width: w, height: Math.max(1, h),
                           rx: 3, fill: b.color }, f.svg);
    r.style.cursor = 'pointer';
    r.addEventListener('pointerenter', () => tip.show(
      (x + w / 2) / f.W * f.wrap.clientWidth, (f.pt + f.ih - h) / f.H * f.wrap.clientHeight,
      `${b.label}<br>${yFormat(b.value)}${b.sub ? '<br>' + b.sub : ''}`));
    r.addEventListener('pointerleave', () => tip.hide());
    el('text', { class: 'ax-txt', x: x + w / 2, y: f.H - 8, 'text-anchor': 'middle' }, f.svg)
      .textContent = b.label;
  });
  return f;
}

/** Histogram from [{x, n}] with an optional vertical cut marker. */
export function histogram(host, { data, cut, color }) {
  const f = frame(host, { pad: [12, 10, 24, 34] });
  const hi = Math.max(...data.map((d) => d.n));
  const bw = f.iw / data.length;
  const tip = tooltip(f.wrap);
  data.forEach((d, i) => {
    const h = (d.n / hi) * f.ih;
    const x = f.pl + i * bw;
    const r = el('rect', { x: x + .3, y: f.pt + f.ih - h, width: Math.max(.8, bw - .8),
                           height: Math.max(.6, h), fill: color, opacity: .85 }, f.svg);
    r.addEventListener('pointerenter', () => tip.show(
      (x + bw / 2) / f.W * f.wrap.clientWidth, (f.pt + f.ih - h) / f.H * f.wrap.clientHeight,
      `${d.x} words<br>${fmtNum(d.n)} tweets`));
    r.addEventListener('pointerleave', () => tip.hide());
  });
  [0, .25, .5, .75, 1].forEach((t) => {
    const i = Math.round(t * (data.length - 1));
    el('text', { class: 'ax-txt', x: f.pl + (i + .5) * bw, y: f.H - 8, 'text-anchor': 'middle' }, f.svg)
      .textContent = data[i].x;
  });
  if (cut != null) {
    const idx = data.findIndex((d) => d.x >= cut);
    if (idx > -1) {
      const x = f.pl + idx * bw;
      el('line', { x1: x, x2: x, y1: f.pt, y2: f.pt + f.ih, class: 'hoverline',
                   'stroke-dasharray': '3 3' }, f.svg);
      el('text', { class: 'ax-txt', x: x - 5, y: f.pt + 10, 'text-anchor': 'end' }, f.svg)
        .textContent = `maxlen ${cut}`;
    }
  }
  return f;
}

/** Horizontal stacked bars; `rows` = [{label, parts:[{value,color,name}]}] */
export function stackedRows(host, { rows, total }) {
  host.innerHTML = '';
  const box = document.createElement('div');
  box.style.display = 'grid';
  box.style.gap = '.6rem';
  box.style.paddingTop = '.2rem';
  for (const row of rows) {
    const sum = row.parts.reduce((a, p) => a + p.value, 0);
    const line = document.createElement('div');
    line.innerHTML = `<div style="display:flex;justify-content:space-between;font-size:.75rem;color:var(--ink-3);margin-bottom:.22rem">
        <span>${row.label}</span><span class="mono">${fmtNum(sum)}</span></div>`;
    const bar = document.createElement('div');
    bar.style.cssText = 'display:flex;height:14px;border-radius:5px;overflow:hidden;box-shadow:inset 0 0 0 1px var(--line)';
    for (const p of row.parts) {
      const seg = document.createElement('div');
      seg.style.cssText = `width:${(p.value / (total ?? sum)) * 100}%;background:${p.color}`;
      seg.title = `${p.name}: ${fmtNum(p.value)} (${fmtPct(p.value / sum)})`;
      bar.appendChild(seg);
    }
    line.appendChild(bar);
    box.appendChild(line);
  }
  host.appendChild(box);
}
