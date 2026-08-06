/**
 * Tweet-emotion BiLSTM — inference engine.
 *
 * Runs the exact network trained in `ml/train.py` directly in the browser:
 * int8 weights are dequantised once at load, then every prediction is a
 * hand-written forward pass over typed arrays. No TensorFlow.js, no WASM,
 * no network round-trip — the model is ~180 KB and the whole thing is
 * dependency-free.
 *
 * Tokenisation mirrors `tf.keras.preprocessing.text.Tokenizer`
 * (num_words=10000, oov_token='<UNK>', post padding/truncating to 50).
 */

const KERAS_FILTERS = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n';
const FILTER_SET = new Set(KERAS_FILTERS.split(''));

export function tokenizeWords(text) {
  let out = '';
  for (const ch of text.toLowerCase()) out += FILTER_SET.has(ch) ? ' ' : ch;
  return out.split(' ').filter(Boolean);
}

/**
 * Dequantise once, at load, into flat Float32Arrays. int8 tensors carry one
 * scale per output row (see `quantize` in ml/export.py); biases are float32.
 */
function dequantize(meta, buffer) {
  const params = {};
  for (const t of meta.tensors) {
    const n = t.shape.reduce((a, b) => a * b, 1);
    if (t.dtype === 'i8') {
      const q = new Int8Array(buffer, t.offset, n);
      const scales = new Float32Array(buffer.slice(
        t.scale_offset, t.scale_offset + t.scale_count * 4));
      const cols = n / t.scale_count;
      const f = new Float32Array(n);
      for (let r = 0; r < t.scale_count; r++) {
        const s = scales[r], o = r * cols;
        for (let c = 0; c < cols; c++) f[o + c] = q[o + c] * s;
      }
      params[t.name] = f;
    } else {
      params[t.name] = new Float32Array(buffer.slice(t.offset, t.offset + t.bytes));
    }
    params[t.name + '_shape'] = t.shape;
  }
  return params;
}

export class EmotionModel {
  constructor(meta, weights, vocab) {
    this.meta = meta;
    this.p = dequantize(meta, weights);
    this.vocab = vocab;
    this.index = new Map();
    vocab.forEach((w, i) => { if (i > 0 && w) this.index.set(w, i); });
    this.maxlen = meta.maxlen;
    this.H = meta.hidden;
    this.E = meta.emb;
    this.classes = meta.classes;
    this.oov = this.index.get('<UNK>') ?? 1;

    const T = this.maxlen, H = this.H;
    // Scratch buffers reused across calls so a prediction allocates nothing.
    this._emb = new Float32Array(T * this.E);
    this._h1 = new Float32Array(T * 2 * H);
    this._seqA = new Float32Array(T * H);
    this._seqB = new Float32Array(T * H);
    this._gates = new Float32Array(4 * H);
    this._h = new Float32Array(H);
    this._c = new Float32Array(H);
    this._ids = new Int32Array(T);
  }

  static async load(base = './model/') {
    const [meta, weights, vocab] = await Promise.all([
      fetch(base + 'model.json').then((r) => r.json()),
      fetch(base + 'weights.bin').then((r) => r.arrayBuffer()),
      fetch(base + 'vocab.json').then((r) => r.json()),
    ]);
    return new EmotionModel(meta, weights, vocab);
  }

  /** Keras `texts_to_sequences` + `pad_sequences(truncating='post')`. */
  encode(text) {
    const words = tokenizeWords(text);
    const ids = [];
    const known = [];
    for (const w of words) {
      const i = this.index.get(w);
      const id = i === undefined || i >= this.meta.vocab_size ? this.oov : i;
      ids.push(id);
      known.push(i !== undefined && i < this.meta.vocab_size);
    }
    return { words, ids, known, truncated: ids.length > this.maxlen };
  }

  /**
   * One LSTM direction over `x` ([T, D] row-major).
   * Writes hidden states into `out` ([T, H]); returns nothing.
   */
  _lstm(x, D, T, Wih, Whh, b, reverse, out) {
    const H = this.H, H2 = 2 * H, H3 = 3 * H;
    const g = this._gates, h = this._h, c = this._c;
    h.fill(0); c.fill(0);
    for (let step = 0; step < T; step++) {
      const t = reverse ? T - 1 - step : step;
      const xo = t * D;
      for (let r = 0; r < 4 * H; r++) {
        let s = b[r];
        const wo = r * D;
        for (let k = 0; k < D; k++) s += Wih[wo + k] * x[xo + k];
        const ho = r * H;
        for (let k = 0; k < H; k++) s += Whh[ho + k] * h[k];
        g[r] = s;
      }
      for (let k = 0; k < H; k++) {
        const i = 1 / (1 + Math.exp(-g[k]));
        const f = 1 / (1 + Math.exp(-g[H + k]));
        const gg = Math.tanh(g[H2 + k]);
        const o = 1 / (1 + Math.exp(-g[H3 + k]));
        const cc = f * c[k] + i * gg;
        c[k] = cc;
        h[k] = o * Math.tanh(cc);
      }
      out.set(h, t * H);
    }
  }

  /** Forward pass over a padded id array. Returns Float32Array of 6 probs. */
  forwardIds(ids) {
    const T = this.maxlen, E = this.E, H = this.H, p = this.p;
    const emb = this._emb;
    for (let t = 0; t < T; t++) {
      const src = ids[t] * E;
      for (let k = 0; k < E; k++) emb[t * E + k] = p.emb[src + k];
    }

    this._lstm(emb, E, T, p.l1f_ih, p.l1f_hh, p.l1f_b, false, this._seqA);
    this._lstm(emb, E, T, p.l1r_ih, p.l1r_hh, p.l1r_b, true, this._seqB);
    const h1 = this._h1;
    for (let t = 0; t < T; t++) {
      h1.set(this._seqA.subarray(t * H, t * H + H), t * 2 * H);
      h1.set(this._seqB.subarray(t * H, t * H + H), t * 2 * H + H);
    }

    this._lstm(h1, 2 * H, T, p.l2f_ih, p.l2f_hh, p.l2f_b, false, this._seqA);
    this._lstm(h1, 2 * H, T, p.l2r_ih, p.l2r_hh, p.l2r_b, true, this._seqB);

    // Keras Bidirectional(return_sequences=False): forward state at t=T-1
    // concatenated with the backward state at t=0.
    const feat = new Float32Array(2 * H);
    feat.set(this._seqA.subarray((T - 1) * H, T * H), 0);
    feat.set(this._seqB.subarray(0, H), H);

    const nC = this.classes.length;
    const logits = new Float32Array(nC);
    for (let c = 0; c < nC; c++) {
      let s = p.fc_b[c];
      for (let k = 0; k < 2 * H; k++) s += p.fc_w[c * 2 * H + k] * feat[k];
      logits[c] = s;
    }
    let mx = -Infinity;
    for (let c = 0; c < nC; c++) mx = Math.max(mx, logits[c]);
    let sum = 0;
    const out = new Float32Array(nC);
    for (let c = 0; c < nC; c++) { out[c] = Math.exp(logits[c] - mx); sum += out[c]; }
    for (let c = 0; c < nC; c++) out[c] /= sum;
    return out;
  }

  _pad(ids) {
    const buf = this._ids;
    buf.fill(0);
    const n = Math.min(ids.length, this.maxlen);
    for (let i = 0; i < n; i++) buf[i] = ids[i];
    return buf;
  }

  predict(text) {
    const enc = this.encode(text);
    return { ...enc, probs: Array.from(this.forwardIds(this._pad(enc.ids))) };
  }

  /**
   * Full analysis: prediction, leave-one-out token attribution, and the
   * belief trajectory as the model reads the sentence prefix by prefix.
   */
  analyze(text, { attribution = true, trajectory = true } = {}) {
    const t0 = performance.now();
    const enc = this.encode(text);
    const n = Math.min(enc.ids.length, this.maxlen);
    const probs = Array.from(this.forwardIds(this._pad(enc.ids)));
    let passes = 1;
    const top = probs.indexOf(Math.max(...probs));

    // Attribution is reported in log-odds, not raw probability. Once the
    // softmax saturates at 0.99 every Δp collapses toward zero and the
    // heat-map goes flat; log-odds keeps resolving which token did the work.
    const logit = (p) => Math.log(Math.min(1 - 1e-9, Math.max(1e-9, p)) /
                                  (1 - Math.min(1 - 1e-9, Math.max(1e-9, p))));
    let attributions = null;
    let deltaP = null;
    if (attribution && n > 0 && n <= this.maxlen) {
      attributions = new Array(n);
      deltaP = new Array(n);
      const base = logit(probs[top]);
      for (let i = 0; i < n; i++) {
        const held = enc.ids.slice(0, n);
        held.splice(i, 1);                       // leave-one-out occlusion
        const q = this.forwardIds(this._pad(held));
        passes++;
        attributions[i] = base - logit(q[top]);
        deltaP[i] = probs[top] - q[top];
      }
    }

    let trace = null;
    if (trajectory && n > 0) {
      trace = new Array(n);
      for (let i = 1; i <= n; i++) {
        trace[i - 1] = Array.from(this.forwardIds(this._pad(enc.ids.slice(0, i))));
        passes++;
      }
    }

    return {
      words: enc.words, known: enc.known, truncated: enc.truncated,
      probs, top, attributions, deltaP, trace, passes,
      ms: performance.now() - t0,
    };
  }

  /** Cosine nearest neighbours in the learned embedding space. */
  neighbours(word, k = 12) {
    const i = this.index.get(word.toLowerCase());
    if (i === undefined || i >= this.meta.vocab_size) return null;
    const E = this.E, emb = this.p.emb;
    const norm = (o) => {
      let s = 0;
      for (let d = 0; d < E; d++) s += emb[o + d] * emb[o + d];
      return Math.sqrt(s) || 1e-9;
    };
    const oi = i * E, ni = norm(oi);
    const scored = [];
    for (let j = 2; j < this.meta.vocab_size; j++) {
      if (j === i || !this.vocab[j]) continue;
      const oj = j * E;
      let dot = 0;
      for (let d = 0; d < E; d++) dot += emb[oi + d] * emb[oj + d];
      scored.push([dot / (ni * norm(oj)), this.vocab[j]]);
    }
    scored.sort((a, b) => b[0] - a[0]);
    return { word: this.vocab[i], rank: i,
             neighbours: scored.slice(0, k).map(([s, w]) => ({ word: w, sim: s })) };
  }

  /**
   * Single-pass latency, measured rather than asserted. `performance.now()`
   * is deliberately coarsened in workers, so the headline number is the total
   * wall time over `n` runs divided by `n` — that stays meaningful even when
   * an individual pass rounds to zero.
   */
  benchmark(text, n = 30) {
    const ids = this._pad(this.encode(text).ids);
    for (let i = 0; i < 5; i++) this.forwardIds(ids);   // warm the JIT
    const times = [];
    const start = performance.now();
    for (let i = 0; i < n; i++) {
      const t0 = performance.now();
      this.forwardIds(ids);
      times.push(performance.now() - t0);
    }
    const total = performance.now() - start;
    times.sort((a, b) => a - b);
    return { mean: total / n, median: times[Math.floor(n / 2)], min: times[0], total, runs: n };
  }
}
