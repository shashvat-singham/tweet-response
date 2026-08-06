/**
 * Asserts that the browser engine reproduces the Python reference forward
 * pass bit-for-bit-ish. `ml/export.py` writes docs/model/golden.json from the
 * numpy implementation; this replays those inputs through the shipped
 * JavaScript and fails the build if any probability drifts past 1e-4.
 *
 *   node ml/parity.mjs
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { EmotionModel } from '../docs/assets/engine.js';

const here = dirname(fileURLToPath(import.meta.url));
const modelDir = join(here, '..', 'docs', 'model');
const read = (f) => readFileSync(join(modelDir, f));
const json = (f) => JSON.parse(read(f).toString('utf8'));

const meta = json('model.json');
const buf = read('weights.bin');
const weights = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
const model = new EmotionModel(meta, weights, json('vocab.json'));
const golden = json('golden.json');

const TOL = 1e-4;
let worst = 0;
let failed = 0;

for (const g of golden) {
  const { probs } = model.predict(g.text);
  const dev = Math.max(...probs.map((p, i) => Math.abs(p - g.probs[i])));
  worst = Math.max(worst, dev);
  const ok = dev < TOL;
  if (!ok) failed++;
  console.log(`${ok ? 'ok  ' : 'FAIL'}  Δ=${dev.toExponential(2)}  ${g.text.slice(0, 58)}`);
}

const sum = model.predict(golden[0].text).probs.reduce((a, b) => a + b, 0);
if (Math.abs(sum - 1) > 1e-5) { console.error(`FAIL  softmax sums to ${sum}`); failed++; }

console.log(`\n${golden.length} vectors · worst deviation ${worst.toExponential(2)} · tolerance ${TOL}`);
if (failed) { console.error(`${failed} check(s) failed`); process.exit(1); }
console.log('parity ok');
