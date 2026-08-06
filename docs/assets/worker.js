/**
 * Inference worker. Keeps the ~40 forward passes behind every keystroke
 * (prediction + leave-one-out attribution + prefix trajectory) off the main
 * thread so typing never drops a frame.
 */
import { EmotionModel } from './engine.js';

let model = null;

self.onmessage = async (e) => {
  const { id, op, payload } = e.data;
  try {
    if (op === 'load') {
      model = await EmotionModel.load(payload.base);
      self.postMessage({ id, ok: true, result: {
        vocab: model.meta.vocab_size, bytes: model.meta.total_bytes,
        classes: model.classes,
      } });
    } else if (op === 'analyze') {
      self.postMessage({ id, ok: true, result: model.analyze(payload.text, payload.opts) });
    } else if (op === 'neighbours') {
      self.postMessage({ id, ok: true, result: model.neighbours(payload.word, payload.k) });
    } else if (op === 'benchmark') {
      self.postMessage({ id, ok: true, result: model.benchmark(payload.text, payload.n) });
    }
  } catch (err) {
    self.postMessage({ id, ok: false, error: String(err && err.message || err) });
  }
};
