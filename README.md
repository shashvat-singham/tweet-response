<div align="center">

# Twitter Emotion Recognition

**A 176k-parameter bidirectional LSTM that classifies tweets into six emotions —
trained in PyTorch, quantised to int8, and executed entirely in the browser.**

### → **[shashvat-singham.github.io/tweet-response](https://shashvat-singham.github.io/tweet-response/)** ←

[![Pages](https://github.com/shashvat-singham/tweet-response/actions/workflows/pages.yml/badge.svg)](https://github.com/shashvat-singham/tweet-response/actions/workflows/pages.yml)
![Test accuracy](https://img.shields.io/badge/test%20accuracy-88.0%25-14c8a6)
![Macro F1](https://img.shields.io/badge/macro%20F1-0.839-14c8a6)
![Params](https://img.shields.io/badge/parameters-175,926-5b8cff)
![Payload](https://img.shields.io/badge/model-214%20KB%20int8-5b8cff)
![Dependencies](https://img.shields.io/badge/runtime%20dependencies-0-lightgrey)

</div>

---

## What this is

The original version of this project was a Colab notebook: load `dair-ai/emotion`,
stack an embedding layer under two bidirectional LSTMs, train for ten epochs, print
88.8% test accuracy, plot a confusion matrix, done.

This repository keeps that model — the exact same four layers, the same 175,926
parameters — and takes it the rest of the way:

| | |
|---|---|
| **Reproduced** | The Keras model re-implemented in PyTorch, including Keras' initialisers, gate ordering and unmasked padding. 88.15% test accuracy against the notebook's 88.80%. |
| **Evaluated properly** | Per-class precision/recall/F1, a clickable confusion matrix backed by real test examples, a reliability diagram with ECE, and a risk–coverage curve for selective prediction. |
| **Quantised** | Symmetric int8 with per-output-row scales. 214 KB on the wire, 99.8% argmax agreement with the float32 checkpoint. |
| **Shipped** | The forward pass rewritten in ~90 lines of typed-array JavaScript. No TensorFlow.js, no ONNX runtime, no WASM, no API. Runs in a web worker. |
| **Verified** | `ml/export.py` emits golden probability vectors from a numpy reference; `ml/parity.mjs` replays them through the JavaScript engine in CI and fails the deploy on any drift past 1e-4. |

The [live site](https://shashvat-singham.github.io/tweet-response/) is the real
deliverable: type a sentence and watch the posterior, the per-token leave-one-out
attributions and the belief trajectory update on every keystroke — all computed on
your own machine.

---

## Results

Held-out test split, 2,000 tweets, int8 weights (the ones the browser actually loads).

| Metric | Value |
|---|---|
| Accuracy | **88.00%** (float32 checkpoint: 88.15%) |
| Macro F1 | 0.839 |
| Weighted F1 | 0.881 |
| Expected calibration error | 0.074 |
| Majority-class baseline | 34.8% |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| sadness | 0.915 | 0.924 | 0.919 | 581 |
| joy | 0.927 | 0.891 | 0.908 | 695 |
| love | 0.722 | 0.799 | 0.758 | 159 |
| anger | 0.873 | 0.847 | 0.860 | 275 |
| fear | 0.843 | 0.862 | 0.852 | 224 |
| surprise | 0.699 | 0.773 | 0.734 | 66 |

The gap between macro and weighted F1 is the whole story: **love** and **surprise**
together account for 11% of the test set and almost all of the lost accuracy. They
lose to **joy** and **fear** respectively, and they lose for a reason — the classes
share most of their vocabulary and the rare one has a quarter of the examples.
The site's error-analysis section walks through it with real misclassified tweets.

---

## Architecture

```
Input          (50,)      token ids, post-padded and post-truncated
  ↓
Embedding      (50, 16)   10,000-word vocabulary, learned from scratch   160,000
  ↓
BiLSTM         (50, 40)   20 units per direction, return_sequences          5,920
  ↓
BiLSTM         (40,)      20 units per direction, final state               9,760
  ↓
Dense          (6,)       softmax over six emotions                           246
                                                                     ─────────────
                                                                          175,926
```

Two details are preserved deliberately rather than "fixed":

- **Padding is not masked.** The original had no `mask_zero`, so all 50 timesteps
  flow through the recurrence. Masking would change the model, not just clean it up.
- **The second BiLSTM returns endpoints, not sequences.** Keras'
  `Bidirectional(..., return_sequences=False)` concatenates the forward state at
  `t=49` with the *backward* state at `t=0`. The PyTorch port and the JavaScript
  engine both replicate that, which is the easiest thing in the world to get wrong.

---

## Repository layout

```
docs/                 the published site (GitHub Pages serves this directory)
  index.html
  assets/
    engine.js         BiLSTM forward pass + Keras-compatible tokenizer, in plain JS
    worker.js         runs inference off the main thread
    charts.js         small SVG chart toolkit
    app.js            page controller — every number comes from metrics.json
    style.css
  model/
    weights.bin       int8 weights + per-row scales
    model.json        tensor layout and architecture metadata
    vocab.json        index → token, identical to the training tokenizer
    metrics.json      the full evaluation report
    golden.json       reference probability vectors for the parity test

ml/
  fetch_data.py       download the official 16k/2k/2k split
  tokenizer.py        faithful re-implementation of Keras' Tokenizer
  train.py            the model + training loop
  export.py           evaluation, quantisation, artefact emission
  parity.mjs          asserts the JS engine matches the numpy reference

Twitter Emotion Recognition using RNN.ipynb   the original notebook, unchanged
Dataset/merged_training.zip                   the 416k-row merged corpus
```

---

## Reproducing it

```bash
pip install -r ml/requirements.txt

python ml/fetch_data.py                  # official dair-ai/emotion split
python ml/train.py  --config baseline    # ~6 min on CPU → ml/artifacts/
python ml/export.py --config baseline    # evaluation + int8 export → docs/model/
node   ml/parity.mjs                     # JS engine vs numpy reference

python -m http.server 8000 --directory docs   # then open localhost:8000
```

`train.py` also accepts `--config scaled`, which trains the same topology on the
416k-row merged corpus in `Dataset/` after removing every text that appears in the
validation or test split.

---

## Data

[`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) — 20,000 English
tweets labelled with one of six emotions, split 16,000 / 2,000 / 2,000. From
Saravia et al., *CARER: Contextualized Affect Representations for Emotion
Recognition*, EMNLP 2018.

The labels are distant-supervised from hashtags, so they are noisy: several of the
model's highest-confidence "mistakes" are arguably mislabelled rather than
misclassified. The site shows them and says so.

---

## Limitations

This is a small model on a narrow corpus, and the site's model card states the
boundaries plainly: it is not for clinical, safety, hiring or moderation decisions;
it does not handle sarcasm, heavy negation, code-switching or any language but
English; and it is overconfident by construction — use the risk–coverage curve,
not the raw softmax, if anything depends on the answer.

---

## Licence

MIT — see [LICENSE](LICENSE).
