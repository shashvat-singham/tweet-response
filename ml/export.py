"""Evaluate a trained checkpoint and export everything the web app needs.

Produces, under `docs/model/`:
  weights.bin   int8 symmetric-quantised weights, concatenated
  model.json    tensor layout + per-tensor scales + architecture metadata
  vocab.json    index -> token, identical to the training tokenizer
  metrics.json  full evaluation report (confusion, per-class P/R/F1,
                calibration, coverage curve, error slices, token log-odds,
                embedding PCA, dataset statistics, training history)

Everything in metrics.json is computed here from the actual held-out test
split; nothing in the site is hand-written.
"""

import argparse
import json
import math
import os
import zipfile
import io
from collections import Counter

import numpy as np
import pandas as pd
import torch

from tokenizer import Tokenizer, pad_sequences, text_to_word_sequence
from train import CLASSES, EmotionRNN, load_splits, HERE, ROOT, ART

OUT = os.path.join(ROOT, "docs", "model")
os.makedirs(OUT, exist_ok=True)


# --------------------------------------------------------------------------
# numpy reference forward pass — mirrors the JS implementation exactly
# --------------------------------------------------------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def lstm_dir(x, W_ih, W_hh, b, reverse=False):
    """x: [B, T, D] -> [B, T, H]. PyTorch gate order i, f, g, o."""
    B, T, _ = x.shape
    H = W_hh.shape[1]
    h = np.zeros((B, H))
    c = np.zeros((B, H))
    outs = np.zeros((B, T, H))
    xz = x @ W_ih.T + b                                   # pre-project inputs
    for step in range(T):
        t = T - 1 - step if reverse else step
        z = xz[:, t] + h @ W_hh.T
        i = sigmoid(z[:, 0:H])
        f = sigmoid(z[:, H:2 * H])
        g = np.tanh(z[:, 2 * H:3 * H])
        o = sigmoid(z[:, 3 * H:4 * H])
        c = f * c + i * g
        h = o * np.tanh(c)
        outs[:, t] = h
    return outs


def forward_np(ids, P):
    """ids: [T] or [B, T] -> probabilities, same shape minus the time axis."""
    ids = np.asarray(ids)
    single = ids.ndim == 1
    if single:
        ids = ids[None, :]
    emb = P["emb"][ids]                                        # [B, T, E]
    f1 = lstm_dir(emb, P["l1f_ih"], P["l1f_hh"], P["l1f_b"], False)
    b1 = lstm_dir(emb, P["l1r_ih"], P["l1r_hh"], P["l1r_b"], True)
    h1 = np.concatenate([f1, b1], axis=2)                      # [B, T, 2h]
    f2 = lstm_dir(h1, P["l2f_ih"], P["l2f_hh"], P["l2f_b"], False)
    b2 = lstm_dir(h1, P["l2r_ih"], P["l2r_hh"], P["l2r_b"], True)
    last = np.concatenate([f2[:, -1], b2[:, 0]], axis=1)       # Keras Bidirectional
    logits = last @ P["fc_w"].T + P["fc_b"]
    e = np.exp(logits - logits.max(1, keepdims=True))
    p = e / e.sum(1, keepdims=True)
    return p[0] if single else p


def unpack(sd):
    def g(k):
        return sd[k].numpy().astype(np.float64)
    return {
        "emb": g("embedding.weight"),
        "l1f_ih": g("lstm1.weight_ih_l0"), "l1f_hh": g("lstm1.weight_hh_l0"),
        "l1f_b": g("lstm1.bias_ih_l0") + g("lstm1.bias_hh_l0"),
        "l1r_ih": g("lstm1.weight_ih_l0_reverse"), "l1r_hh": g("lstm1.weight_hh_l0_reverse"),
        "l1r_b": g("lstm1.bias_ih_l0_reverse") + g("lstm1.bias_hh_l0_reverse"),
        "l2f_ih": g("lstm2.weight_ih_l0"), "l2f_hh": g("lstm2.weight_hh_l0"),
        "l2f_b": g("lstm2.bias_ih_l0") + g("lstm2.bias_hh_l0"),
        "l2r_ih": g("lstm2.weight_ih_l0_reverse"), "l2r_hh": g("lstm2.weight_hh_l0_reverse"),
        "l2r_b": g("lstm2.bias_ih_l0_reverse") + g("lstm2.bias_hh_l0_reverse"),
        "fc_w": g("fc.weight"), "fc_b": g("fc.bias"),
    }


TENSOR_ORDER = ["emb",
                "l1f_ih", "l1f_hh", "l1f_b", "l1r_ih", "l1r_hh", "l1r_b",
                "l2f_ih", "l2f_hh", "l2f_b", "l2r_ih", "l2r_hh", "l2r_b",
                "fc_w", "fc_b"]


def quantize(P):
    """Symmetric int8 with a per-output-row scale.

    A single scale per tensor is ~4x cheaper to describe but loses badly on
    the embedding matrix, where a handful of high-norm rows dominate the
    dynamic range of 10,000 otherwise-small vectors. One float32 scale per
    row costs 40 KB and buys back most of the error. Biases stay float32.
    """
    blobs, meta, deq = [], [], {}
    offset = 0
    for name in TENSOR_ORDER:
        t = P[name]
        if name.endswith("_b"):
            raw = t.astype(np.float32)
            blobs.append(raw.tobytes())
            meta.append({"name": name, "shape": list(t.shape), "dtype": "f32",
                         "offset": offset, "bytes": raw.nbytes})
            offset += raw.nbytes
            deq[name] = raw.astype(np.float64)
        else:
            scales = np.abs(t).max(axis=1) / 127.0
            scales[scales == 0] = 1.0
            q = np.clip(np.round(t / scales[:, None]), -127, 127).astype(np.int8)
            s32 = scales.astype(np.float32)
            blobs.append(q.tobytes())
            blobs.append(s32.tobytes())
            meta.append({"name": name, "shape": list(t.shape), "dtype": "i8",
                         "offset": offset, "bytes": q.nbytes,
                         "scale_offset": offset + q.nbytes, "scale_count": len(s32)})
            offset += q.nbytes + s32.nbytes
            deq[name] = q.astype(np.float64) * scales[:, None]
    return b"".join(blobs), meta, deq


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def per_class_report(y, yhat, n=6):
    rows = []
    for c in range(n):
        tp = int(((yhat == c) & (y == c)).sum())
        fp = int(((yhat == c) & (y != c)).sum())
        fn = int(((yhat != c) & (y == c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append({"class": CLASSES[c], "precision": round(prec, 4),
                     "recall": round(rec, 4), "f1": round(f1, 4),
                     "support": int((y == c).sum()), "tp": tp, "fp": fp, "fn": fn})
    return rows


def calibration(probs, y, bins=10):
    conf = probs.max(1)
    correct = (probs.argmax(1) == y).astype(float)
    out, ece = [], 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        m = (conf > lo) & (conf <= hi) if b else (conf >= lo) & (conf <= hi)
        n = int(m.sum())
        if n:
            acc, cf = float(correct[m].mean()), float(conf[m].mean())
            ece += n / len(y) * abs(acc - cf)
        else:
            acc = cf = 0.0
        out.append({"lo": round(lo, 2), "hi": round(hi, 2), "n": n,
                    "accuracy": round(acc, 4), "confidence": round(cf, 4)})
    return out, round(ece, 4)


def coverage_curve(probs, y, steps=41):
    conf = probs.max(1)
    correct = (probs.argmax(1) == y)
    order = np.argsort(-conf)
    c_sorted = correct[order]
    pts = []
    for k in range(1, steps):
        cov = k / (steps - 1)
        m = max(1, int(round(cov * len(y))))
        pts.append({"coverage": round(cov, 3),
                    "accuracy": round(float(c_sorted[:m].mean()), 4),
                    "threshold": round(float(conf[order][m - 1]), 4)})
    return pts


# Held out of the *displayed* ranking only; the model still trains on them.
DISPLAY_BLOCKLIST = {"fucked", "fucking", "fuck", "shit", "shitty", "bitch",
                     "horny", "damn", "ass", "crap", "crappy", "pissed"}


def log_odds_tokens(texts, labels, top_k=18, min_count=25):
    """Monroe et al. weighted log-odds with an informative Dirichlet prior."""
    counts = [Counter() for _ in CLASSES]
    total = Counter()
    for t, l in zip(texts, labels):
        for w in set(text_to_word_sequence(t)):
            counts[l][w] += 1
            total[w] += 1
    a0 = sum(total.values())
    n_all = sum(sum(c.values()) for c in counts)
    out = {}
    for ci, c in enumerate(counts):
        n_i = sum(c.values())
        scores = []
        for w, cnt in c.items():
            if total[w] < min_count or w in DISPLAY_BLOCKLIST:
                continue
            a_w = total[w]
            num = (cnt + a_w) / (n_i + a0 - cnt - a_w)
            den = (total[w] - cnt + a_w) / (n_all - n_i + a0 - (total[w] - cnt) - a_w)
            d = math.log(num) - math.log(den)
            var = 1.0 / (cnt + a_w) + 1.0 / (total[w] - cnt + a_w)
            scores.append((d / math.sqrt(var), w, cnt))
        scores.sort(reverse=True)
        out[CLASSES[ci]] = [{"token": w, "z": round(z, 2), "count": n}
                            for z, w, n in scores[:top_k]]
    return out


def demo_examples(test_df, y, yhat, probs, log_odds):
    """One readable, correctly-classified test tweet per emotion.

    Real held-out sentences rather than invented ones: a demo that quietly
    feeds the model out-of-distribution text is a demo that lies about it.
    Among the correct predictions we prefer sentences that actually carry the
    class's characteristic vocabulary, so the token attribution below has
    something to show.
    """
    picks = []
    for c, name in enumerate(CLASSES):
        marker = {t["token"] for t in log_odds[name][:20]}
        cand = [int(i) for i in np.where((y == c) & (yhat == c))[0]
                if 8 <= len(test_df.text.iloc[int(i)].split()) <= 16
                and 0.60 <= probs[i, c] <= 0.995]
        if not cand:
            cand = [int(i) for i in np.where((y == c) & (yhat == c))[0][:1]]

        def score(i):
            words = set(text_to_word_sequence(test_df.text.iloc[i]))
            return (len(words & marker), "i feel" in test_df.text.iloc[i], probs[i, c])

        i = max(cand, key=score)
        picks.append({"text": test_df.text.iloc[i], "gold": name,
                      "confidence": round(float(probs[i, c]), 3)})
    return picks


def pca_2d(X):
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    coords = Xc @ Vt[:2].T
    var = (S ** 2) / (S ** 2).sum()
    return coords, var[:2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="baseline")
    args = ap.parse_args()
    tag = args.config

    ckpt = torch.load(os.path.join(ART, f"{tag}.pt"), map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    run = json.load(open(os.path.join(ART, f"{tag}_run.json"), encoding="utf-8"))
    vocab = json.load(open(os.path.join(ART, f"{tag}_vocab.json"), encoding="utf-8"))

    model = EmotionRNN(cfg["vocab"], cfg["emb"], cfg["hidden"], cfg["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    P = unpack(ckpt["state_dict"])
    blob, meta, deq = quantize(P)

    train_df, val_df, test_df = load_splits()
    tok = Tokenizer(num_words=cfg["vocab"], oov_token="<UNK>")
    fit_texts = train_df.text.tolist()
    if cfg["corpus"] == "merged":
        z = zipfile.ZipFile(os.path.join(ROOT, "Dataset", "merged_training.zip"))
        m = pd.read_pickle(io.BytesIO(z.read("merged_training.pkl")))
        m = m[~m.text.isin(set(val_df.text) | set(test_df.text))].drop_duplicates("text")
        fit_texts = m.text.tolist()
    tok.fit_on_texts(fit_texts)
    assert tok.vocab(cfg["vocab"]) == vocab, "tokenizer drift vs training run"

    Xte = np.array(pad_sequences(tok.texts_to_sequences(test_df.text.tolist()), cfg["maxlen"]))
    yte = test_df.label.values

    with torch.no_grad():
        logits_f32 = model(torch.tensor(Xte, dtype=torch.long)).numpy()
    probs_f32 = softmax(logits_f32)
    acc_f32 = float((probs_f32.argmax(1) == yte).mean())

    # Exactly the arithmetic the browser runs: int8 weights, numpy reference.
    probs_i8 = forward_np(Xte, deq)
    acc_i8 = float((probs_i8.argmax(1) == yte).mean())
    max_dev = float(np.abs(probs_i8 - probs_f32).max())
    agree = float((probs_i8.argmax(1) == probs_f32.argmax(1)).mean())
    print(f"float32 acc {acc_f32:.4f} | int8 acc {acc_i8:.4f} | "
          f"argmax agreement {agree:.4f} | max |dp| {max_dev:.4f}")

    probs = probs_i8  # report what actually ships
    yhat = probs.argmax(1)
    cm = np.zeros((6, 6), dtype=int)
    for t, p in zip(yte, yhat):
        cm[t, p] += 1

    # confusion pairs with real examples
    pairs = []
    for a in range(6):
        for b in range(6):
            if a != b and cm[a, b]:
                idx = np.where((yte == a) & (yhat == b))[0]
                idx = idx[np.argsort(-probs[idx, b])][:3]
                pairs.append({
                    "actual": CLASSES[a], "predicted": CLASSES[b], "count": int(cm[a, b]),
                    "rate": round(float(cm[a, b] / cm[a].sum()), 4),
                    "examples": [{"text": test_df.text.iloc[int(i)],
                                  "confidence": round(float(probs[int(i), b]), 3)} for i in idx],
                })
    pairs.sort(key=lambda p: -p["count"])

    wrong = np.where(yhat != yte)[0]
    conf_wrong = probs[wrong].max(1)
    worst = wrong[np.argsort(-conf_wrong)][:8]
    right = np.where(yhat == yte)[0]

    lengths = [len(text_to_word_sequence(t)) for t in train_df.text]
    hist = Counter(min(l, 60) for l in lengths)
    seqs = tok.texts_to_sequences(test_df.text.tolist())
    oov_id = tok.word_index["<UNK>"]
    oov_rate = sum(s.count(oov_id) for s in seqs) / max(1, sum(len(s) for s in seqs))
    trunc = float(np.mean([len(s) > cfg["maxlen"] for s in seqs]))

    emb = deq["emb"]
    freq_ids = np.arange(2, min(1600, cfg["vocab"]))
    coords, var = pca_2d(emb[freq_ids])
    # class affinity of a token = mean predicted distribution when shown alone
    solo_ids = np.zeros((len(freq_ids), cfg["maxlen"]), dtype=int)
    solo_ids[:, 0] = freq_ids
    solo = forward_np(solo_ids, deq)
    scatter = [{"w": vocab[int(i)], "x": round(float(coords[k, 0]), 3),
                "y": round(float(coords[k, 1]), 3), "c": int(solo[k].argmax()),
                "p": round(float(solo[k].max()), 3)}
               for k, i in enumerate(freq_ids)]

    dataset_stats = {
        "splits": [{"name": n, "rows": int(len(d)),
                    "counts": [int((d.label == c).sum()) for c in range(6)]}
                   for n, d in [("train", train_df), ("validation", val_df), ("test", test_df)]],
        "length_histogram": [{"len": k, "n": hist[k]} for k in sorted(hist)],
        "length_percentiles": {p: int(np.percentile(lengths, p)) for p in (50, 90, 95, 99)},
        "unique_tokens": len(tok.word_index) - 1,
        "vocab_cutoff": cfg["vocab"],
        "coverage_at_cutoff": round(
            sum(c for w, c in tok.word_counts.items()
                if tok.word_index[w] < cfg["vocab"]) / sum(tok.word_counts.values()), 4),
        "test_oov_rate": round(float(oov_rate), 4),
        "test_truncation_rate": round(trunc, 4),
        "source": "dair-ai/emotion (Saravia et al., EMNLP 2018)",
    }

    log_odds = log_odds_tokens(train_df.text.tolist(), train_df.label.values)
    cal, ece = calibration(probs, yte)
    report = per_class_report(yte, yhat)
    macro_f1 = round(float(np.mean([r["f1"] for r in report])), 4)
    weighted_f1 = round(float(np.average([r["f1"] for r in report],
                                         weights=[r["support"] for r in report])), 4)

    metrics = {
        "model": {
            "name": tag,
            "architecture": [
                {"layer": "Embedding", "output": f"(50, {cfg['emb']})",
                 "params": cfg["vocab"] * cfg["emb"],
                 "detail": f"vocab {cfg['vocab']:,} · {cfg['emb']}-d, learned from scratch"},
                {"layer": "Bidirectional LSTM", "output": f"(50, {2*cfg['hidden']})",
                 "params": 2 * 4 * cfg["hidden"] * (cfg["emb"] + cfg["hidden"] + 1),
                 "detail": f"{cfg['hidden']} units per direction · return_sequences"},
                {"layer": "Bidirectional LSTM", "output": f"({2*cfg['hidden']},)",
                 "params": 2 * 4 * cfg["hidden"] * (2 * cfg["hidden"] + cfg["hidden"] + 1),
                 "detail": f"{cfg['hidden']} units per direction · final state"},
                {"layer": "Dense", "output": "(6,)",
                 "params": 2 * cfg["hidden"] * 6 + 6, "detail": "softmax over 6 emotions"},
            ],
            "params": run["params_keras_equivalent"],
            "config": cfg,
            "train_rows": run["train_rows"],
            "epochs_trained": run["epochs_trained"],
            "wall_seconds": run["wall_seconds"],
            "device": run["device"],
            "weights_bytes": len(blob),
        },
        "history": run["history"],
        "headline": {
            "test_accuracy": round(acc_i8, 4),
            "test_accuracy_float32": round(acc_f32, 4),
            "val_accuracy": run["val_accuracy"],
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "ece": ece,
            "int8_argmax_agreement": round(agree, 4),
            "int8_max_prob_deviation": round(max_dev, 4),
            "reference_keras_accuracy": 0.888,
            "majority_class_baseline": round(float((yte == np.bincount(yte).argmax()).mean()), 4),
        },
        "classes": CLASSES,
        "per_class": report,
        "confusion": cm.tolist(),
        "confusion_pairs": pairs[:8],
        "confident_errors": [
            {"text": test_df.text.iloc[int(i)], "actual": CLASSES[int(yte[i])],
             "predicted": CLASSES[int(yhat[i])], "confidence": round(float(probs[i].max()), 3)}
            for i in worst],
        "confident_hits": [
            {"text": test_df.text.iloc[int(i)], "actual": CLASSES[int(yte[i])],
             "confidence": round(float(probs[i].max()), 3)}
            for i in right[np.argsort(-probs[right].max(1))][:4]],
        "calibration": cal,
        "coverage": coverage_curve(probs, yte),
        "token_log_odds": log_odds,
        "embedding_pca": {"variance": [round(float(v), 4) for v in var], "points": scatter},
        "dataset": dataset_stats,
        "demo_examples": demo_examples(test_df, yte, yhat, probs, log_odds),
    }

    with open(os.path.join(OUT, "weights.bin"), "wb") as f:
        f.write(blob)
    with open(os.path.join(OUT, "model.json"), "w", encoding="utf-8") as f:
        json.dump({"maxlen": cfg["maxlen"], "vocab_size": cfg["vocab"],
                   "emb": cfg["emb"], "hidden": cfg["hidden"], "classes": CLASSES,
                   "gate_order": "ifgo", "tensors": meta,
                   "total_bytes": len(blob)}, f)
    with open(os.path.join(OUT, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f)
    with open(os.path.join(OUT, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f)

    # Golden vectors so the JS port can be asserted against this reference —
    # checked live on the page and in CI via `node ml/parity.mjs`.
    golden = []
    for t in ([d["text"] for d in metrics["demo_examples"]]
              + [test_df.text.iloc[i] for i in range(3)]):
        ids = pad_sequences(tok.texts_to_sequences([t]), cfg["maxlen"])[0]
        golden.append({"text": t,
                       "probs": [round(float(v), 8) for v in forward_np(np.array(ids), deq)]})
    with open(os.path.join(OUT, "golden.json"), "w", encoding="utf-8") as f:
        json.dump(golden, f, indent=2)

    print(f"wrote {OUT}  (weights {len(blob)/1024:.0f} KB, "
          f"vocab {os.path.getsize(os.path.join(OUT,'vocab.json'))/1024:.0f} KB, "
          f"metrics {os.path.getsize(os.path.join(OUT,'metrics.json'))/1024:.0f} KB)")


if __name__ == "__main__":
    main()
