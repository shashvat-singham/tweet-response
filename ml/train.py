"""Train the tweet-emotion BiLSTM.

`--config baseline` reproduces the original Keras notebook layer-for-layer
(Embedding(10k, 16) -> BiLSTM(20, seq) -> BiLSTM(20) -> Dense(6, softmax),
Adam, batch 32, early stopping on val accuracy with patience 2) in PyTorch,
trained on the same 16k/2k/2k dair-ai `emotion` split.

`--config scaled` trains the same topology on the 416k-row merged corpus
shipped in `Dataset/`, after removing every text that appears in the
validation or test split, and evaluates on the identical held-out test set.

Both runs write a checkpoint + history JSON to `ml/artifacts/`.
"""

import argparse
import hashlib
import io
import json
import os
import time
import zipfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from tokenizer import Tokenizer, pad_sequences

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ART = os.path.join(HERE, "artifacts")
os.makedirs(ART, exist_ok=True)

# dair-ai/emotion integer label order.
CLASSES = ["sadness", "joy", "love", "anger", "fear", "surprise"]

CONFIGS = {
    "baseline": dict(
        vocab=10000, emb=16, hidden=20, maxlen=50, batch=32,
        epochs=15, patience=2, lr=1e-3, dropout=0.0, corpus="split", seed=1337,
    ),
    "scaled": dict(
        vocab=10000, emb=16, hidden=20, maxlen=50, batch=256,
        epochs=12, patience=2, lr=2e-3, dropout=0.1, corpus="merged", seed=1337,
    ),
}


def keras_init(module):
    """Match Keras' default initialisers.

    PyTorch defaults differ enough to change the optimisation trajectory:
    `nn.Embedding` samples from N(0, 1) where Keras uses U(-0.05, 0.05), and
    Keras seeds LSTM recurrent kernels orthogonally with a unit forget-gate
    bias. Without this the same architecture trains visibly slower.
    """
    for m in module.modules():
        if isinstance(m, nn.Embedding):
            nn.init.uniform_(m.weight, -0.05, 0.05)
        elif isinstance(m, nn.LSTM):
            for name, p in m.named_parameters():
                if name.startswith("weight_ih"):
                    nn.init.xavier_uniform_(p)
                elif name.startswith("weight_hh"):
                    h = p.shape[1]
                    for gate in range(4):  # i, f, g, o — orthogonal per gate
                        nn.init.orthogonal_(p.data[gate * h:(gate + 1) * h])
                elif name.startswith("bias"):
                    nn.init.zeros_(p)
                    h = p.shape[0] // 4
                    if name.startswith("bias_ih"):
                        p.data[h:2 * h] = 1.0  # unit_forget_bias
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    return module


class EmotionRNN(nn.Module):
    """Keras-equivalent forward pass.

    No masking (the original model had none), so padding is fed through the
    recurrence. The second bidirectional layer returns the forward state at
    t=T-1 concatenated with the backward state at t=0, which is exactly what
    `Bidirectional(LSTM(..., return_sequences=False))` emits.
    """

    def __init__(self, vocab, emb, hidden, dropout=0.0, n_classes=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab, emb, padding_idx=None)
        self.lstm1 = nn.LSTM(emb, hidden, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(2 * hidden, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout) if dropout else nn.Identity()
        self.fc = nn.Linear(2 * hidden, n_classes)
        self.hidden = hidden

    def forward(self, x):
        h = self.embedding(x)
        h, _ = self.lstm1(h)
        h = self.drop(h)
        h, _ = self.lstm2(h)
        last = torch.cat([h[:, -1, : self.hidden], h[:, 0, self.hidden:]], dim=1)
        return self.fc(self.drop(last))


def load_splits():
    d = os.path.join(HERE, "data")
    return (
        pd.read_parquet(os.path.join(d, "train.parquet")),
        pd.read_parquet(os.path.join(d, "validation.parquet")),
        pd.read_parquet(os.path.join(d, "test.parquet")),
    )


def load_merged():
    z = zipfile.ZipFile(os.path.join(ROOT, "Dataset", "merged_training.zip"))
    df = pd.read_pickle(io.BytesIO(z.read("merged_training.pkl")))
    df = df.rename(columns={"emotions": "emotion"})
    df["label"] = df["emotion"].map({c: i for i, c in enumerate(CLASSES)})
    return df.dropna(subset=["label"]).astype({"label": int})[["text", "label"]]


def evaluate(model, X, y, batch=512, device="cpu"):
    model.eval()
    logits = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            logits.append(model(X[i:i + batch].to(device)).cpu())
    logits = torch.cat(logits)
    loss = nn.functional.cross_entropy(logits, y).item()
    acc = (logits.argmax(1) == y).float().mean().item()
    return loss, acc, logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="baseline", choices=list(CONFIGS))
    args = ap.parse_args()
    cfg = CONFIGS[args.config]

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_df, val_df, test_df = load_splits()

    if cfg["corpus"] == "merged":
        merged = load_merged()
        before = len(merged)
        held_out = set(val_df.text) | set(test_df.text)
        merged = merged[~merged.text.isin(held_out)]
        merged = merged.drop_duplicates(subset="text")
        print(f"merged corpus: {before} -> {len(merged)} rows "
              f"after removing eval texts + exact duplicates")
        fit_df = merged
    else:
        fit_df = train_df

    tok = Tokenizer(num_words=cfg["vocab"], oov_token="<UNK>")
    tok.fit_on_texts(fit_df.text.tolist())

    def prep(df):
        X = pad_sequences(tok.texts_to_sequences(df.text.tolist()), cfg["maxlen"])
        return (torch.tensor(X, dtype=torch.long),
                torch.tensor(df.label.values, dtype=torch.long))

    Xtr, ytr = prep(fit_df)
    Xva, yva = prep(val_df)
    Xte, yte = prep(test_df)

    model = keras_init(
        EmotionRNN(cfg["vocab"], cfg["emb"], cfg["hidden"], cfg["dropout"])
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    # Keras counts one bias vector per LSTM direction; PyTorch stores two.
    keras_params = n_params - sum(
        m.bias_hh_l0.numel() + m.bias_hh_l0_reverse.numel()
        for m in [model.lstm1, model.lstm2]
    )
    print(f"params (torch)={n_params}  params (keras-equivalent)={keras_params}")

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    lossf = nn.CrossEntropyLoss()

    history = {"accuracy": [], "loss": [], "val_accuracy": [], "val_loss": [], "seconds": []}
    best_acc, best_state, wait = -1.0, None, 0
    t_all = time.time()

    for epoch in range(cfg["epochs"]):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(Xtr))
        tot_loss = tot_correct = 0
        for i in range(0, len(perm), cfg["batch"]):
            idx = perm[i:i + cfg["batch"]]
            xb, yb = Xtr[idx].to(device), ytr[idx].to(device)
            opt.zero_grad()
            out = model(xb)
            loss = lossf(out, yb)
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(idx)
            tot_correct += (out.argmax(1) == yb).sum().item()

        tr_loss = tot_loss / len(Xtr)
        tr_acc = tot_correct / len(Xtr)
        va_loss, va_acc, _ = evaluate(model, Xva, yva, device=device)
        dt = time.time() - t0

        history["loss"].append(round(tr_loss, 4))
        history["accuracy"].append(round(tr_acc, 4))
        history["val_loss"].append(round(va_loss, 4))
        history["val_accuracy"].append(round(va_acc, 4))
        history["seconds"].append(round(dt, 1))
        print(f"epoch {epoch+1}/{cfg['epochs']} - {dt:.0f}s - loss {tr_loss:.4f} "
              f"acc {tr_acc:.4f} - val_loss {va_loss:.4f} val_acc {va_acc:.4f}")

        if va_acc > best_acc:
            best_acc, wait = va_acc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg["patience"]:
                print(f"early stopping (no val_accuracy gain for {cfg['patience']} epochs)")
                break

    model.load_state_dict(best_state)
    te_loss, te_acc, te_logits = evaluate(model, Xte, yte, device=device)
    print(f"TEST  loss {te_loss:.4f}  accuracy {te_acc:.4f}")

    tag = args.config
    torch.save({"state_dict": best_state, "cfg": cfg}, os.path.join(ART, f"{tag}.pt"))
    with open(os.path.join(ART, f"{tag}_vocab.json"), "w", encoding="utf-8") as f:
        json.dump(tok.vocab(cfg["vocab"]), f)
    np.save(os.path.join(ART, f"{tag}_test_logits.npy"), te_logits.numpy())
    with open(os.path.join(ART, f"{tag}_run.json"), "w", encoding="utf-8") as f:
        json.dump({
            "config_name": tag,
            "config": cfg,
            "params_keras_equivalent": keras_params,
            "train_rows": len(Xtr),
            "epochs_trained": len(history["loss"]),
            "wall_seconds": round(time.time() - t_all, 1),
            "history": history,
            "val_accuracy": round(best_acc, 4),
            "test_accuracy": round(te_acc, 4),
            "test_loss": round(te_loss, 4),
            "device": device,
            "vocab_hash": hashlib.sha1(
                json.dumps(tok.vocab(cfg["vocab"])).encode()).hexdigest()[:12],
        }, f, indent=2)
    print("wrote artifacts to", ART)


if __name__ == "__main__":
    main()
