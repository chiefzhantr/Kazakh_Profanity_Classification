"""Step 3 - fine-tune XLM-RoBERTa-base: review -> 0 (clean) / 1 (profanity).

* after every epoch: F1 of class 1 on VAL; only the best epoch is kept
* then the probability threshold is tuned on VAL
* TEST is not touched here (evaluate.py, once)

Usage:  DEVICE=cpu python src/train.py
Output: model/xlmr-kazakh-profanity/  (weights, tokenizer, inference_config.json)
        results/train_log.json
"""
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, get_linear_schedule_with_warmup)

SEED = 42
MODEL_NAME = "FacebookAI/xlm-roberta-base"
MAX_LENGTH = 128        # tokens; longer reviews are cut (the median review is 6 words)
BATCH_SIZE = 16
EPOCHS = 4
LR = 2e-5               # small: we only adjust a model that already knows the languages
WEIGHT_DECAY = 0.01
WARMUP = 0.1            # first 10% of steps: lr grows from 0, then falls linearly to 0
MAX_GRAD_NORM = 1.0
OUT_DIR = "model/xlmr-kazakh-profanity"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    # DEVICE=cpu: Apple MPS crashes on this Mac (macOS 14.3 + torch 2.15); Colab -> cuda
    if os.environ.get("DEVICE"):
        return torch.device(os.environ["DEVICE"])
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReviewDataset(torch.utils.data.Dataset):
    """Text -> token ids. Padding happens later, per batch (DataCollatorWithPadding)."""

    def __init__(self, texts, labels, tokenizer):
        self.enc = tokenizer(list(texts), truncation=True, max_length=MAX_LENGTH)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = int(self.labels[i])
        return item


@torch.no_grad()
def predict_proba(model, loader, device) -> np.ndarray:
    """P(label = 1) for every review."""
    model.eval()
    probs = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        probs.append(torch.softmax(model(**batch).logits, dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


def scores(y, probs, threshold=0.5) -> dict:
    pred = (probs >= threshold).astype(int)
    return {"precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0)}


def tune_threshold(y, probs):
    """Threshold 0.05..0.95 with the best VAL F1 (on a tie: the one closest to 0.5)."""
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    f1s = [scores(y, probs, t)["f1"] for t in grid]
    best = max(range(len(grid)), key=lambda i: (round(f1s[i], 6), -abs(grid[i] - 0.5)))
    return float(grid[best]), float(f1s[best])


def main():
    set_seed(SEED)
    device = get_device()
    print("device:", device)
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")

    # pre-trained tokenizer (250k sub-words), NOT fitted on our data -> no leakage
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # the pre-trained body + a NEW random classification head with 2 outputs
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2,
        id2label={0: "clean", 1: "profanity"}, label2id={"clean": 0, "profanity": 1}).to(device)

    collator = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(ReviewDataset(train_df["text"], train_df["label"], tokenizer),
                              batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator,
                              generator=torch.Generator().manual_seed(SEED))
    val_loader = DataLoader(ReviewDataset(val_df["text"], val_df["label"], tokenizer),
                            batch_size=64, collate_fn=collator)

    loss_fn = torch.nn.CrossEntropyLoss()      # classes are ~40/60, no weighting needed
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, int(WARMUP * total), total)

    log, best_f1 = [], -1.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0, running = time.time(), 0.0
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            loss = loss_fn(model(**batch).logits, labels)
            optimizer.zero_grad()
            loss.backward()                                             # gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)  # no huge jumps
            optimizer.step()                                            # update weights
            scheduler.step()                                            # update lr
            running += loss.item()
            if step % 20 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} loss {running / step:.4f}",
                      flush=True)

        val = scores(val_df["label"].values, predict_proba(model, val_loader, device))
        entry = {"epoch": epoch, "train_loss": round(running / len(train_loader), 4),
                 **{f"val_{k}": round(v, 4) for k, v in val.items()},
                 "minutes": round((time.time() - t0) / 60, 1)}
        log.append(entry)
        print(entry, flush=True)
        if val["f1"] > best_f1:                    # keep only the best epoch
            best_f1 = val["f1"]
            model.save_pretrained(OUT_DIR)
            tokenizer.save_pretrained(OUT_DIR)
            print(f"  saved best model (val F1 {best_f1:.4f})", flush=True)

    # threshold, with the BEST saved epoch
    model = AutoModelForSequenceClassification.from_pretrained(OUT_DIR).to(device)
    val_probs = predict_proba(model, val_loader, device)
    threshold, thr_f1 = tune_threshold(val_df["label"].values, val_probs)
    print(f"threshold {threshold}: val F1 {thr_f1:.4f} (at 0.5: {best_f1:.4f})")

    with open(f"{OUT_DIR}/inference_config.json", "w") as f:
        json.dump({"threshold": threshold, "max_length": MAX_LENGTH,
                   "base_model": MODEL_NAME, "seed": SEED}, f, indent=2)
    os.makedirs("results", exist_ok=True)
    with open("results/train_log.json", "w") as f:
        json.dump({"model": MODEL_NAME, "seed": SEED, "epochs": EPOCHS, "lr": LR,
                   "batch_size": BATCH_SIZE, "max_length": MAX_LENGTH, "device": str(device),
                   "log": log, "best_val_f1_at_0.5": best_f1, "threshold": threshold,
                   "val_at_threshold": scores(val_df["label"].values, val_probs, threshold)},
                  f, indent=2)


if __name__ == "__main__":
    main()
