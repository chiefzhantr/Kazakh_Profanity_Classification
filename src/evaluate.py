"""Step 4 - evaluate the final model ONCE on the test set.

Usage:  DEVICE=cpu python src/evaluate.py
Output: results/test_metrics.json, results/confusion_matrix.png, results/test_errors.csv
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from train import OUT_DIR, ReviewDataset, get_device, predict_proba, scores


def main():
    device = get_device()
    threshold = json.load(open(f"{OUT_DIR}/inference_config.json"))["threshold"]   # from VAL
    tokenizer = AutoTokenizer.from_pretrained(OUT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(OUT_DIR).to(device)

    test = pd.read_csv("data/processed/test.csv")
    loader = DataLoader(ReviewDataset(test["text"], test["label"], tokenizer),
                        batch_size=64, collate_fn=DataCollatorWithPadding(tokenizer))
    test["prob"] = predict_proba(model, loader, device).round(4)
    test["pred"] = (test["prob"] >= threshold).astype(int)

    y, pred = test["label"].values, test["pred"].values
    cm = confusion_matrix(y, pred, labels=[0, 1])          # rows = true, columns = predicted
    tn, fp, fn, tp = cm.ravel()
    metrics = {"n_test": len(test), "threshold": threshold,
               "class_1": {k: round(v, 4) for k, v in scores(y, test["prob"].values, threshold).items()},
               "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
               "report": classification_report(y, pred, target_names=["clean", "profanity"],
                                               output_dict=True, digits=4)}
    print(json.dumps({k: metrics[k] for k in ["n_test", "threshold", "class_1", "confusion_matrix"]},
                     indent=2))
    print(classification_report(y, pred, target_names=["clean", "profanity"], digits=4))

    os.makedirs("results", exist_ok=True)
    with open("results/test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["clean (0)", "profanity (1)"])
    ax.set_yticks([0, 1], ["clean (0)", "profanity (1)"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Test confusion matrix (n={len(test)}, thr={threshold})")
    fig.tight_layout()
    fig.savefig("results/confusion_matrix.png", dpi=150)

    errors = test[test["label"] != test["pred"]].copy()
    errors["error"] = errors["label"].map({0: "FP (clean flagged)", 1: "FN (profanity missed)"})
    errors.sort_values("prob")[["error", "prob", "label", "pred", "pool", "id", "text"]].to_csv(
        "results/test_errors.csv", index=False)
    print(f"{len(errors)} errors -> results/test_errors.csv")


if __name__ == "__main__":
    main()
