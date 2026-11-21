"""Step 2 - build the dataset from my own labels.

my_labels_round1.xlsx + to_label_round2.xlsx
  -> one table (text, label)  -> clean text  -> drop exact duplicates
  -> group near-duplicates    -> train / val / test = 80 / 10 / 10, seed 42
  -> data/processed/{train,val,test}.csv + split_info.json

Usage:  python src/prepare_data.py
"""
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from text_utils import normalize_text

SEED = 42
NEAR_DUP_JACCARD = 0.8          # texts this similar are "the same review" for the split
ROUND2_LAST_ROW = 2037          # I labeled Excel rows 2..2037; an empty cell there means 0
DROP_IDS = {"pla019716pla"}     # contains a person's full name -> removed (no personal data)
OUT = "data/processed"


# --- part 1: read my two labeling rounds --------------------------------------------------
def load_labels() -> pd.DataFrame:
    r1 = pd.read_excel("my_labels/my_labels_round1.xlsx")
    r1["label"] = pd.to_numeric(r1["human_label"], errors="coerce")
    r1 = r1[r1["label"].notna()].rename(columns={"kazsandra_id": "id"})
    r1["round"] = 1

    r2 = pd.read_excel("my_labels/to_label_round2.xlsx")
    r2 = r2.iloc[:ROUND2_LAST_ROW - 1].copy()                    # Excel row 2 = index 0
    r2["label"] = pd.to_numeric(r2["label"], errors="coerce").fillna(0)
    r2["round"] = 2

    cols = ["id", "domain", "pool", "round", "text", "label"]
    df = pd.concat([r1[cols], r2[cols]], ignore_index=True)
    df["label"] = df["label"].astype(int)
    assert df["label"].isin([0, 1]).all()
    return df[~df["id"].isin(DROP_IDS)]


# --- part 2: duplicates -------------------------------------------------------------------
def dedup_key(text: str) -> str:
    """'Қотақ ойын!!!' and 'қотақ ойын' -> same key: lowercase, letters and digits only."""
    return re.sub(r"[\W_]+", " ", text.lower()).strip()


def near_duplicate_groups(texts, threshold: float) -> np.ndarray:
    """Give every text a group id; texts with char-4-gram Jaccard >= threshold share one.

    Jaccard = |common 4-grams| / |all 4-grams|. Union-find joins chains of similar texts.
    """
    grams = []
    for t in texts:
        s = dedup_key(t)
        grams.append({s[i:i + 4] for i in range(max(1, len(s) - 3))})
    parent = list(range(len(texts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            a, b = grams[i], grams[j]
            if min(len(a), len(b)) < threshold * max(len(a), len(b)):
                continue                                   # sizes too different, skip fast
            if len(a & b) / len(a | b) >= threshold:
                parent[find(j)] = find(i)
    return np.array([find(i) for i in range(len(texts))])


# --- part 3: everything together ------------------------------------------------------------
def main():
    df = load_labels()
    report = {"labeled_rows": len(df), "per_round": df.groupby("round")["label"]
              .value_counts().unstack().to_dict(orient="index")}

    df["text"] = df["text"].map(normalize_text)
    df = df[df["text"].str.len() > 0]

    # exact duplicates: same text twice -> keep one; same text with labels 0 AND 1 -> drop both
    df["key"] = df["text"].map(dedup_key)
    conflicts = df.groupby("key")["label"].nunique()
    conflict_keys = set(conflicts[conflicts > 1].index)
    df = df[~df["key"].isin(conflict_keys)]
    before = len(df)
    df = df.drop_duplicates("key").reset_index(drop=True)
    report["exact_duplicates_removed"] = before - len(df)
    report["label_conflicts_removed"] = len(conflict_keys)

    df["group"] = near_duplicate_groups(df["text"].tolist(), NEAR_DUP_JACCARD)
    report["near_duplicate_groups"] = int((df["group"].value_counts() > 1).sum())

    # split: 10 folds, each with the same 0/1 ratio and no group cut in half
    #   fold 0 -> test (10%), fold 1 -> val (10%), the other 8 -> train (80%)
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED)
    folds = list(sgkf.split(df, df["label"], groups=df["group"]))
    df["split"] = "train"
    df.loc[folds[0][1], "split"] = "test"
    df.loc[folds[1][1], "split"] = "val"
    assert (df.groupby("group")["split"].nunique() == 1).all()     # no leak between splits

    os.makedirs(OUT, exist_ok=True)
    for name in ["train", "val", "test"]:
        part = df[df["split"] == name].sample(frac=1, random_state=SEED)
        part[["id", "domain", "pool", "round", "text", "label"]].to_csv(
            f"{OUT}/{name}.csv", index=False)
        report[name] = {"n": len(part), "label_0": int((part["label"] == 0).sum()),
                        "label_1": int((part["label"] == 1).sum())}
    report["seed"] = SEED
    report["final_rows"] = len(df)
    report["domains"] = df["domain"].value_counts().to_dict()
    report["words_per_review"] = df["text"].str.split().str.len().describe().round(1).to_dict()

    with open(f"{OUT}/split_info.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=int)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=int))


if __name__ == "__main__":
    main()
