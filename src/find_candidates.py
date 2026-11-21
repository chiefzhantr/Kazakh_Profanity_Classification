"""Step 1 - find candidate reviews to label.

Source of texts : KazSAnDRA (175k Kazakh reviews), data/kazsandra_full/
Search words    : Kazakh Swear Words dictionary by Rustem Kaimolla (Hugging Face)
                  + roots of Russian swear words (RUSSIAN_ROOTS, added by me)
Output          : my_labels/to_label_round2.xlsx  -> I label the `label` column by hand

Usage:  python src/find_candidates.py
"""
import glob
import json
import re

import pandas as pd
from huggingface_hub import hf_hub_download

SEED = 42
N_DICTIONARY_HITS = 1000
N_RUSSIAN_HITS = 300

# Rustem's dictionary has almost no Russian swear words, but Kazakh reviews often mix them in.
# Added by me: roots of common Russian obscene words (also Latin and Kazakh-keyboard spellings).
RUSSIAN_ROOTS = {
    "ху*": r"ху[йяеёиюү]|наху|xu[iy]|hu[iy]",
    "пизд*": r"п[иі]зд|пезд|п[иі]с[дт]ец|pizd",
    "бля*": r"\bбл[яә]\b|бля[тд]|blya",
    "еб*/заеб*": r"\bе[бп]а[нлтш]|\bеб[ау]|заеб|за[йи]бал|уеб|ёб|zaeb",
    "сука/сучка": r"\bсук[аиі]\b|сучк|\bsuka\b",
    "мудак": r"мудак|мудил",
    "гандон": r"г[ао]ндон",
    "пидор": r"п[иеі]д[оа]р",
    "долбоёб": r"долбо[её]|далба[её]",
    "говно": r"г[оа]вн[оаеи]",
}
N_CLEAN = {"negative_1_2_stars": 500, "marketplace": 300, "short_1_3_words": 200}

# Kazakh letters and the look-alike letters people type instead (қотақ ~ котак).
LOOKALIKE = {"қ": "[қк]", "к": "[кқ]", "ғ": "[ғг]", "г": "[гғ]", "ң": "[ңн]", "н": "[нң]",
             "ә": "[әа]", "а": "[аә]", "ө": "[өо]", "о": "[оө]", "ұ": "[ұүу]", "ү": "[үұу]",
             "у": "[уұү]", "і": "[іиы]", "и": "[иіы]", "ы": "[ыиі]", "һ": "[һх]", "х": "[хһ]"}


# --- part 1: load the dictionary --------------------------------------------------------
def load_dictionary() -> pd.DataFrame:
    rows = []
    for name in ["data.jsonl", "sentences.jsonl"]:          # single words + phrases
        path = hf_hub_download("Rustem-Kaimolla/kazakh-swear-words", name, repo_type="dataset")
        rows += [json.loads(line) for line in open(path, encoding="utf-8")]
    d = pd.DataFrame(rows)
    return d[d["label"] == 1]                               # neutral words are not search terms


# --- part 2: dictionary entry -> spelling-tolerant regex ---------------------------------
def to_pattern(entry: str) -> str:
    """'қотақ' -> '[қк][оө]т[аә][қк]'; a space may be missing: 'қотақ же' ~ 'қотақже'."""
    p = "".join(r"\s?" if ch == " " else LOOKALIKE.get(ch, re.escape(ch)) for ch in entry.lower())
    # very short entries (ам, шшс, 667) only as a whole word, otherwise 'ам' is inside 'адам'
    return rf"\b{p}\b" if len(entry) <= 3 else p


# --- part 3: search KazSAnDRA -------------------------------------------------------------
def main():
    reviews = pd.concat([pd.read_csv(f) for f in glob.glob("data/kazsandra_full/*/*.csv")])
    reviews = reviews.rename(columns={"custom_id": "id"})
    reviews["stars"] = reviews["label"] + 1                  # KazSAnDRA label 0..4 = 1..5 stars
    reviews = reviews.drop_duplicates("text")

    # skip what I already labeled in round 1
    done = pd.read_excel("my_labels/my_labels_round1.xlsx")
    done = done[pd.to_numeric(done["human_label"], errors="coerce").notna()]
    reviews = reviews[~reviews["id"].isin(done["kazsandra_id"])]

    dictionary = load_dictionary()
    lower = reviews["text"].astype(str).str.lower()
    found = pd.Series("", index=reviews.index)
    counts = {}
    for entry in dictionary["text"]:
        hit = lower.str.contains(to_pattern(entry), regex=True)
        counts[entry] = int(hit.sum())
        found[hit] = found[hit] + entry + "; "
    kazakh_hit = found != ""

    for name, pattern in RUSSIAN_ROOTS.items():
        hit = lower.str.contains(pattern, regex=True)
        counts["[ru] " + name] = int(hit.sum())
        found[hit] = found[hit] + "[ru] " + name + "; "
    reviews["found_words"] = found
    print("reviews found per search word:")
    print(pd.Series(counts).sort_values(ascending=False).to_string())

    hits = reviews[kazakh_hit]                                    # Rustem's dictionary
    russian_only = reviews[~kazakh_hit & (found != "")]           # only Russian roots
    clean = reviews[found == ""]
    print(f"\nKazakh dictionary hits: {len(hits)}, Russian-only hits: {len(russian_only)}")

    parts = [hits.sample(min(N_DICTIONARY_HITS, len(hits)), random_state=SEED)
                 .assign(pool="dictionary_hit"),
             russian_only.sample(min(N_RUSSIAN_HITS, len(russian_only)), random_state=SEED)
                 .assign(pool="russian_hit")]
    n_words = clean["text"].astype(str).str.split().str.len()
    pools = {"negative_1_2_stars": clean[clean["stars"] <= 2],
             "marketplace": clean[clean["domain"] == "market"],
             "short_1_3_words": clean[n_words <= 3]}
    taken = set()
    for name, df in pools.items():
        df = df[~df["id"].isin(taken)]
        s = df.sample(N_CLEAN[name], random_state=SEED).assign(pool=name)
        taken |= set(s["id"])
        parts.append(s)

    out = pd.concat(parts).sample(frac=1, random_state=SEED)   # shuffle: pools mixed
    out["label"] = ""                                           # <- filled in by hand
    out["note"] = ""
    out = out[["id", "domain", "stars", "pool", "found_words", "text", "label", "note"]]
    out.to_excel("my_labels/to_label_round2.xlsx", index=False)
    print(out["pool"].value_counts().to_string())
    print(f"saved {len(out)} rows -> my_labels/to_label_round2.xlsx")


if __name__ == "__main__":
    main()
