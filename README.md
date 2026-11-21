# Kazakh Profanity Detection

KBTU · DSSC5113 · Bonus project.

A binary classifier for Kazakh product and app reviews:

- **0** = no profanity
- **1** = profanity (obscene words)

The model is fine-tuned **XLM-RoBERTa-base** and returns a label and the probability of profanity.

| | precision | recall | F1 |
|---|---|---|---|
| **Test, class 1 (profanity)** | **0.932** | **0.873** | **0.901** |

- Model on Hugging Face: **`<hf-username>/xlmr-kazakh-profanity`** (link: https://huggingface.co/<hf-username>/xlmr-kazakh-profanity)
- Quick use:

```python
from predict import ProfanityClassifier          # run from the src/ folder
clf = ProfanityClassifier("<hf-username>/xlmr-kazakh-profanity")   # or the local model folder
clf.predict("Қотақбас ойын, ақшамды жеді")
# [{'text': 'Қотақбас ойын, ақшамды жеді', 'label': 1, 'probability': 0.998}]
```

---

## 1. Dataset

### Source
- **Texts.** All texts come from **KazSAnDRA**, about 175k Kazakh reviews from app stores, marketplaces, maps and a bookstore. It is published by ISSAI under the CC-BY-4.0 license: https://github.com/IS2AI/KazSAnDRA.
- **Labels.** Every label was assigned **by hand by me**. No labels were taken from other people's datasets.
- **Search words.** Profanity is rare among random reviews, so I first searched KazSAnDRA for candidate reviews (`src/find_candidates.py`) with these word lists:
  - the **Kazakh Swear Words** dictionary by Rustem Kaimolla (https://huggingface.co/datasets/Rustem-Kaimolla/kazakh-swear-words, MIT). It has 56 words and 63 phrases, of which only the entries with `label == 1` were used;
  - **Russian obscene roots added by me** (`RUSSIAN_ROOTS`: ху-, пизд-, бля, еб-/заеб-, сука, мудак, гандон, пидор, долбоёб, говно). Kazakh reviews often mix in Russian swear words, and the dictionary almost does not cover them.

  The search tolerates common spelling variants:
  - Kazakh letters and their look-alikes (қ/к, ө/о, ұ/у, і/и …), so «қотақ» also finds «котак»;
  - an optional space inside phrases;
  - whole-word matching for very short entries, so «ам» does not match inside «адам».
- **Candidate pools.** Round 2 sampled pools, all with seed 42:
  - 1000 dictionary hits and 300 Russian-root hits;
  - **hard negatives** with no hits: 500 one- and two-star reviews, 300 marketplace reviews and 200 very short reviews. These teach the model that "angry review" is not the same as "profanity".
- **Earlier labeled set.** An earlier labeling round (round 1, 507 reviews) is also included.

### Labeling rule
- **1:** the review contains an **obscene word**, Kazakh or Russian. This includes masked forms (`ше***`), forms with Kazakh suffixes («гандондар», «пиздеспін»), Latin spelling (`gandon`) and typos (`котакбас`, `зайбал`).
- **0:** everything else. This includes **insults without obscene words** («жынды», «мал», «тупой», «ақымақ», «далбаса»), angry and negative reviews, and normal reviews.
- Insults were deliberately labeled 0. The task asks about *profanity*, and an insult like «жынды ойын» ("crazy game") is rude but is not a swear word.

### Mixed language and masked words
- Reviews are kept as they are: Kazakh, Russian, code-switched text and Latin script.
- XLM-R is multilingual, so no language filtering or transliteration is done.
- Case and symbols are kept, so masked words like `ше***` stay recognisable.

### Cleaning (`src/text_utils.py`, applied in training and in prediction)
- Unicode NFC.
- Personal data is replaced:
  - URLs → `<url>`;
  - @mentions → `@user`;
  - Kazakhstan phone numbers → `<phone>`;
  - surnames and patronymics (-ов/-ова/-ев/-ева, -ұлы/-қызы) → `<name>`. Surname-shaped swear words such as «Далбаев» are kept.
- One review containing a person's full name was removed completely.
- Runs of 4 or more identical characters are shortened to 3 («ебааааа» → «ебааа»).

### Duplicates and split (`src/prepare_data.py`)
- **Exact duplicates.** Texts are compared after lowercasing and keeping only letters and digits. 4 duplicates were removed. If the same text had labels 0 and 1, both copies would be dropped; there were 0 such cases.
- **Near-duplicates.** Texts with character 4-gram Jaccard ≥ 0.8 are joined into one group with union-find. 4 such groups were found.
- **Split.** `StratifiedGroupKFold(10, shuffle=True, random_state=42)` is used:
  - fold 0 goes to test, fold 1 to val, and the other 8 folds to train (80/10/10);
  - the 0/1 ratio is the same in each split;
  - a near-duplicate group never appears in two splits. An `assert` in the code checks this.

| split | total | 0 (clean) | 1 (profanity) |
|---|---|---|---|
| train | 2032 | 1252 | 780 |
| val | 254 | 149 | 105 |
| test | 253 | 143 | 110 |
| **total** | **2539** | **1544** | **995** |

- The median review is 6 words long. The data mostly comes from the app store (2001), then the marketplace (389), maps (87) and the bookstore (62).
- The ready splits are in `data/processed/`, and the statistics are in `data/processed/split_info.json`.
- The original labeling spreadsheets are **not** in the repository because they contain the unredacted raw texts.

## 2. Model and training (`src/train.py`)
- **Base model.** `FacebookAI/xlm-roberta-base`: a pre-trained multilingual transformer with Kazakh and Russian in its pre-training data, plus a new 2-class classification head.
- **Training:**
  - cross-entropy loss with AdamW (lr 2e-5, weight decay 0.01);
  - linear learning-rate schedule with 10% warmup and gradient clipping at 1.0;
  - batch size 16, max length 128 tokens, dynamic padding, seed 42;
  - trained on CPU, about 2.8 min per epoch.
- **Model selection.**
  - After each epoch, class-1 F1 is computed on **val**, and only the best epoch is saved.
  - Then the decision threshold is searched on val over 0.05–0.95. The best threshold is 0.5.
- **Number of epochs.** It was also chosen on val:

| run | best val F1 |
|---|---|
| **4 epochs (final)** | **0.929** (epoch 4) |
| 6 epochs | 0.919 (epoch 5; epoch 6 already worse: overfitting) |

- Logs: `results/train_log.json` and `results/train_log_6epochs.json`.

## 3. Test results (`src/evaluate.py`, run once)

| | precision | recall | F1 | support |
|---|---|---|---|---|
| clean (0) | 0.907 | 0.951 | 0.928 | 143 |
| **profanity (1)** | **0.932** | **0.873** | **0.901** | 110 |
| accuracy | | | 0.917 | 253 |

Confusion matrix (rows = true label, columns = predicted label), also in `results/confusion_matrix.png`:

| | pred 0 | pred 1 |
|---|---|---|
| **true 0** | TN 136 | FP 7 |
| **true 1** | FN 14 | TP 96 |

All 21 mistakes are listed in `results/test_errors.csv`.

### Error analysis
1. **Rare, Latin or suffixed forms of swear words → FN.**
   - «Pashol bon **gandon** jalap» gets P = 0.003, and «**Гандондар** 20 рет регистрация жасадым» also gets 0.003. "гандон" appears only 3 times in train, and Latin-script profanity is almost absent.
   - «… ааа **фак** 😥» is missed: "фак" appears once in train, and there it is labeled 0.
2. **One obscene word in a long or positive text → FN.**
   - In «Просто топ ойын барго **еба**👍👍👍» the positive words and emojis win (P = 0.02).
   - In a 60-word complaint, the one word «дахуя» is lost.
   - «Жаман нахуй» (P = 0.003) is surprising, because «наху» is labeled 1 in 49 of 50 train examples. Short «жаман …» reviews are strongly associated with the clean "angry review" class.
3. **Insults without obscene words → FP.**
   - «Мал **жынды** ойын … көт ойын» gets 0.99, and «**Жиндиче** ойун екен сор сор» gets 0.996.
   - Under my rule these are 0, but in the data «жынды» often appears next to real swear words. So the model partly learned "rude = profanity".
   - This insult/profanity border is the hardest part of the task.
4. **Noisy or garbage texts.**
   - A long, incoherent text containing «көт татақ» and «ұятсыз» is labeled 0, while the model says 0.90. This is arguably a labeling error, not a model error.
5. **Round-1 data.** 9 of the 14 FN come from the earlier round-1 labels: the model found only 3 of their 12 profane test reviews. Round 1 has more rare and masked forms («Алдапсигер», «Ой біләә ма?», «котагымды жеген»). These were **not** fixed by looking at test, which would leak test information. The proper fix is more labeled data of this kind.

### What FP and FN mean for moderation
- **False positive.** A normal review is hidden or flagged. The author is annoyed, and honest (sometimes harsh) criticism can disappear. A human should review flagged items before deleting them.
- **False negative.** Profanity is published and every visitor sees it. For a public platform this is usually the worse error. If recall matters more, lower the threshold, chosen on val, for example to 0.3. Recall goes up, but FP goes up too.

### Limitations
- The dataset is small (2.5k reviews) and mostly app-store reviews.
- There is very little Latin-script profanity.
- One annotator, so there is no inter-annotator agreement.
- Single first names without surnames are not redacted.

## 4. How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# (only to redo candidate search) download KazSAnDRA from GitHub into data/kazsandra_full/kazsandra/
#   https://github.com/IS2AI/KazSAnDRA/tree/main/dataset  -> 06_sc_train_ib.zip, 09_sc_valid.zip, 10_sc_test.zip
python src/find_candidates.py        # -> labeling spreadsheet (labels were then added by hand)
python src/prepare_data.py           # labels -> data/processed/{train,val,test}.csv

DEVICE=cpu python src/train.py       # ~11 min on CPU; on a GPU (Colab) just: python src/train.py
DEVICE=cpu python src/evaluate.py    # test metrics, confusion matrix, errors
python src/predict.py "Мына тауар өте жақсы екен" "Заебал реклама"
```

To run inference with the model from Hugging Face, no training is needed:

```bash
python src/predict.py --model <hf-username>/xlmr-kazakh-profanity "Қотақбас ойын"
```

`DEVICE=cpu` is needed on macOS 14 because Apple MPS crashes with torch 2.15.

### Files
| file | what it does |
|---|---|
| `src/find_candidates.py` | searches KazSAnDRA with the swear-word dictionary and Russian roots, builds the labeling sheet |
| `src/text_utils.py` | `normalize_text`: the same cleaning for training and prediction |
| `src/prepare_data.py` | merges my labels, cleans, removes duplicates, near-duplicate groups, 80/10/10 split |
| `src/train.py` | fine-tunes XLM-R, picks the best epoch and threshold on val |
| `src/evaluate.py` | one-time test evaluation |
| `src/predict.py` | `ProfanityClassifier`: inference on new text (local folder or HF repo) |
| `src/push_to_hub.py` | uploads the model folder to Hugging Face |

## 5. Acknowledgements and AI assistance
- **Data:**
  - KazSAnDRA (ISSAI, CC-BY-4.0);
  - the Kazakh Swear Words dictionary by Rustem Kaimolla (MIT).
- **My own work:**
  - all labeling (about 2.5k reviews);
  - the labeling rule (insults are not profanity);
  - adding the Russian obscene roots to the candidate search;
  - the decisions at each step: what data to use, the label rule, choosing 4 epochs over 6;
  - checking the results.
- **AI assistance.** I used Claude (Anthropic) as an assistant to write and explain the code step by step: candidate search, cleaning, the split, the training and evaluation scripts, and this README. Every step was discussed and checked by me.
