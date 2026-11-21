"""Step 5 - use the trained model on new reviews (inference, no training).

Command line:
  python src/predict.py "Мына тауар өте жақсы екен" "..."           # local model
  python src/predict.py --model <hf-user>/xlmr-kazakh-profanity "..."  # model from Hugging Face

Python / Colab:
  from predict import ProfanityClassifier
  clf = ProfanityClassifier("model/xlmr-kazakh-profanity")   # or "<hf-user>/xlmr-kazakh-profanity"
  clf.predict("текст")  ->  [{"text": ..., "label": 0/1, "probability": P(profanity)}]
"""
import argparse
import json
import os

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from text_utils import normalize_text

DEFAULT_MODEL = "model/xlmr-kazakh-profanity"


class ProfanityClassifier:
    def __init__(self, model_path: str = DEFAULT_MODEL, device: str = None):
        """model_path: a local folder OR a Hugging Face repo id."""
        if os.path.isdir(model_path):
            cfg_file = os.path.join(model_path, "inference_config.json")
        else:
            cfg_file = hf_hub_download(model_path, "inference_config.json")
        cfg = json.load(open(cfg_file))
        self.threshold = cfg["threshold"]          # tuned on the validation set
        self.max_length = cfg["max_length"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, texts, batch_size: int = 64):
        if isinstance(texts, str):
            texts = [texts]
        results = []
        for i in range(0, len(texts), batch_size):
            raw = texts[i:i + batch_size]
            clean = [normalize_text(t) for t in raw]      # same cleaning as in training
            enc = self.tokenizer(clean, truncation=True, max_length=self.max_length,
                                 padding=True, return_tensors="pt").to(self.device)
            probs = torch.softmax(self.model(**enc).logits, dim=-1)[:, 1].cpu().tolist()
            results += [{"text": t, "label": int(p >= self.threshold), "probability": round(p, 4)}
                        for t, p in zip(raw, probs)]
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("texts", nargs="*")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    clf = ProfanityClassifier(args.model)
    texts = args.texts or [input("review: ")]
    for r in clf.predict(texts):
        name = "PROFANITY" if r["label"] else "clean"
        print(f"label={r['label']} ({name})  P(profanity)={r['probability']:.3f}  | {r['text']}")


if __name__ == "__main__":
    main()
