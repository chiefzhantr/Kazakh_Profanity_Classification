"""Text cleaning used everywhere: for training data AND for new reviews at prediction time.

The model must always see text in the same format, so there is only one normalize_text().
"""
import re
import unicodedata

URL = re.compile(r"https?://\S+|www\.\S+")
MENTION = re.compile(r"@\w+")
PHONE = re.compile(r"(?:\+7|\b8)[\s\-(]*\d{3}[\s\-)]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}\b")  # KZ phones
# Surnames / patronymics: Ахметов, Сүлейменова, Алашұлы, Нұржанқызы -> <name>  (personal data)
SURNAME = re.compile(r"\b[А-ЯӘІҢҒҮҰҚӨҺ][а-яәіңғүұқөһ]+(?:ов|ова|ев|ева)\b"
                     r"|\b[А-ЯӘІҢҒҮҰҚӨҺа-яәіңғүұқөһ]{3,}(?:ұлы|улы|қызы|кызы)\b")
# ...but surname-shaped swear words (Далбаев, Нехуев) are kept - they are the signal we need
NOT_A_NAME = re.compile(r"далба|долба|ху[йяеё]|пизд|еба|ебу", re.I)
REPEAT = re.compile(r"(.)\1{3,}")          # 4+ same characters in a row
SPACES = re.compile(r"\s+")


def normalize_text(text) -> str:
    text = unicodedata.normalize("NFC", str(text))   # one code for letters typed differently
    text = URL.sub("<url>", text)
    text = MENTION.sub("@user", text)
    text = PHONE.sub("<phone>", text)
    text = SURNAME.sub(lambda m: m.group(0) if NOT_A_NAME.search(m.group(0)) else "<name>", text)
    text = REPEAT.sub(r"\1\1\1", text)               # 'ебааааа' -> 'ебааа', '🤬🤬🤬🤬🤬' -> '🤬🤬🤬'
    return SPACES.sub(" ", text).strip()
    # case and punctuation are kept: XLM-R is cased, and masks like 'ше***' need the symbols
