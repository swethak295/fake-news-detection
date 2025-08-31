import re
import string

_URL_RE = re.compile(r"(https?://\S+|www\.\S+)")

def basic_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _URL_RE.sub(" URL ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text
