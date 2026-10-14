"""Tiền xử lý dùng chung cho train.py và app.py (giống notebook Midterm)."""
import re
from pathlib import Path
from underthesea import word_tokenize

KEEP = {"không", "rất", "quá", "tệ", "tốt", "thích", "buồn", "vui", "ghét"}  # từ cảm xúc, không bỏ
STOP = set((Path(__file__).parent / "data/vietnamese-stopwords.txt").read_text(encoding="utf-8").split("\n")) - KEEP


def preprocess(text: str) -> str:
    t = re.sub(r"\d+", "", text.lower())          # chữ thường, bỏ số
    t = re.sub(r"[^\w\s]", "", t)                  # bỏ dấu câu, giữ dấu tiếng Việt
    t = re.sub(r"\s+", " ", t).strip()
    words = word_tokenize(t, format="text").split()  # tách từ
    return " ".join(w for w in words if w not in STOP)
