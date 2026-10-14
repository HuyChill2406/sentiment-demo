import json, sqlite3, time
from collections import defaultdict, deque
from pathlib import Path

import joblib, pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.staticfiles import StaticFiles
from gensim.models.doc2vec import Doc2Vec
from pydantic import BaseModel

from nlp import preprocess

BASE = Path(__file__).parent
DB = BASE / "data" / "app.db"
CLASSES = ["Negative", "Neutral", "Positive"]
PAGE_SIZE = 12

# ---------- Database (SQLite) ----------
def db():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    con = db()
    con.execute("CREATE TABLE IF NOT EXISTS samples(id INTEGER PRIMARY KEY, comment TEXT, label TEXT)")
    con.execute("CREATE TABLE IF NOT EXISTS submissions(id INTEGER PRIMARY KEY, text TEXT, results TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP)")
    if con.execute("SELECT COUNT(*) FROM samples").fetchone()[0] == 0:  # lần đầu: nạp dataset.csv vào DB
        df = pd.read_csv(BASE / "data/dataset.csv", encoding="utf-8-sig", sep=";").dropna()
        con.executemany("INSERT INTO samples(comment, label) VALUES(?,?)", df[["Comment", "Label"]].values.tolist())
        con.execute("CREATE INDEX IF NOT EXISTS idx_label ON samples(label)")
    con.commit(); con.close()


# ---------- Models ----------
vec, lr_bow = joblib.load(BASE / "models/bow_lr.joblib")
d2v = Doc2Vec.load(str(BASE / "models/d2v_dbow.model"))
lr_d2v = joblib.load(BASE / "models/d2v_lr.joblib")
mlp = joblib.load(BASE / "models/d2v_mlp.joblib")
METRICS = json.load(open(BASE / "models/metrics.json"))

MODELS = [
    dict(id="bow_lr", name="Logistic Regression", representation="Binary Bag of Words (5.000 từ, có/không xuất hiện)"),
    dict(id="d2v_lr", name="Logistic Regression", representation="Doc2Vec PV-DBOW (vector 100 chiều)"),
    dict(id="d2v_mlp", name="MLP (1 lớp ẩn, 128 nơ-ron)", representation="Doc2Vec PV-DBOW (vector 100 chiều)"),
]
for m in MODELS:
    m["metrics"] = METRICS[m["id"]]


def predict_all(text: str):
    clean = preprocess(text)
    tokens = clean.split()
    known = sum(t in vec.vocabulary_ for t in tokens) / len(tokens) if tokens else 0.0
    v_bow = vec.transform([clean])
    v_d2v = d2v.infer_vector(tokens).reshape(1, -1)
    outs = [lr_bow.predict_proba(v_bow)[0], lr_d2v.predict_proba(v_d2v)[0], mlp.predict_proba(v_d2v)[0]]
    results = []
    for m, p in zip(MODELS, outs):
        results.append(dict(id=m["id"], label=CLASSES[int(p.argmax())], probs={c: round(float(x), 4) for c, x in zip(CLASSES, p)}))
    return results, round(known, 2)


# ---------- Giới hạn tốc độ (chống cào dữ liệu hàng loạt) ----------
hits = defaultdict(deque)

def rate_limit(request: Request, key: str, limit: int, window: int = 60):
    ip = request.headers.get("x-forwarded-for", request.client.host).split(",")[0].strip()
    q = hits[(ip, key)]
    now = time.time()
    while q and now - q[0] > window:
        q.popleft()
    if len(q) >= limit:
        raise HTTPException(429, "Bạn thao tác quá nhanh, vui lòng thử lại sau ít giây.")
    q.append(now)


# ---------- API ----------
app = FastAPI(title="Demo phân loại cảm xúc tiếng Việt")
init_db()


class PredictIn(BaseModel):
    text: str


@app.get("/api/models")
def get_models():
    return MODELS


@app.get("/api/samples")
def get_samples(request: Request, page: int = Query(1, ge=1), label: str = "", q: str = Query("", max_length=100)):
    rate_limit(request, "samples", 40)
    where, args = [], []
    if label in CLASSES:
        where.append("label = ?"); args.append(label)
    if q.strip():
        where.append("comment LIKE ?"); args.append(f"%{q.strip()}%")
    w = ("WHERE " + " AND ".join(where)) if where else ""
    con = db()
    total = con.execute(f"SELECT COUNT(*) FROM samples {w}", args).fetchone()[0]
    rows = con.execute(f"SELECT id, comment, label FROM samples {w} ORDER BY id LIMIT ? OFFSET ?", args + [PAGE_SIZE, (page - 1) * PAGE_SIZE]).fetchall()
    con.close()
    return dict(total=total, page=page, pages=max(1, -(-total // PAGE_SIZE)), items=[dict(r) for r in rows])


@app.get("/api/samples/random")
def random_sample(request: Request, label: str = ""):
    rate_limit(request, "samples", 40)
    con = db()
    if label in CLASSES:
        r = con.execute("SELECT id, comment, label FROM samples WHERE label=? ORDER BY RANDOM() LIMIT 1", (label,)).fetchone()
    else:
        r = con.execute("SELECT id, comment, label FROM samples ORDER BY RANDOM() LIMIT 1").fetchone()
    con.close()
    return dict(r)


@app.post("/api/predict")
def predict(body: PredictIn, request: Request):
    rate_limit(request, "predict", 20)
    text = body.text.strip()
    if len(text) < 3 or len(text) > 500:
        raise HTTPException(422, "Bình luận cần dài từ 3 đến 500 ký tự.")
    if not any(ch.isalpha() for ch in text):
        raise HTTPException(422, "Bình luận cần có chữ cái (tiếng Việt).")
    results, known = predict_all(text)
    con = db()
    con.execute("INSERT INTO submissions(text, results) VALUES(?,?)", (text, json.dumps(results)))
    con.commit(); con.close()
    return dict(results=results, known_ratio=known)


@app.get("/api/stats")
def stats():
    con = db()
    out = dict(samples=con.execute("SELECT COUNT(*) FROM samples").fetchone()[0],
               submissions=con.execute("SELECT COUNT(*) FROM submissions").fetchone()[0])
    con.close()
    return out


app.mount("/", StaticFiles(directory=BASE / "static", html=True), name="static")  # chỉ phục vụ thư mục static
