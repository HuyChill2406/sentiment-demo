"""Huấn luyện lại 3 mô hình từ data/dataset.csv và lưu vào models/. Chạy: python train.py"""
import json, time
import joblib, numpy as np, pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from nlp import preprocess

t0 = time.time()
df = pd.read_csv("data/dataset.csv", encoding="utf-8-sig", sep=";").dropna()
df["clean"] = df["Comment"].map(preprocess)
print(f"Tiền xử lý xong {len(df)} dòng ({time.time()-t0:.0f}s)", flush=True)

CLASSES = ["Negative", "Neutral", "Positive"]          # 0, 1, 2 (giống LabelEncoder)
y = df["Label"].map(CLASSES.index).values
tr, te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)  # cùng cách chia như notebook
txt_tr, txt_te = df["clean"].values[tr], df["clean"].values[te]
metrics = {}


def evaluate(key, model, X_te):
    p = model.predict(X_te)
    metrics[key] = dict(accuracy=accuracy_score(y[te], p), f1_weighted=f1_score(y[te], p, average="weighted"),
                        f1_macro=f1_score(y[te], p, average="macro"), n_train=len(tr), n_test=len(te))
    print(key, {k: round(v, 4) for k, v in metrics[key].items()}, flush=True)


# 1) Logistic Regression + Binary BoW (vectorizer chỉ fit trên tập train)
vec = CountVectorizer(max_features=5000, binary=True)
lr_bow = LogisticRegression(max_iter=1000).fit(vec.fit_transform(txt_tr), y[tr])
evaluate("bow_lr", lr_bow, vec.transform(txt_te))
joblib.dump((vec, lr_bow), "models/bow_lr.joblib")

# 2) Doc2Vec PV-DBOW (chỉ học trên tập train)
d2v = Doc2Vec(vector_size=100, window=5, min_count=2, workers=4, epochs=50, dm=0, seed=42)
docs = [TaggedDocument(t.split(), [i]) for i, t in enumerate(txt_tr)]
d2v.build_vocab(docs)
d2v.train(docs, total_examples=len(docs), epochs=d2v.epochs)
print(f"Doc2Vec xong ({time.time()-t0:.0f}s), đang infer vector...", flush=True)
# Train/test/web đều dùng infer_vector để vector nhất quán với lúc chạy thật
X_tr = np.array([d2v.infer_vector(t.split()) for t in txt_tr])
X_te = np.array([d2v.infer_vector(t.split()) for t in txt_te])
d2v.save("models/d2v_dbow.model")

lr_d2v = LogisticRegression(max_iter=1000).fit(X_tr, y[tr])
evaluate("d2v_lr", lr_d2v, X_te)
joblib.dump(lr_d2v, "models/d2v_lr.joblib")

# 3) MLP (1 lớp ẩn 128, ReLU, Adam, lr 0.001 – giống kiến trúc MLP trong notebook)
mlp = MLPClassifier(hidden_layer_sizes=(128,), learning_rate_init=0.001, max_iter=300,
                    early_stopping=True, random_state=42).fit(X_tr, y[tr])
evaluate("d2v_mlp", mlp, X_te)
joblib.dump(mlp, "models/d2v_mlp.joblib")

json.dump(metrics, open("models/metrics.json", "w"), indent=2)
print(f"Hoàn tất sau {time.time()-t0:.0f}s")
