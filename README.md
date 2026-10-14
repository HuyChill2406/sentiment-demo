# Phân loại cảm xúc bình luận tiếng Việt: so sánh Bag of Words, Doc2Vec và Deep Learning

Dự án giữa kỳ môn **Nhập môn Xử lý ngôn ngữ tự nhiên**.

Bài toán: phân loại một bình luận tiếng Việt thành **Negative / Neutral / Positive**. Dự án gồm hai phần:

1. **Notebook `Midterm.ipynb`**: tạo dataset bằng LLM, so sánh nhiều cách biểu diễn văn bản và nhiều mô hình ML/DL.
2. **Web demo** (FastAPI + SQLite + HTML): cho người dùng xem dataset và thử 3 mô hình tốt nhất.

Demo trực tuyến: _(dán link Render vào đây)_. Gói miễn phí tự ngủ khi không có người dùng, lần mở đầu có thể mất khoảng 1 phút.

---

## 1. Dataset

- **41.818 bình luận** tiếng Việt do mô hình `deepseek-chat` sinh ra (dữ liệu tổng hợp, synthetic).
- Mỗi lần gọi API chọn ngẫu nhiên một prompt chủ đề (giải trí, game, anime, thể thao, công nghệ, chính trị, đời sống...) và một nhãn cảm xúc, nhờ đó dữ liệu cân bằng giữa 3 lớp. Kết quả được lọc trùng.
- Phân bố nhãn: Negative 14.320 · Positive 14.112 · Neutral 13.386.
- File `data/dataset.csv`: dấu phân cách `;`, mã hóa UTF-8, hai cột `Comment` và `Label`.

> Vì là dữ liệu do LLM sinh, các chỉ số bên dưới có thể **cao hơn** so với khi gặp bình luận thật trên mạng xã hội.

## 2. Pipeline

1. **Tiền xử lý:** chữ thường, bỏ số và dấu câu (giữ dấu tiếng Việt), tách từ bằng `underthesea`, bỏ stopwords nhưng **giữ lại các từ mang cảm xúc** (không, rất, quá, tệ, tốt, thích, buồn, vui, ghét).
2. **Biểu diễn văn bản:**
   - Bag of Words: Count, Binary, TF, TF-IDF, Bigram, Trigram (5.000 đặc trưng).
   - Doc2Vec tự huấn luyện: PV-DM và PV-DBOW (100 chiều).
   - Word2Vec pretrained tiếng Việt (400 chiều, lấy trung bình các từ trong câu).
3. **Mô hình:**
   - ML: Naive Bayes, Logistic Regression, Decision Tree, Random Forest.
   - DL (PyTorch): MLP, LSTM, BiLSTM, chạy trên vector Doc2Vec/Word2Vec.
4. **Đánh giá:** chia 80/20 (`random_state=42`), dùng Accuracy, Precision, Recall, F1.

## 3. Kết quả trong notebook

| Nhóm phương pháp | Mô hình tốt nhất | Accuracy |
|---|---|---|
| Bag of Words + ML | Logistic Regression + Binary BoW | **95,27%** |
| Doc2Vec (không pretrain) + ML | Logistic Regression + PV-DBOW | 88,03% |
| Doc2Vec (không pretrain) + DL | MLP + PV-DBOW | 81,01% |
| Word2Vec pretrained + ML / DL | Logistic Regression / MLP | 74,18% / 68,09% |

Nhận xét chính:

- Logistic Regression + Binary BoW cho kết quả cao nhất; với dữ liệu sinh bằng LLM, các từ khóa cảm xúc rõ ràng nên BoW khai thác rất tốt.
- Trong các biểu diễn dày (dense), **PV-DBOW** tốt hơn PV-DM và Word2Vec pretrained ở mọi mô hình.
- Bigram và Trigram kém rõ rệt; Decision Tree dễ overfit (train accuracy = 1,0 trên Doc2Vec).
- Các mô hình DL trong notebook huấn luyện 20 epoch nên **chưa hội tụ** (xem mục 4).
- "Pretrained" ở đây là Word2Vec lấy trung bình vector từ, không phải Doc2Vec pretrain.

## 4. Web demo

Chọn một bình luận trong dataset (xem, lọc, tìm kiếm, nhưng **không có chức năng tải về**) hoặc tự viết bình luận tiếng Việt, rồi xem 3 mô hình dự đoán. Giao diện ghi rõ từng thẻ kết quả thuộc mô hình nào và dùng biểu diễn văn bản nào.

| Mô hình | Biểu diễn văn bản | Accuracy | F1 (weighted) |
|---|---|---|---|
| Logistic Regression | Binary Bag of Words (5.000 từ) | 95,21% | 0,9521 |
| Logistic Regression | Doc2Vec PV-DBOW (100 chiều) | 87,46% | 0,8745 |
| MLP (1 lớp ẩn, 128 nơ-ron) | Doc2Vec PV-DBOW (100 chiều) | 92,90% | 0,9291 |

Các số này do `train.py` tính lại (xem `models/metrics.json`), nên khác nhẹ so với notebook: vectorizer và Doc2Vec chỉ học trên tập train, vector luôn lấy bằng `infer_vector` giống lúc chạy web, và MLP (scikit-learn, cùng kiến trúc 1 lớp ẩn 128, ReLU, Adam) được huấn luyện đến khi hội tụ, nên cao hơn nhiều so với MLP chưa hội tụ trong notebook (81,01%).

**Kiến trúc:** frontend HTML thuần (`static/index.html`) gọi API của backend FastAPI (`app.py`); dataset được nạp vào SQLite ở lần chạy đầu và các câu người dùng thử cũng được lưu ở đó.

**API** (tài liệu tự sinh ở `/docs`): `GET /api/models` · `GET /api/samples?page=&label=&q=` · `GET /api/samples/random` · `POST /api/predict` với body `{"text": "..."}` · `GET /api/stats`.

**Về việc không cho tải dataset:** không có endpoint xuất file, file CSV/DB không nằm trong thư mục được phục vụ, mỗi trang chỉ trả 12 dòng và mỗi IP bị giới hạn 40 lượt xem mỗi phút. Người dùng vẫn đọc được từng trang, nên cách này làm khó việc sao chép hàng loạt chứ không ngăn tuyệt đối. Nếu repo công khai thì `data/dataset.csv` vẫn tải được từ GitHub, vì vậy hãy để repo ở chế độ private nếu cần giữ dataset.

## 5. Cấu trúc repo

```
.
├── README.md
├── Midterm.ipynb            # notebook thực nghiệm (xóa API key trước khi commit)
├── app.py                   # backend FastAPI + SQLite
├── nlp.py                   # tiền xử lý dùng chung
├── train.py                 # huấn luyện lại 3 mô hình từ data/dataset.csv
├── static/index.html        # frontend
├── data/
│   ├── dataset.csv
│   └── vietnamese-stopwords.txt
├── models/                  # model đã huấn luyện + metrics.json
├── requirements.txt         # thư viện cho web demo (đã ghim phiên bản)
├── Dockerfile
└── .dockerignore
```

## 6. Chạy web demo trên máy

```bash
pip install -r requirements.txt
uvicorn app:app --port 8000        # mở http://localhost:8000
```

Model đã có sẵn trong `models/`. Muốn huấn luyện lại: `python train.py` (khoảng 2–3 phút).

## 7. Chạy notebook

Dùng Google Colab (khuyến nghị) hoặc Jupyter cục bộ:

```bash
pip install -U pandas numpy scikit-learn gensim torch matplotlib seaborn wordcloud nltk tqdm requests plotly underthesea imbalanced-learn
```

- Phần tạo dataset gọi DeepSeek API (`https://api.deepseek.com/chat/completions`). Đặt khóa API trong **biến môi trường** hoặc Colab Secrets, **không ghi trực tiếp vào notebook hay commit lên repo**. Sinh nhiều mẫu rất lâu và tốn token, nên có thể để mỗi request trả về 5 mẫu và chạy song song nhiều kernel.
- Notebook đọc `dataset.csv` (dấu phân cách `;`, cột `Comment`, `Label`). Nếu file sinh ra tên `comments.csv`, hãy đổi tên thành `dataset.csv`.
- Phần Word2Vec pretrained cần tải thêm file `wiki.vi.model.bin` (Word2Vec tiếng Việt, 400 chiều).
- Chạy lần lượt: tiền xử lý → BoW + ML → Doc2Vec/Word2Vec + ML → DL → so sánh tổng hợp → demo dự đoán.

## 8. Deploy miễn phí trên Render

1. Đẩy repo lên GitHub (private nếu muốn giữ dataset), đăng nhập Render bằng tài khoản GitHub.
2. **New → Web Service**, chọn repo, Language = **Docker**, Region = Singapore, Instance Type = **Free**, rồi Create.
3. Chờ build 5–10 phút, mở link `https://<tên-dịch-vụ>.onrender.com`.

Gói Free có 512 MB RAM (app dùng khoảng 270 MB), tự ngủ sau 15 phút không có truy cập, và dữ liệu ghi trên đĩa mất mỗi lần ngủ. Dataset được nạp lại tự động, chỉ các câu người dùng đã thử là bị xóa. Sửa code chỉ cần `git push`, Render sẽ tự build lại.

## 9. Hạn chế và hướng phát triển

- Dữ liệu do LLM sinh nên chưa phản ánh đầy đủ bình luận thật; nhãn do LLM gán chưa được kiểm tra thủ công.
- Chưa dùng mô hình ngôn ngữ tiền huấn luyện cho tiếng Việt như PhoBERT, có thể cho kết quả tốt hơn với câu phủ định hoặc châm biếm.
- Chưa có cross-validation và confusion matrix; kết quả dựa trên một lần chia 80/20.
