FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# tải sẵn model tách từ của underthesea khi build
RUN python -c "from underthesea import word_tokenize; word_tokenize('xin chào các bạn')"
COPY . .
ENV PORT=8000
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT}
