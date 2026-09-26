# Space root must contain: app.py, requirements.txt, Dockerfile, README.md,
# docs.jsonl, plus the triadlm/ and demo/ package folders from the repo.
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py docs.jsonl ./
COPY triadlm/ ./triadlm/
COPY demo/ ./demo/
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
