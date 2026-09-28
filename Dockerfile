FROM python:3.12-slim
WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY oceanembed ./oceanembed
COPY web ./web
COPY outputs/public ./outputs/public
ENV OCEANEMBED_RESULTS=/app/outputs/public
EXPOSE 7860
CMD ["python", "-m", "uvicorn", "oceanembed.api:app", "--host", "0.0.0.0", "--port", "7860"]
