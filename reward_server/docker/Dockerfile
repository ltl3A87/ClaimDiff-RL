FROM python:3.10-slim

RUN pip install --no-cache-dir -U \
    uvicorn \
    fastapi \
    pydantic \
    requests \
    aiohttp \
    Pillow \
    json_repair

WORKDIR /app

COPY reward_serving_fastapi.py .
COPY verifier/ verifier/

EXPOSE 8000

CMD ["python", "reward_serving_fastapi.py"]
