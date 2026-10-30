# One image for both commands of the evaluation service (run, compare). The service has no
# runtime dependencies, so the image holds only the project wheel; the datasets in data/ are
# mounted, not baked in, so a dataset bump needs no rebuild.
FROM python:3.14-slim AS builder
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip wheel --no-deps --no-cache-dir --wheel-dir /wheels .

FROM python:3.14-slim
COPY --from=builder /wheels/*.whl /tmp/
RUN pip install --no-cache-dir --no-deps /tmp/*.whl && rm /tmp/*.whl
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /data/eval-store && chown app:app /data/eval-store
USER app
WORKDIR /home/app
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "-m", "scene_rag.evaluation"]
