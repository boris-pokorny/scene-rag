# One image for every role of the extraction service: the image installs every step's extra,
# a role process imports only its own (see pyproject.toml). Dependencies come from pdm.lock
# through `pdm export`, so the image and a local `pdm install` agree; the runtime stage has
# neither pdm nor __pypackages__.
FROM python:3.14-slim AS builder
RUN pip install --no-cache-dir pdm==2.26.8
WORKDIR /build
COPY pyproject.toml pdm.lock README.md ./
RUN pdm export --prod -G segmentation -G enrichment -G scene-api -o requirements.txt
COPY src ./src
RUN pip wheel --no-deps --no-cache-dir --wheel-dir /wheels .

FROM python:3.14-slim
# ffmpeg: audio clips in the segmentation step.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*
# Dependencies before the project wheel: a code change does not re-download them.
COPY --from=builder /build/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY --from=builder /wheels/*.whl /tmp/
RUN pip install --no-cache-dir --no-deps /tmp/*.whl && rm /tmp/*.whl /tmp/requirements.txt
# The /data directories seed the ownership of the volumes mounted over them, so the non-root
# worker can write derivatives and the speech model cache.
RUN useradd --create-home --uid 1000 app \
 && mkdir -p /data/media /data/derivatives /data/models \
 && chown -R app:app /data
USER app
WORKDIR /data
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "-m", "scene_rag.extraction"]
