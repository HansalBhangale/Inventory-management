# Right-sized image for a single-store pilot. uv for fast, fully-pinned reproducible envs.
# Written so a later Kubernetes lift is cheap, but K8s is deliberately NOT used yet (overhead for
# one store). One image, run as different commands (batch scorer / API / monitor) via compose.
FROM python:3.12-slim

# uv: fast, reproducible dependency installs (largely displaced pip-tools/poetry for speed)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

COPY src/ ./src/
COPY config/ ./config/

# Default command = batch scoring/reorder job; compose overrides for the API / monitor services.
# (Entry points are stubs until the orchestrator DAG lands — see deploy/docker-compose.yml.)
CMD ["python", "-c", "print('kirana image OK — set the command in docker-compose.yml')"]
