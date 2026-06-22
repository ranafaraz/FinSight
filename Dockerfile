FROM python:3.12-slim

WORKDIR /app

# Install the package (offline backends only — no model downloads, no API keys).
COPY pyproject.toml README.md ./
COPY finsight ./finsight
COPY evals ./evals
RUN pip install --no-cache-dir -e ".[dev]"

# Default: run the full offline eval and print the results table.
CMD ["python", "-m", "evals.harness"]
