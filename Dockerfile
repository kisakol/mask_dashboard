# ── GeoMx Mask Dashboard — CPU image ─────────────────────────────────────────
# Runs on any machine. No CUDA required.
# Build:  docker build -t geomx-mask-dashboard .
# Run:    docker run -p 8501:8501 -v /path/to/data:/data geomx-mask-dashboard

FROM python:3.11-slim

# --------------------------------------------------------------------------- #
# System dependencies
# opencv-python needs libGL; tifffile/scipy need libgomp for OpenMP threading  #
# --------------------------------------------------------------------------- #
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------------- #
# Python dependencies                                                           #
# --------------------------------------------------------------------------- #
WORKDIR /app

# Copy only requirements first so Docker can cache this layer independently
COPY requirements.txt ./requirements.txt

# Strip comment-only lines (the GPU install notes) so pip doesn't choke,
# then install. --no-cache-dir keeps the image smaller.
RUN pip install --no-cache-dir --upgrade pip \
 && grep -v '^\s*#' requirements.txt | grep -v '^\s*$' \
    | pip install --no-cache-dir -r /dev/stdin

# --------------------------------------------------------------------------- #
# Application code                                                              #
# --------------------------------------------------------------------------- #
COPY . .

# --------------------------------------------------------------------------- #
# Streamlit configuration                                                       #
# headless=true  — no browser auto-open (no display in container)               #
# enableCORS=false — safe behind a reverse proxy or for local Docker use        #
# enableXsrfProtection=false — needed when serving behind a path prefix         #
# --------------------------------------------------------------------------- #
RUN mkdir -p /root/.streamlit && cat > /root/.streamlit/config.toml <<'TOML'
[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = false
maxUploadSize = 500

[browser]
gatherUsageStats = false

[theme]
base = "light"
TOML

# /data is the bind-mount point for TIFF files and output masks
RUN mkdir -p /data

EXPOSE 8501

# Healthcheck — Streamlit exposes a /_stcore/health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
