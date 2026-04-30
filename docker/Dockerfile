FROM python:3.11-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    curl git build-essential libhdf5-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install OpenHarness and pin MCP version
RUN pip install --no-cache-dir "openharness-ai==0.1.7" "mcp>=1.9.0"

# Patch MCP streamable_http API change (streamable_http_client -> streamablehttp_client)
COPY docker/patch_mcp.py /tmp/patch_mcp.py
RUN python3 /tmp/patch_mcp.py

# Install bioinformatics and ML stack
RUN pip install --no-cache-dir \
    scanpy==1.11.5 anndata \
    pandas numpy scipy \
    scikit-learn xgboost \
    matplotlib seaborn \
    leidenalg python-igraph \
    h5py requests tqdm

COPY agents/ /workspace/agents/
COPY analysis/ /workspace/analysis/

ENV PYTHONPATH=/workspace
ENV OMP_NUM_THREADS=4

ENTRYPOINT ["oh"]
