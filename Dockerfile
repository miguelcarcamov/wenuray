# Base image: Python, Wenuray's pinned dependencies and CASA's measures data, kept small.
# - No CUDA: nothing in Wenuray uses the GPU yet. GPUvmem was removed and Pyralysis (#24) brings its
#   own CUDA requirements when it is added.
# - The casatools wheels bundle casacore, so nothing is compiled.
# - Python 3.12 is supported by both CASA (>= 6.7.2) and Pyralysis (#30).
# - The builder stage installs, strips and prunes; the measures stage fetches CASA's data; the
#   final image only copies their results, without the build tools.
ARG PYTHON_IMAGE=python:3.12-slim-bookworm

FROM ${PYTHON_IMAGE} AS builder

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends binutils && \
    rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH=/opt/venv/bin:$PATH
RUN python -m venv /opt/venv

# The pinned dependency set that CI tests. Then, to save space:
# - strip the debug symbols left in some libraries bundled with casatools (~30 MB);
# - drop the test suites of numpy, scipy and matplotlib (~55 MB). astropy imports its own tests
#   package at startup, so it is kept.
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && \
    site=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])") && \
    find "$site/casatools" -type f -name "*.so*" -exec strip --strip-unneeded {} + && \
    find "$site/numpy" "$site/scipy" "$site/matplotlib" "$site/mpl_toolkits" \
        -depth -type d -name tests -exec rm -rf {} + && \
    python -c "import casaconfig, numpy, scipy, matplotlib"

FROM builder AS measures

# CASA reads its measures data (IERS Earth orientation, observatory positions, ephemerides) from
# measurespath. Only the measures (~20 MB) and CASA's maintained observatory table are kept, not the
# full casarundata (~1 GB), which adds antenna responses and simulator files used by e.g. the
# awproject gridder; mount it at /opt/casa/data if you need it. tools/fetch_casa_data.py explains
# where each part comes from. The NRAO server can be slow, so the download is retried;
# MEASURES_SITE selects another measures server.
ARG MEASURES_SITE=""
COPY tools/fetch_casa_data.py /tmp/fetch_casa_data.py
RUN mkdir -p /opt/casa/data && \
    for attempt in 1 2 3; do \
        rm -rf /opt/casa/data/* && \
        python /tmp/fetch_casa_data.py /opt/casa/data --measures-site "${MEASURES_SITE}" && break; \
        [ "$attempt" = 3 ] && exit 1; \
        sleep 30; \
    done && \
    test -f /opt/casa/data/geodetic/readme.txt && test -d /opt/casa/data/geodetic/Observatories && \
    printf '%s\n' 'measurespath = "/opt/casa/data"' 'measures_auto_update = False' \
        'data_auto_update = False' > /opt/casa/casasiteconfig.py && \
    chmod -R a+rX /opt/casa

FROM ${PYTHON_IMAGE}

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH=/opt/venv/bin:$PATH \
    VIRTUAL_ENV=/opt/venv

# OpenMP's runtime: Numba's thread-safe threading layer, which Pyralysis (#24) needs to run its
# parallel kernels from several Dask threads (without it, Numba's fallback aborts the process)
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
# Auto updates are off, so every container of an image uses the same data. The directory is
# readable by any user (e.g. under Apptainer/Singularity).
COPY --from=measures /opt/casa /opt/casa

# CASA looks telescopes up by name in its observatory table (e.g. for frame conversions)
RUN python -c "import casatasks, casatools; print('casatools', casatools.version_string()); \
me = casatools.measures(); assert all(me.observatory(t) for t in ('VLA', 'ALMA'))"

LABEL org.opencontainers.image.source="https://github.com/miguelcarcamov/wenuray"
LABEL org.opencontainers.image.description="Container image for Wenuray, radio-interferometric self-calibration"
LABEL org.opencontainers.image.licenses=GPL-3.0-or-later
