# Base image: Python, Wenuray's pinned dependencies with Pyralysis and CASA's measures data, kept
# small.
# - Two variants of one Dockerfile (#148): the default builds :base, Pyralysis on CPU;
#   --build-arg PYRALYSIS=pyralysis-gpu builds :base-gpu, which adds CUDA 12 for Pyralysis
#   (`docker run --gpus all`; the host needs only the NVIDIA driver, >= 525).
# - One environment, /opt/venv, holds Wenuray's dependencies and Pyralysis, pinned together in
#   requirements.txt (#170). Pyralysis's worker runs in a process of its own, with the same Python.
# - The casatools wheels bundle casacore, so nothing is compiled.
# - Python 3.12 is supported by both CASA (>= 6.7.2) and Pyralysis (#30).
# - The builder stages install, strip and prune; the measures stage fetches CASA's data; the
#   final image only copies their results, without the build tools.
ARG PYTHON_IMAGE=python:3.12-slim-bookworm
# The Pyralysis environment the final image takes: pyralysis (CPU) or pyralysis-gpu
ARG PYRALYSIS=pyralysis

FROM ${PYTHON_IMAGE} AS builder

# binutils to strip casatools; OpenMP's runtime for the Numba check below
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends binutils libgomp1 && \
    rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH=/opt/venv/bin:$PATH
RUN python -m venv /opt/venv

# The pinned dependency set that CI tests (it names SKA's index, for the ska-* packages). Then, to
# save space:
# - strip the debug symbols left in some libraries bundled with casatools (~30 MB). Nothing else is
#   stripped: NumPy's OpenBLAS and Numba's threading layers were rewritten by auditwheel, and strip
#   breaks them ("ELF load command address/offset not page-aligned");
# - drop the test suites of the largest packages. astropy imports its own tests package at startup,
#   so it is kept.
# The check runs a parallel Numba function on the OpenMP layer, as Pyralysis's kernels do.
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && \
    site=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])") && \
    find "$site/casatools" -type f -name "*.so*" -exec strip --strip-unneeded {} + && \
    find "$site/numpy" "$site/scipy" "$site/matplotlib" "$site/mpl_toolkits" "$site/pandas" \
        "$site/numba" "$site/llvmlite" -depth -type d -name tests -exec rm -rf {} + && \
    NUMBA_THREADING_LAYER=omp python -c "import casaconfig, matplotlib, pyralysis, casacore.tables, \
scipy.linalg, numba, numpy; numba.njit(parallel=True)(lambda a: a.sum())(numpy.ones(8)); \
print('numba', numba.threading_layer())"

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

# Pyralysis on CPU: the builder's environment as it is
FROM builder AS pyralysis

FROM builder AS pyralysis-gpu

# CUDA 12 through CuPy and numba-cuda, with the CUDA libraries as NVIDIA's pip wheels (about 3 GB as
# installed): no nvidia/cuda base image. See requirements-gpu.txt for why the whole toolkit is kept.
COPY requirements-gpu.txt /tmp/requirements-gpu.txt
RUN pip install --no-compile -r /tmp/requirements-gpu.txt && \
    python -c "import cupy, cupyx.scipy.sparse, numba.cuda"

# The variant the final image takes (a FROM can't name a stage through a variable in COPY --from)
FROM ${PYRALYSIS} AS pyralysis-selected

FROM ${PYTHON_IMAGE}

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH=/opt/venv/bin:$PATH \
    VIRTUAL_ENV=/opt/venv \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# OpenMP's runtime: Numba's thread-safe threading layer, which Pyralysis (#24) needs to run its
# parallel kernels from several Dask threads (without it, Numba's fallback aborts the process)
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=pyralysis-selected /opt/venv /opt/venv
# Auto updates are off, so every container of an image uses the same data. The directory is
# readable by any user (e.g. under Apptainer/Singularity).
COPY --from=measures /opt/casa /opt/casa

# CASA looks telescopes up by name in its observatory table (e.g. for frame conversions)
RUN python -c "import casatasks, casatools; print('casatools', casatools.version_string()); \
me = casatools.measures(); assert all(me.observatory(t) for t in ('VLA', 'ALMA'))" && \
    python -c "import importlib.metadata as m, pyralysis, numba; \
print('pyralysis', m.version('pyralysis'))"

LABEL org.opencontainers.image.source="https://github.com/miguelcarcamov/wenuray"
LABEL org.opencontainers.image.description="Container image for Wenuray, radio-interferometric self-calibration"
LABEL org.opencontainers.image.licenses=GPL-3.0-or-later
