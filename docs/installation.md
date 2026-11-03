# Installation

## Requirements

- **Python** `>=3.11, <3.14`: the versions with both CASA wheels and Pyralysis support. On 3.11,
  CASA 6.7.0 is the newest release with wheels; 3.12 and 3.13 get the latest CASA.
- **Dependencies**: casatools, casatasks (CASA >= 6.7), numpy, astropy and reproject. The
  package declares lower bounds;
  [`requirements.txt`](https://github.com/miguelcarcamov/wenuray/blob/development/requirements.txt)
  pins the set tested in CI and used in the Docker images.
- **CASA data**: CASA downloads its measures data the first time `casatools` is imported (see
  [External Data](https://casadocs.readthedocs.io/en/stable/notebooks/external-data.html)).
  Without network access, point `measurespath` to an existing copy.

```{note}
On Debian, Ubuntu or Arch, the `manylinux_2_34` casatools wheel that pip picks fails to import
(`OPENSSL_3.0.1 not found`). Install from `requirements.txt`, which uses the working
`manylinux_2_28` wheel, or install that wheel first.
```

## From PyPI

```bash
pip install wenuray
```

Release candidates are on PyPI too, and pip installs them only when asked:
`pip install --pre wenuray`.

Optional extras:

| Extra | For | Install |
| --- | --- | --- |
| `plots` | `GainPlotObserver` (`--plots` on the command line) | `pip install "wenuray[plots]"` |
| `pyralysis` | The Pyralysis imager | its own environment: see [below](#pyralysis-on-cpu-or-gpu) |
| `reproject` | `wenuray.utils.reproject()` | `pip install "wenuray[reproject]"` |

## From GitHub

```bash
pip install -U git+https://github.com/miguelcarcamov/wenuray.git
```

## From source

```bash
git clone https://github.com/miguelcarcamov/wenuray
cd wenuray
pip install .        # or `pip install -e .` for an editable, development install
```

## With conda

```bash
conda env create -f environment.yml
conda activate wenuray-env
```

The environment provides Python 3.12 and the pinned dependencies from `requirements.txt`; then
install Wenuray itself with `pip install -e .` (from a clone) or `pip install wenuray`.

## Pyralysis, on CPU or GPU

Pyralysis's pins conflict with Wenuray's, so it gets a conda environment of its own, which Wenuray's
Pyralysis imager runs in, whichever way you installed Wenuray. **Two environments**, then: Wenuray's
(above, or any of the other ways on this page), where you run `wenuray`, and Pyralysis's, which
Wenuray runs its reconstructions in. Don't install Wenuray in Pyralysis's.

[`envs/`](https://github.com/miguelcarcamov/wenuray/tree/master/envs) has one file for CPU and one
for each CUDA line. Each CUDA line needs a newer NVIDIA driver, and the newer lines dropped the
oldest GPUs, so take **the newest line your GPU and your driver both support**:

| File | CUDA | GPUs | Minimum driver |
| --- | --- | --- | --- |
| `pyralysis-cpu.yml` | - | none | - |
| `pyralysis-cuda11.yml` | 11.4 | Kepler to Ampere (K80, P100, V100, T4, RTX 30xx, A100) | R470 |
| `pyralysis-cuda12.yml` | 12.9 | Maxwell and newer (GTX 9xx and later) | R525 |
| `pyralysis-cuda13.yml` | 13.0 | Turing and newer (RTX 20xx and later, T4, A100, H100) | R580 |

For instance, an A100 takes `cuda13` on driver 580 or newer, `cuda12` on 525 to 579, and `cuda11`
on 470 to 524. `nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv` shows your GPU
and driver; the "CUDA Version" at the top right of `nvidia-smi` is the newest CUDA your driver
runs. A file your driver is too old for fails to solve, e.g. on driver 520 (CUDA 11.8):

```text
package cuda-version-13.0-... has constraint __cuda >=13 conflicting with __cuda-11.8-0
```

Take an older file then, or update the driver.

The whole setup, from a clone (`conda env create -f` works as `micromamba create -f`):

```bash
# Wenuray's environment, where you run wenuray
conda env create -f environment.yml
conda activate wenuray-env
pip install -e .
# Pyralysis's, which Wenuray runs its reconstructions in
conda env create -f envs/pyralysis-cuda12.yml
export WENURAY_PYRALYSIS_PYTHON=$(conda run -n wenuray-pyralysis-cuda12 which python)
wenuray run recipe.toml        # from wenuray-env
```

```{note}
`OPENSSL_3.0.1 not found` when `wenuray` imports casatools means pip installed casatools's
`manylinux_2_34` wheel, which can't load in a conda environment (or on Debian, Ubuntu, Arch).
`environment.yml` and `requirements.txt` install the `manylinux_2_28` wheel, which can; so does
`pip install -r requirements.txt` in any environment. It's also the error you get from installing
Wenuray in Pyralysis's environment.
```

Installed from PyPI, without a clone, download the file and the pins it installs,
`requirements-pyralysis.txt`, keeping their layout (the file names the pins as
`../requirements-pyralysis.txt`):

```bash
mkdir -p wenuray-envs/envs && cd wenuray-envs
base=https://raw.githubusercontent.com/miguelcarcamov/wenuray/master
curl -fsSL -o requirements-pyralysis.txt "$base/requirements-pyralysis.txt"
curl -fsSL -o envs/pyralysis-cuda12.yml "$base/envs/pyralysis-cuda12.yml"
micromamba create -f envs/pyralysis-cuda12.yml
export WENURAY_PYRALYSIS_PYTHON=$(micromamba run -n wenuray-pyralysis-cuda12 which python)
```

For a release other than the latest, replace `master` with its tag (e.g. `v1.2.0`).

The CUDA environments bring the toolkit and CuPy from conda-forge, so the host needs only the NVIDIA
driver. [`envs/README.md`](https://github.com/miguelcarcamov/wenuray/blob/master/envs/README.md)
has each file's compute capabilities and CuPy version. The [Docker images](#with-docker) come
with this environment already.

## With Docker

Two images, both with tclean and Pyralysis:

| Image | For | Size (unpacked) |
| --- | --- | --- |
| `ghcr.io/miguelcarcamov/wenuray:latest` | Pyralysis on CPU | about 1.7 GB |
| `ghcr.io/miguelcarcamov/wenuray:gpu` | Pyralysis on CUDA 12 | about 4.8 GB |

```bash
docker pull ghcr.io/miguelcarcamov/wenuray:latest
docker run --rm -v "$PWD":/work -w /work ghcr.io/miguelcarcamov/wenuray:latest \
  wenuray run recipe.toml
```

Each image has Python 3.12; Wenuray with its pinned dependencies in `/opt/venv`; Pyralysis in an
environment of its own, `/opt/pyralysis` (its pins conflict with Wenuray's, and its worker runs with
that Python, set by `WENURAY_PYRALYSIS_PYTHON`); and CASA's measures data and observatory table in
`/opt/casa/data`, readable by any user (also under Apptainer/Singularity). To stay small it skips the
rest of casarundata (antenna responses and simulator files, used by e.g. the `awproject` gridder);
mount a full copy at `/opt/casa/data` if you need it.

**On a GPU**, `:gpu` adds CuPy and numba-cuda with the CUDA 12 libraries as NVIDIA's pip wheels, so
the host needs only the NVIDIA driver (525 or later, which supports Pascal GPUs and newer) and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/):

```bash
docker run --rm --gpus all -v "$PWD":/work -w /work ghcr.io/miguelcarcamov/wenuray:gpu \
  wenuray run recipe.toml
```

Apptainer/Singularity runs it with `--nv`. Pyralysis's own options select its GPU backend.

CASA needs a writable home directory. Apptainer/Singularity provides yours; with `docker run -u`,
also pass `-e HOME=/tmp`.
