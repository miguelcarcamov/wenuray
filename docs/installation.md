# Installation

## Requirements

- **Python** `>=3.11, <3.14`: the versions with both CASA wheels and Pyralysis support. On 3.11,
  CASA 6.7.0 is the newest release with wheels; 3.12 and 3.13 get the latest CASA.
- **Dependencies**: casatools, casatasks (CASA >= 6.7), numpy and astropy; Pyralysis for its
  imager. The package declares lower bounds;
  [`requirements.txt`](https://github.com/miguelcarcamov/wenuray/blob/development/requirements.txt)
  pins the set tested in CI, with Pyralysis, which the conda files and the Docker images install.
- **CASA data**: CASA downloads its measures data the first time `casatools` is imported (see
  [External Data](https://casadocs.readthedocs.io/en/stable/notebooks/external-data.html)).
  Without network access, point `measurespath` to an existing copy.

```{note}
On Debian, Ubuntu or Arch, the `manylinux_2_34` casatools wheel that pip picks fails to import
(`OPENSSL_3.0.1 not found`), and so it does in a conda environment. The conda files and
`requirements.txt` install the working `manylinux_2_28` wheel; with pip, install that wheel first.
```

## With conda or micromamba

One command makes an environment with everything: Wenuray, CASA and Pyralysis, and on a GPU CUDA's
toolkit and CuPy. Take the file for your machine:

```bash
conda env create -f environment.yml          # CPU
conda env create -f envs/wenuray-cuda12.yml  # or a GPU: envs/wenuray-cuda11.yml, -cuda12, -cuda13
conda activate wenuray                       # wenuray-cuda12, ...
wenuray run recipe.toml
```

`micromamba create -f ...` works the same way. Each file stands alone: without a clone, download
just that one, e.g.
`curl -fsSLO https://raw.githubusercontent.com/miguelcarcamov/wenuray/master/envs/wenuray-cuda12.yml`.
They install the latest Wenuray release from PyPI; from a clone, `pip install -e . --no-deps` in
the environment installs your checkout instead.

### Which GPU file

Each CUDA line needs a newer NVIDIA driver, and the newer lines dropped the oldest GPUs: take **the
newest line your GPU and your driver both support**.

| File | CUDA | GPUs | Minimum driver |
| --- | --- | --- | --- |
| `envs/wenuray-cuda11.yml` | 11.8 | Kepler to Ada (K80, P100, V100, T4, RTX 30xx, A100, RTX 40xx) | R520 |
| `envs/wenuray-cuda12.yml` | 12.9 | Maxwell and newer (GTX 9xx and later) | R525 |
| `envs/wenuray-cuda13.yml` | 13.0 | Turing and newer (RTX 20xx and later, T4, A100, H100) | R580 |

An A100, for instance, takes `cuda13` on driver 580 or newer, `cuda12` on 525 to 579, and `cuda11`
on 520 to 524. `nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv` shows your GPU
and driver; the "CUDA Version" at the top right of `nvidia-smi` is the newest CUDA your driver runs.
A file your driver is too old for fails to solve, e.g. on driver 520 (CUDA 11.8):

```text
package cuda-version-13.0-... has constraint __cuda >=13 conflicting with __cuda-11.8-0
```

Take an older file then, or update the driver. The host needs only the driver: the environment
brings CUDA's toolkit, and Wenuray runs Pyralysis with it (`CUDA_PATH` set to the environment), not
with a system CUDA the host may also have. In a recipe, `array_backend = "cupy"` under `[imager]`
runs Pyralysis on the GPU.

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
| `pyralysis` | The Pyralysis imager | `pip install "wenuray[pyralysis]" --extra-index-url https://artefact.skao.int/repository/pypi-internal/simple` (its `ska-ost-array-config` comes from SKA's index) |
| `reproject` | `wenuray.utils.reproject()` | `pip install "wenuray[reproject]"` |

## From GitHub

```bash
pip install -U git+https://github.com/miguelcarcamov/wenuray.git
```

## From source

```bash
git clone https://github.com/miguelcarcamov/wenuray
cd wenuray
pip install -r requirements.txt -e .
```

`requirements.txt` is the pinned set CI tests, with Pyralysis; `pip install .` alone resolves the
dependencies freely, without Pyralysis.

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

Each image has Python 3.12; Wenuray, CASA and Pyralysis with their pinned dependencies in
`/opt/venv`; and CASA's measures data and observatory table in
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
