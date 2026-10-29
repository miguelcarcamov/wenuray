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
| `pyralysis` | The Pyralysis imager | see [Imagers](imagers.md#pyralysis) |
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

## With Docker

```bash
docker pull ghcr.io/miguelcarcamov/wenuray:latest
```

The image has Python 3.12, Wenuray with its pinned dependencies, and CASA's measures data and
observatory table in `/opt/casa/data`, readable by any user (also under Apptainer/Singularity). To
stay small it has no CUDA, and it skips the rest of casarundata (antenna responses and simulator
files, used by e.g. the `awproject` gridder); mount a full copy at `/opt/casa/data` if you need it.

```bash
docker run --rm -v "$PWD":/work -w /work ghcr.io/miguelcarcamov/wenuray:latest \
  wenuray run recipe.toml
```

CASA needs a writable home directory. Apptainer/Singularity provides yours; with `docker run -u`,
also pass `-e HOME=/tmp`.
