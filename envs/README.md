# Environments for Pyralysis

Pyralysis's pinned dependencies conflict with Wenuray's, so Pyralysis lives in an environment of its
own. Wenuray runs each Pyralysis reconstruction in a worker process with that environment's Python,
set by `WENURAY_PYRALYSIS_PYTHON`. Wenuray's own environment is `environment.yml`, at the
repository's root.

Each GPU generation needs its own file: each CUDA major release dropped the oldest architectures
(CUDA 12 dropped Kepler; CUDA 13 dropped Maxwell, Pascal and Volta), and a conda file has no optional
sections. These follow [Pyralysis's own](https://gitlab.com/clirai/pyralysis/-/tree/development/envs),
with Pyralysis's version and dependencies pinned to `requirements-pyralysis.txt`, the set Wenuray's
Docker images test.

| File | GPUs | Compute capability | CUDA | Minimum driver | CuPy |
| --- | --- | --- | --- | --- | --- |
| `pyralysis-cpu.yml` | none | - | - | - | - |
| `pyralysis-cuda11.yml` | Kepler (Tesla K40, K80) | 3.5, 3.7 | 11.4 | R470 | 13.x |
| `pyralysis-cuda12.yml` | Maxwell, Pascal, Volta (GTX 9xx and 10xx, P100, V100) | 5.0 to 7.2 | 12.9 | R525 | 14.x |
| `pyralysis-cuda13.yml` | Turing and newer (RTX 20xx and later, T4, A100, H100) | 7.5 and above | 13.0 | R580 | 14.x |

Check your card, its compute capability and its driver with:

```bash
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
```

A card that CUDA 13 supports also runs CUDA 12, if you would rather keep an older driver.

## Create and use

From the repository's root (`conda env create -f` and `mamba env create -f` work the same way):

```bash
micromamba create -f envs/pyralysis-cuda12.yml
export WENURAY_PYRALYSIS_PYTHON=$(micromamba run -n wenuray-pyralysis-cuda12 which python)
```

Then use Wenuray from its own environment as usual; its `Pyralysis` imager runs in this one. The
CUDA files install the toolkit and CuPy from conda-forge into the environment, so they need only
the NVIDIA driver on the host. Don't also `pip install` a CuPy wheel or Pyralysis's `cupy` extras
on top. If your shell exports `CUDA_PATH` to a system toolkit, unset it.

Wenuray's Docker images hold the same environment in `/opt/pyralysis`: on CPU in `:latest` and
on CUDA 12 in `:gpu` ([Installation](../docs/installation.md#with-docker)).
