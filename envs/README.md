# Conda environments for a GPU

Each file makes one environment with Wenuray, CASA and Pyralysis, and CUDA's toolkit and CuPy from
conda-forge, so the host needs only NVIDIA's driver. On CPU, use `environment.yml` at the
repository's root. The files are generated from `requirements.txt` by `tools/sync_envs.py`: edit
that file and run the script, not the files.

```bash
conda env create -f envs/wenuray-cuda12.yml    # or micromamba create -f ...
conda activate wenuray-cuda12
wenuray run recipe.toml                        # with array_backend = "cupy" under [imager]
```

Each file stands alone, so it can be downloaded without a clone. They install the latest Wenuray
release from PyPI; from a clone, `pip install -e . --no-deps` in the environment installs your
checkout instead.

## Which file

Each CUDA line needs a newer driver, and the newer lines dropped the oldest GPUs: take the newest
line your GPU **and** your driver both support.

| File | CUDA | GPUs | Compute capability | Minimum driver | CuPy |
| --- | --- | --- | --- | --- | --- |
| `wenuray-cuda11.yml` | 11.4 | Kepler to Ampere (K80, P100, V100, T4, RTX 30xx, A100) | 3.5 to 8.6 | R470 | 13.x |
| `wenuray-cuda12.yml` | 12.9 | Maxwell and newer (GTX 9xx and later) | 5.0 and above | R525 | 14.x |
| `wenuray-cuda13.yml` | 13.0 | Turing and newer (RTX 20xx and later, T4, A100, H100) | 7.5 and above | R580 | 14.x |

An A100, for instance, takes `cuda13` on driver R580 or newer, `cuda12` on R525 to R579, and
`cuda11` on R470 to R524. Check your GPU, its compute capability and its driver with:

```bash
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
```

The "CUDA Version" at the top right of `nvidia-smi` is the newest CUDA your driver runs. A file your
driver is too old for fails to solve:

```text
package cuda-version-13.0-... has constraint __cuda >=13 conflicting with __cuda-11.8-0
```

Take an older file, or update the driver.

The environment shadows a system CUDA while it is active: unset `CUDA_PATH` if your shell exports
one. Don't also `pip install` a CuPy wheel or Pyralysis's `cupy` extras on top. Wenuray's `:gpu`
Docker image has the same packages, with CUDA 12 from NVIDIA's pip wheels instead
([Installation](../docs/installation.md#with-docker)).
