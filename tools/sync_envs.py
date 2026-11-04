"""
Regenerate the conda files (environment.yml, envs/wenuray-*.yml) from requirements.txt, the pinned
set (#170). Each file is self-contained, so it works from a clone or downloaded alone: the
interpreter, NumPy and, on a GPU, CUDA's toolkit and CuPy from conda-forge; then every pin and
Wenuray itself from pip. `--check` fails when a file differs from what this script writes.
"""

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "requirements.txt"
SKA_INDEX = "https://artefact.skao.int/repository/pypi-internal/simple"
# Numba's CUDA target, a package of its own since Numba 0.61; it supports CUDA 12 and 13 only
NUMBA_CUDA = "numba-cuda==0.30.4"


@dataclass(frozen=True)
class Platform:
    """
    One conda file: where it goes, its environment's name, what it says it is for, and what it adds
    from conda-forge and pip
    """
    path: str
    name: str
    about: Tuple[str, ...]
    conda: Tuple[str, ...] = ()
    pip: Tuple[str, ...] = ()


PLATFORMS = (
    Platform(
        "environment.yml", "wenuray",
        ("Wenuray with CASA and Pyralysis, on CPU. For a GPU, use a file in envs/ instead.", )
    ),
    Platform(
        "envs/wenuray-cuda11.yml",
        "wenuray-cuda11",
        (
            "Wenuray with CASA and Pyralysis on a GPU, with CUDA 11.8: driver R520 or newer, GPUs",
            "from Kepler to Ada (compute capability 3.5 to 8.9: K80, P100, V100, T4, RTX 30xx, A100,",
            "RTX 40xx). For drivers R520 to R524, older than CUDA 12 needs, and Kepler GPUs, which",
            "CUDA 12 dropped. CuPy 14 dropped CUDA 11, hence CuPy 13, built against CUDA 11.8, the",
            "toolkit pinned here (#173); Numba's CUDA target no longer supports CUDA 11, so",
            "Pyralysis's GPU work runs through CuPy only.",
        ),
        conda=("cuda-version=11.8", "cudatoolkit=11.8", "cupy>=13.3,<14"),
    ),
    Platform(
        "envs/wenuray-cuda12.yml",
        "wenuray-cuda12",
        (
            "Wenuray with CASA and Pyralysis on a GPU, with CUDA 12.9: driver R525 or newer, GPUs",
            "from Maxwell on (compute capability 5.0 and above). For Maxwell, Pascal and Volta",
            "GPUs (GTX 9xx and 10xx, P100, V100), which CUDA 13 dropped, and drivers R525 to R579.",
        ),
        conda=("cuda-version=12.9", "cuda", "cupy>=14.0,<15"),
        pip=(NUMBA_CUDA, ),
    ),
    Platform(
        "envs/wenuray-cuda13.yml",
        "wenuray-cuda13",
        (
            "Wenuray with CASA and Pyralysis on a GPU, with CUDA 13.0: driver R580 or newer, GPUs",
            "from Turing on (compute capability 7.5 and above: RTX 20xx and later, T4, A100, H100).",
        ),
        conda=("cuda-version=13.0", "cuda", "cupy>=14.0,<15"),
        pip=(NUMBA_CUDA, ),
    ),
)


def pins(text: str) -> Tuple[str, ...]:
    """
    The requirement lines of requirements.txt, without its comments and options (the conda files
    give SKA's index themselves)
    """
    return tuple(
        line.strip() for line in text.splitlines()
        if line.strip() and not line.startswith(("#", "--"))
    )


def numpy_version(lines: Tuple[str, ...]) -> str:
    for line in lines:
        match = re.fullmatch(r"numpy==(\S+)", line)
        if match:
            return match.group(1)
    raise ValueError("requirements.txt pins no numpy")


def render(platform: Platform, lines: Tuple[str, ...]) -> str:
    about = "\n".join("# " + line for line in platform.about)
    conda = "".join("- {0}\n".format(package) for package in platform.conda)
    gpu = "  # CUDA's toolkit and CuPy\n" + conda if conda else ""
    pip = "".join("  - {0}\n".format(line) for line in lines + platform.pip)
    return """{about}
#
# Generated from requirements.txt by tools/sync_envs.py: edit that file and run the script.
#
# Create:  conda env create -f {path}   (or micromamba create -f ...)
# Use:     conda activate {name}, then wenuray run recipe.toml
# From a clone, pip install -e . --no-deps in it installs your checkout instead of the release.
#
# On a GPU, CUDA's toolkit and CuPy come from conda-forge into the environment, which shadows a
# system CUDA while it is active (unset CUDA_PATH if the shell exports one): the host needs only
# NVIDIA's driver. Don't also pip install a cupy-* wheel or Pyralysis's cupy extras on top.
name: {name}

channels:
- conda-forge

dependencies:
  # requirements.txt's interpreter and NumPy, so pip installs one NumPy only
- python=3.12
- numpy={numpy}
- pip
{gpu}- pip:
  # ska-ost-array-config and the ska-* packages come from SKA's index
  - --extra-index-url {index}
{pip}  - wenuray
""".format(
        about=about,
        path=Path(platform.path).name if platform.path == "environment.yml" else platform.path,
        name=platform.name,
        numpy=numpy_version(lines),
        gpu=gpu,
        index=SKA_INDEX,
        pip=pip
    )


def main() -> int:
    lines = pins(REQUIREMENTS.read_text(encoding="utf-8"))
    stale = []
    for platform in PLATFORMS:
        path = ROOT / platform.path
        text = render(platform, lines)
        if "--check" in sys.argv[1:]:
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                stale.append(platform.path)
        else:
            path.write_text(text, encoding="utf-8")
    if stale:
        print(
            "Out of date with requirements.txt (run python tools/sync_envs.py): " +
            ", ".join(stale)
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
