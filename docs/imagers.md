# Imagers

Every self-calibration round images the data, and the model of that image becomes the reference
the next `gaincal` solves against. Wenuray doesn't mind which algorithm makes that model, as long
as the imager keeps to one contract.

## The contract

An imager images `CORRECTED_DATA` and writes its model to `MODEL_DATA`. `Imager.image()` enforces
both and raises `ImagerContractError` otherwise.

It also raises `ValueError`, before imaging, when the cell is coarser than the Nyquist cell of the
selected data: the image grid can't hold the longest baselines, so the model written for them would
mislead the next solve. The message gives the largest cell that works, and the
`Selection(uvrange=...)` that leaves those baselines out of the solve, the apply and the image
alike.

**The grid can come from the data.** `ImageGeometry(cell="auto", imsize="auto")` derives the cell
from the Nyquist cell of the selected data, divided by `oversampling` (default 2.5, about 5 pixels
across the beam), and the size from the primary beam: `field_of_view` FWHMs (default 2.5, out to
the main lobe's first null), 1.02 λ/D at the lowest selected frequency with the smallest dish,
rounded up to a size FFTs handle well. Both factors are yours to change, in Python or a recipe
(`geometry = { cell = "auto", imsize = "auto", oversampling = 3 }`). A run derives the grid once,
when it starts, logs it, and records it in `selfcal.toml`'s provenance.

Imagers take two config objects, `ImageGeometry` (`imsize`, `cell`, ...) and `Weighting` (natural
by default), and pass every other option to the backend with the backend's own parameter names.

## tclean

`Tclean(geometry, weighting, **tclean_options)` runs CASA tclean, with tclean's own parameter
names. See the [tclean documentation](https://casadocs.readthedocs.io/en/stable/api/tt/casatasks.imaging.tclean.html)
and the [Automasking Guide](https://casaguides.nrao.edu/index.php/Automasking_Guide).

## Pyralysis

`Pyralysis(geometry, weighting, terms=[...], optimizer=..., **pyralysis_options)` is
optimization-based reconstruction with [Pyralysis](https://gitlab.com/clirai/pyralysis), with its
own names for terms (`"chi_squared"`, `"l1norm:1e-3"`), the optimizer (`"lbfgsb"`, `"fista"`) and
options (`max_iter`, `mask_mode`, ...). Each reconstruction runs in a separate process, since
Pyralysis's casacore and CASA's can't share one.

```{important}
**Give it a mask** (`mask_mode="file"`, `mask_path=...`): a model free in every pixel can absorb
the gain errors self-calibration should solve for.
```

Wenuray needs Pyralysis 3.2.1 or later, from PyPI; its `ska-ost-array-config` dependency comes
from SKA's index:

```bash
pip install "wenuray[pyralysis]" \
  --extra-index-url https://artefact.skao.int/repository/pypi-internal/simple
```

A `Selection(uvrange=...)` works with Pyralysis in metres or km (e.g. `"<3090m"`, the form the
Nyquist check suggests); a range in wavelengths is refused, since it depends on each channel's
frequency.

Numba needs OpenMP's runtime (`libgomp1`, present on most systems and in Wenuray's images) or TBB
for Pyralysis's parallel kernels; Wenuray checks for it when the imager is created.

## Your own imager

Subclass `Imager` and implement `_reconstruct`, wrap a function in `CallableImager`, or register it
under the `wenuray.imagers` entry point to build it by name (`type = "..."` in a recipe).
`wenuray.testing.ImagerContractSuite` checks that it follows the contract.

If your imager's model varies with frequency (a spectral index, Taylor terms, a cube), override the
`models_spectrum` property to return `True`. Otherwise Wenuray assumes a flat spectrum and warns
before amplitude steps that would absorb the source's spectrum.
