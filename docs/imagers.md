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

### Gridding

Pyralysis fits χ² on the visibilities, row by row, but predicts each row's model visibility, and
the χ² gradient, from an FFT of the model on a padded grid, interpolated at the row's (u, v). Its
defaults, bilinear interpolation on a grid padded by 1.2, are off by a few percent of the peak
around bright emission ([clirai/pyralysis#414](https://gitlab.com/clirai/pyralysis/-/work_items/414)),
and a model fitted through them makes up for it. A convolution kernel tabulated finely, on a grid
padded by 2, is within 0.06% of the exact transform (tclean's default is within 0.02%):

```python
Pyralysis(geometry, weighting, terms=[...], optimizer="lbfgsb",
          estimator="degridding", ckernel="gaussian-sinc", ckernel_oversampling=8,
          padding_factor=2.0)
```

Each iteration then costs more: the grid has about 2.8 times as many cells. Of the kernels
Pyralysis's options build, all 3 cells wide, Gaussian-sinc is the most accurate; its prolate
spheroidal (`pswf1`) needs 7 cells, as tclean's, which the options can't set yet.

Pyralysis 3.3.0 grids the residual image, which gives the round's noise and peak S/N, at padding 1.2
without a kernel whatever these options say, so it keeps a few percent of error around bright
emission until Pyralysis grids it with them.

Inside its mask, a model with a weak regularisation also fits the noise: the residual there falls
below the thermal noise, which flatters the round's peak S/N. A stronger regularisation term, a
tighter mask or fewer iterations keep the noise in the residual. Wenuray warns when the residual
inside the source mask falls below 0.9 times the thermal noise, but doesn't know Pyralysis's mask
yet ([#167](https://github.com/miguelcarcamov/wenuray/issues/167)).

Wenuray needs Pyralysis 3.2.1 or later, from PyPI; its `ska-ost-array-config` dependency comes
from SKA's index:

```bash
pip install "wenuray[pyralysis]" \
  --extra-index-url https://artefact.skao.int/repository/pypi-internal/simple
```

A `Selection(uvrange=...)` works with Pyralysis in metres or km (e.g. `"<3090m"`, the form the
Nyquist check suggests); a range in wavelengths is refused, since it depends on each channel's
frequency.

Pyralysis pins exact versions of its dependencies, which may conflict with Wenuray's. Since each
reconstruction runs in a worker process that imports neither Wenuray nor CASA, Pyralysis can live in
an environment of its own: set `WENURAY_PYRALYSIS_PYTHON` to that environment's Python (e.g.
`/opt/pyralysis/bin/python`, as in Wenuray's images). The conda files in `envs/` build that
environment on CPU or GPU ([Installation](installation.md#pyralysis-on-cpu-or-gpu)).
The run record takes Pyralysis's version, and
its environment's, from the worker.

Numba needs OpenMP's runtime (`libgomp1`, present on most systems and in Wenuray's images) or TBB
for Pyralysis's parallel kernels; Wenuray checks for it when the imager is created.

## Your own imager

Subclass `Imager` and implement `_reconstruct`, wrap a function in `CallableImager`, or register it
under the `wenuray.imagers` entry point to build it by name (`type = "..."` in a recipe).
`wenuray.testing.ImagerContractSuite` checks that it follows the contract.

If your imager's model varies with frequency (a spectral index, Taylor terms, a cube), override the
`models_spectrum` property to return `True`. Otherwise Wenuray assumes a flat spectrum and warns
before amplitude steps that would absorb the source's spectrum.
