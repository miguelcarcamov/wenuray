# Migrating to Wenuray 1.0

Wenuray 1.0 replaces the self-calibration and imaging classes (#22, #23). A procedure is now one
`Selfcal` with a list of steps, instead of chained `Phasecal`/`Ampcal`/`AmpPhasecal` objects that
each ran their own loop. Code written for snow < 1.0 raises a `TypeError` that shows the new call.

## The new name

snow is now **Wenuray**. Install `wenuray` instead of `snow-selfcal`, import `wenuray` instead of
`snow`, and run the `wenuray` command instead of `snow`. Imagers of your own register under the
`wenuray.imagers` entry point group instead of `snow.imagers`, and the container images are
`ghcr.io/miguelcarcamov/wenuray`. Flag versions saved during a run are named `wenuray_selfcal_...`.

## Before and after

```python
# snow < 1.0
imager = Tclean(inputvis="target.ms", output="target_sc", M=1024, N=1024, cell="0.005arcsec",
                robust=0.5, niter=100, use_mask="auto-multithresh", sidelobe_threshold=3.0)
phscal = Phasecal(visfile="target.ms", imager=imager, solint=["inf", "600s"], refant="DA51",
                  combine="spw", want_plot=True)
phscal.run()
apcal = AmpPhasecal(previous_selfcal=phscal, imager=imager, solint=["inf"], refant="DA51")
apcal.run()
apcal.selfcal_output(overwrite=True)
```

```python
# Wenuray 1.0
from casatasks import split
from wenuray.config import ImageGeometry, SolveOptions, Weighting
from wenuray.imaging import Tclean
from wenuray.selfcalibration import AmpPhasecal, GainPlotObserver, LoggingObserver, Phasecal, Selfcal

imager = Tclean(ImageGeometry(imsize=1024, cell="0.005arcsec"), Weighting("briggs", robust=0.5),
                niter=100, usemask="auto-multithresh", sidelobethreshold=3.0)
result = Selfcal(
    "target.ms",
    imager,
    [Phasecal("inf", solve={"combine": "spw"}), Phasecal("600s", solve={"combine": "spw"}),
     AmpPhasecal("inf")],
    solve=SolveOptions(refant="DA51"),
    output="target_sc",
).run(observers=[LoggingObserver(), GainPlotObserver()])
split(vis=result.ms, outputvis="target_sc/selfcal.ms", datacolumn="corrected")
```

## Self-calibration

| snow < 1.0 | Wenuray 1.0 |
| --- | --- |
| `Phasecal(visfile=..., imager=..., solint=["inf", "60s"], ...)` | `Selfcal(ms, imager, [Phasecal("inf"), Phasecal("60s")])` |
| `previous_selfcal=phscal` | one schedule with all the steps, or `Selfcal.resume(result, imager, steps)` |
| `input_caltable="pcal", spwmap=[...]` | `initial_stack=CalStack([CalTable("pcal", "p", spwmap)])` |
| `refant`, `minblperant`, `minsnr`, `gaintype`, `combine` | `solve=SolveOptions(...)`; a step overrides them with `solve={...}` |
| `field`, `spw`, `uvrange` | `selection=Selection(...)`, shared by the solve, the apply and the imager |
| `applymode`, `interp` | `apply=ApplyOptions(...)` |
| `spwmap` | derived for each table from `combine` |
| `varchange_imager={"nsigma": [3.0, 2.5]}` | a step's `imaging={"nsigma": 2.5}` |
| `varchange_selfcal` | a step's `solve={...}` |
| `restore_psnr=True` | the default: a round is kept only if the acceptance policy accepts it, and a rejected round is rolled back. PSNR alone never decides |
| `want_plot=True` | `run(observers=[LoggingObserver(), GainPlotObserver()])` |
| `output_caltables` | `output=...`, a directory for the tables, the images and `selfcal.toml` |
| `selfcal_output()` | `split(vis=result.ms, outputvis=..., datacolumn="corrected")` |
| `solnorm` | `Ampcal(..., solnorm=False)`; `True` by default |
| `flag_dataset=True`, `flag_mode` | a step's `flagging=ResidualFlagging("rflag", timedevscale=3.0, ...)`, with `flagdata`'s names and defaults; it flags the residuals after the step's apply, and a rejected round's flags are rolled back |
| `subtract_source` | removed; it did nothing |

The measurement set is no longer copied for every round: a rejected round restores the flags and
recalibrates `CORRECTED_DATA` with the accepted tables. Everything a run writes goes to `output`.

## Imaging

| snow < 1.0 | Wenuray 1.0 |
| --- | --- |
| `Tclean(inputvis=..., output=...)` | the MS and the output come from `Selfcal` |
| `M`, `N`, `cell`, `phase_center`, `stokes` | `ImageGeometry(imsize=(M, N), cell=..., phasecenter=..., stokes=...)` |
| `weighting`, `robust`, `uvtaper` | `Weighting(...)`; the default is now **natural**, so set `Weighting("briggs", robust=...)` to keep earlier results |
| `use_mask`, `sidelobe_threshold`, `noise_threshold`, `low_noise_threshold`, `negative_threshold`, `min_beam_frac`, `grow_iterations`, `cycle_niter`, `wproj_planes`, `reference_freq` | tclean's own names: `usemask`, `sidelobethreshold`, `noisethreshold`, `lownoisethreshold`, `negativethreshold`, `minbeamfrac`, `growiterations`, `cycleniter`, `wprojplanes`, `reffreq` |
| `data_column`, `save_model` | always `CORRECTED_DATA` in and `MODEL_DATA` out: the imager contract |
| `imager.psnr`, `imager.peak`, `imager.stdv` | `result.rounds[i].psnr`, `.peak`, `.rms`, computed the same way for every imager |
| a subclass of `Imager` overriding `run()` | a subclass implementing `_reconstruct(request)`, or `CallableImager(function)`; test it with `wenuray.testing.ImagerContractSuite` |
| `WSClean` | removed; it was never implemented |
