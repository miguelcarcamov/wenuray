# How a run works

## Preparing the data

- **Calibrated data in `DATA`.** `gaincal` solves against `DATA`, while the imager images
  `CORRECTED_DATA`. Split the calibrated target first
  (`split(vis=..., outputvis=..., datacolumn="corrected")`). A run resets `CORRECTED_DATA` to
  `DATA` (plus `initial_stack`) and warns if it held something else.
- **One selection.** `Selection(field=..., spw=..., uvrange=...)` is used by the solve, the apply
  and the imager, so the model is always compared with the same visibilities it was made from.
- **`applymode`.** The default `ApplyOptions(applymode="calflag")` flags data whose solutions fail
  `minsnr`. `"calonly"` leaves those data uncalibrated but unflagged.

## The schedule

- **One `Selfcal`, a list of steps.** Each step is one round: `Phasecal`, `Ampcal` or
  `AmpPhasecal` with its solution interval. A step can override the run's solve options
  (`solve={"minsnr": 5.0}`), the imager's options (`imaging={"nsigma": 2.0}`) or the acceptance
  policy (`acceptance=...`), and can flag residual outliers before imaging
  (`flagging=ResidualFlagging("rflag")`, off by default).
- **Invalid schedules fail before any I/O.** An amplitude step needs a phase solution before it,
  in the schedule or in `initial_stack`; otherwise `Selfcal(...)` raises `ScheduleError`. Solving
  amplitudes against data whose phases are still wrong would fold the phase errors into the
  amplitudes.
- **Every accepted table stays applied.** Each solve pre-applies all accepted tables, each with its
  own `spwmap` (derived from `combine`) and `interp`.

## Each round

- **Each solve is checked first.** Before `gaincal`, Wenuray predicts a solution's S/N for the
  step's solint and combine from the data: the median model amplitude over the noise of one
  visibility, times √(N_ant − 1) · √(n_int · n_chan · n_pol). It warns when the prediction is
  below `minsnr`, giving the shortest solint that would reach it, and records it as
  `predicted_snr`.
- **Amplitudes and the source's spectrum.** An amplitude step that solves per spectral window
  against a model without a spectrum lets each window's gain take up the source's flux at that
  frequency: the residual drops, but the calibrated data lose their spectral index. A spectral
  index α changes the amplitude by about α times the fractional bandwidth, so Wenuray warns when
  the solutions are precise enough (1 / S/N below the fractional bandwidth) to absorb α = 1. Give
  the imager a spectral model (tclean's `deconvolver="mtmfs"`, Pyralysis's `spectral_index`) or
  solve with `combine="spw"`.
- **The noise floor.** After the initial image, Wenuray estimates the image's thermal noise from
  the data: the noise of one visibility, from differences of consecutive integrations, averaged
  down by the data weights (σ · √Σw² / Σw). It is exact for natural weighting and a floor for
  briggs or uniform. Each round logs and records its residual noise against it (`thermal_noise`,
  and `rms_in_mask` inside the source mask). A residual inside the mask below 0.9× thermal is
  warned about: the model and the gains are fitting noise there. Image noise is always the MAD
  standard deviation, since it is correlated by the beam and isn't Gaussian.
- **A round is kept only if it improves the image.** By default the PSNR must rise and every
  solution interval must keep enough antennas (3 for phases, 4 for amplitudes), and amplitude
  rounds must not lose model flux. A rejected round is rolled back and stops the run.
- **Continue later** with `Selfcal.resume(result_or_output_directory, imager, more_steps)`, or
  start from tables of your own with `initial_stack=CalStack([CalTable("pcal", "p")])`.
- **`result.final`** is the last accepted round; every round's metrics, decision and reasons are in
  `result.rounds` and in `<output>/selfcal.toml`.

## Calibrating other data

Self-calibration usually runs on the continuum; the accepted tables then go to the line data,
other spectral windows, or another measurement set of the same observation. List them in the run:

```toml
[[apply.targets]]
ms = "line.ms"            # relative to the recipe
selection = { spw = "" }  # the spectral windows that hold data
```

or, in Python, `ApplyOptions(targets=(ApplyTarget("line.ms"), ))`. When the run ends, each target
gets the accepted stack with the run's `applymode`. After a run, `result.apply_to("line.ms")` or
`wenuray apply target_selfcal --ms line.ms` does the same. Each application is recorded in
`selfcal.toml` (`[[applications]]`: the target, its selection, the tables with their spwmaps, and a
fingerprint of the target).

A split measurement set renumbers its spectral windows, so they are matched to the tables' by
frequency, never by id:

- a table solved with `combine="spw"` holds one solution for the band, which every target spectral
  window takes;
- a table solved per spectral window calibrates a target spectral window from the solved one that
  overlaps it in frequency. If none does, Wenuray raises an error naming the nearest one: per-window
  gains at another frequency would be a guess. Solve with `combine="spw"` to calibrate other
  frequencies.

The target must hold the tables' antennas, by the same ids, and overlap their times.

## Recipes

A recipe is the same objects written as TOML: each table is a constructor and its keys are that
class's arguments, so a recipe and a script build equal objects and raise the same errors. The
imager, the steps, the acceptance policies and the flagging name their class with `type`.

```python
selfcal = Selfcal.from_toml("recipe.toml")  # an ordinary Selfcal: inspect it, change it, run it
selfcal.to_toml("copy.toml")                # and back; values equal to the defaults are left out
```

From the command line:

```bash
wenuray init > recipe.toml                 # a generic recipe to start from
wenuray validate recipe.toml               # builds the run and checks it, without touching the data
wenuray show --defaults recipe.toml        # the resolved recipe: presets laid in, every value shown
wenuray run recipe.toml --plots            # runs it, plotting the gain solutions
wenuray run recipe.toml --ms other.ms      # the same procedure on another dataset
```

Errors are the same as from Python, e.g. a misspelt `refant` or an amplitude step before any phase
step, and `wenuray` exits with status 1.

`preset = "group.toml"` lays a recipe over another TOML file, e.g. the solve options and imager
settings your group shares: tables merge key by key and the recipe wins. Wenuray ships no presets,
since its defaults come from theory or the data rather than from an instrument. A class of your
own, e.g. an acceptance policy, is named as `type = "package.module:Class"`. Policies built with
`Check(name, function)` can't be written to a recipe; make them a class.

## Reproducing a run

Every run records how to repeat it in its output directory:

- **`recipe.toml`**: the run as a resolved recipe, written before anything runs, with presets laid
  in and every default written out, so later changes to Wenuray's defaults can't change it.
- **`selfcal.toml`**: the rounds and accepted tables, plus a `[provenance]` table with the start
  time, a fingerprint of the input measurement set (a hash of its metadata and of sampled `DATA`
  and `FLAG` rows) and the versions of Python, Wenuray, CASA and the imager's packages.

```bash
wenuray run target_selfcal/recipe.toml --ms fresh_copy.ms --output rerun
```

reruns it on a fresh copy of the input. A run Wenuray can't write as a recipe, e.g. one with a
`Check(name, lambda ...)` policy, still runs; it logs a warning and records no recipe.

## API overview

### Self-calibration (`wenuray.selfcalibration`)

- **`Selfcal`**: the self-calibration loop; `run()` returns a `SelfcalResult`, and `resume()`
  continues one.
- **`Phasecal`, `Ampcal`, `AmpPhasecal`**: calibration steps.
- **Acceptance policies**: `PSNRImproves`, `EnoughAntennas`, `FluxNotLost`, `BeamStable`,
  `FlaggedFractionBelow`, `Check(name, predicate)` for your own, combined with `AllOf`/`AnyOf`.
- **Observers**: `LoggingObserver` (the default) and `GainPlotObserver`, passed to `run()`.

### Configuration (`wenuray.config`)

Frozen objects with CASA's parameter names: `Selection`, `SolveOptions`, `ApplyOptions` (with its
`ApplyTarget`s),
`ImageGeometry` and `Weighting` (natural by default).

### Imaging (`wenuray.imaging`)

`Tclean`, `Pyralysis`, the `Imager` base class and `CallableImager`; see [Imagers](imagers.md).

The full reference is generated from the docstrings: see the API pages.
