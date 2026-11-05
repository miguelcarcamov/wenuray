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

### Solints from the data

`solint="auto"` lets the data choose (#66). When the round starts, Wenuray predicts the solution
S/N against the model the last accepted round left (see below), and takes the shortest solint
whose S/N reaches twice `minsnr`, so that most solutions stay above `minsnr` despite their own
scatter. If no solint that short fits in a scan (or in the observation, with `combine="scan"`),
it takes `"inf"` when that still reaches `minsnr`; if not even that does, the round doesn't run,
and the log says why.

gaincal cuts each scan into intervals from its start and leaves the rest as a last, shorter one:
a 420 s solint on a 120-integration scan of 60 s gives 17 solutions of 7 integrations and one of

1. A remainder of r integrations, in a solint sized for n, has an S/N of 2 · `minsnr` · √(r / n),
below `minsnr` when r < n / 4, and calflag flags its data. So of the lengths from n to 2n
integrations, `"auto"` takes the shortest that leaves the fewest integrations in such remainders
across the scans: 8 integrations here, 15 solutions of 8. The log says when it lengthened the
solint, and what share of the data still falls in a remainder too short, e.g. a scan of a single
integration.

`Repeat(Phasecal("auto"))` runs the step again, each round at a shorter solint than the last, as a
better model raises the S/N. It ends when the gains are consistent with their noise, as shorter
solints would only fit noise, when the data support no solint at least `min_shortening` (25% by
default) shorter than the last round's, at one integration, or when a round is rejected. The
supported solint shortens as the square of the S/N, so without that margin a model a few percent
brighter would buy a whole round for a solint a few percent shorter; `min_shortening=0` asks only
for a shorter one. Then the run goes on with the next entry: a rejection inside a repeat
ends the repeat, not the run.

```python
steps = [Repeat(Phasecal("auto")), AmpPhasecal("auto")]
```

On simulated data with 1 Jy over 1 Jy of noise per visibility, this schedule ran:

```text
Round 1 Phasecal('auto' -> '300s'): accepted
  gains      phase scatter 2.0 x noise: a correction
  image      PSNR 54.5 (+127.7% on round 0), rms 16.1 mJy/beam (1.30 x thermal), model flux 1.18 Jy
Round 2 Phasecal('auto' -> '180s'): accepted
  gains      phase scatter 1.1 x noise: consistent with noise on 280 solutions
  image      PSNR 77.6 (+42.3% on round 1), rms 12.1 mJy/beam (0.99 x thermal), model flux 1.088 Jy
Repeat of Phasecal('auto') ends: its last round's gains are consistent with their noise
Round 3 AmpPhasecal('auto' -> '180s'): accepted
  gains      phase scatter 0.5 x noise: consistent with noise on 278 solutions; amplitude scatter 0.9 x noise: consistent with noise on 278 solutions
```

The better model in round 2 supports 180 s, where round 1's supported 300 s. The repeat stops when
the residual reaches the thermal noise, and the amplitude step's corrections are at their noise:
on these data it wasn't needed. Each round records the solint it resolved to, e.g.
`Phasecal('auto' -> '180s')`, so the same recipe on the same data takes the same decisions. In a
recipe:

```toml
[[steps]]
type = "repeat"
step = { type = "phase", solint = "auto" }

[[steps]]
type = "ampphase"
solint = "auto"
```

An amplitude step after a `Repeat` of phases needs at least one of its rounds accepted; if every
one was rejected, the run stops before it, as no phase solution is there to build on.

## Each round

- **Each solve is checked first.** Before `gaincal`, Wenuray predicts a solution's S/N for the
  step's solint and combine from the data: the median model amplitude over the noise of one
  visibility, times √(N_ant − 1) · √(n_int · n_chan · n_pol). For a step that solves amplitudes,
  it is that over √2, as gaincal's own S/N, the one `minsnr` is compared with, is: on the same
  data, gaincal reports 75 for `calmode="p"` and 53 for `"ap"`. It warns when the prediction is
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
  the data: the noise of one visibility, averaged down by the data weights (σ · √Σw² / Σw). The
  noise of one visibility comes from differences that cancel the source. Consecutive integrations
  of a baseline cancel it while the gains hold over an integration; the parallel hands of a
  visibility (XX − YY or RR − LL) cancel an unpolarized source and the gains both hands share, as
  the atmosphere's are, however fast they change. Each estimate can only be raised by source that
  doesn't cancel: gains that change within an integration raise the first, a polarized source or
  different gains in the two hands the second. So Wenuray takes the lower one, the hands' only when
  it is lower by more than 3 standard errors, and the log names it and gives both
  ([#153](https://github.com/miguelcarcamov/wenuray/issues/153)). Adjacent channels aren't used:
  smoothed spectra (e.g. Hanning) correlate their noise, which would read it too low. It is exact for natural weighting and a floor for
  briggs or uniform. Each round logs and records its residual noise against it (`thermal_noise`,
  and `rms_in_mask` inside the source mask). A residual inside the mask below 0.9× thermal is
  warned about: the model and the gains are fitting noise there. Image noise is always the MAD
  standard deviation, since it is correlated by the beam and isn't Gaussian.
- **The weights.** From the same differences, Wenuray compares each baseline and spectral window's
  measured noise with the noise its `SIGMA` states, and warns when the weights misstate it by more
  than 10% (the median over baselines), or when a baseline stands out. The initial round records
  the median as `noise_over_sigma`. `statwt` or Pyralysis's `EmpiricalWeightEstimator` can
  re-estimate the weights (#124).
- **What the gains say.** Each solved round records diagnostics of its gain table (`gains`): its
  phase (and amplitude) corrections against the noise of the solutions, from the table's SNR
  column. Each table is the increment over the accepted stack, so a ratio falling towards 1 means
  the run is converging: the round corrected little beyond noise. It also names the antennas whose
  phases scatter more than their distance from the array centre predicts (atmospheric phase grows
  with baseline length), and, for gaintype `G`, compares X with Y: an X−Y phase ratio near 1 says
  `T` would do. `RoundMetrics.gains` holds them for acceptance policies of your own.
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

`wenuray run` logs each phase as it starts and how long it took, and Pyralysis's iterations as it
works, so a long round shows what it is doing:

```text
23:29:52 INFO    Round 1/2 Phasecal('inf'): imaging with Pyralysis
23:29:58 INFO    Pyralysis LBFGS-B: iteration 1, objective 0.41353077
23:30:04 INFO    Pyralysis LBFGS-B: iteration 10, objective 0.059638306
23:30:21 INFO    Round 1/2 Phasecal('inf'): imaging with Pyralysis: done in 29 s
```

Each round then ends with a block that says what was solved and how well, what the gains corrected,
what the image shows and why the round was kept or not (#179). From a run on simulated data:

```text
12:25:26 INFO    Round 2/3 Phasecal('120s'): accepted
  solutions  480 solutions over 60 intervals, all with S/N above 3
  gains      phase scatter 1.9 x noise: a correction
  image      PSNR 422.9 (+27.8% on round 1), rms 1.7 mJy/beam (2.78 x thermal), model flux 2.022 Jy
  decision   accepted: PSNR 331.0 -> 422.9; every interval >= 3 antennas
12:26:26 INFO    Round 3/3 AmpPhasecal('inf'): accepted
  solutions  8 solutions over 1 interval, all with S/N above 3
  gains      phase scatter 4.3 x noise: a correction; amplitude scatter 1.1 x noise: consistent with noise on 7 solutions
  image      PSNR 472.5 (+11.8% on round 2), rms 1.57 mJy/beam (2.57 x thermal), model flux 1.906 Jy
  decision   accepted: PSNR 422.9 -> 472.5; every interval >= 4 antennas; flux -5.7% (limit -10%)
```

- **solutions**: the antenna solutions gaincal found, one per antenna, interval and spectral
  window. A solution fails when its S/N is below `minsnr` or it has no data, and `applycal`
  (calflag) then flags the data it would have calibrated; the block names the antennas that
  failed most. An interval with fewer antennas than the step needs, 3 for phases (closure phases)
  and 4 for amplitudes (closure amplitudes), leaves its gains undetermined.

  A table solved per spectral window can lack whole windows. CASA's applycal then prints
  `The following MS spws have no corresponding cal spws in round4_ap.cal: 0 1 2 ...`, and the
  block says which case it is. Windows without unflagged data, e.g. those a split keeps by their
  original ids (ALMA's data in 13, 15, 17, 19 of 0-20), lose nothing: "spws 0-12, 14, 16, 18, 20
  hold no unflagged data: CASA's "no corresponding cal spws" about them is harmless". A selected
  window with unflagged data but no solutions is left uncalibrated, not flagged, by applycal,
  and enters the image so; Wenuray warns about it.
- **gains**: the scatter of the solutions over their own noise, from their S/N. About 1 means the
  round corrected nothing beyond noise; "a correction" means it stands more than 3 standard
  errors above 1, on as many solutions as the table has: on few solutions, e.g. 7 in one
  interval of 8 antennas, even 2 x noise can be chance. Above, the amplitudes of round 3 are
  consistent with noise: that step added nothing the phases hadn't.
- **image**: the peak S/N and its change on the last accepted round, the residual noise against
  the thermal noise from the data, and the model's flux.
- **decision**: each acceptance rule and its result: the PSNR must rise, every interval must keep the
  antennas the step needs, and an amplitude step may not take more of the model's flux than its
  limit. A round rejected for lack of S/N in its
  solutions adds a hint, e.g. a longer `solint` or `combine = "spw"`.

At the end, a table of every round and the final image:

```text
12:26:26 INFO    Self-calibration finished in 3 min 35 s: 3 of 3 steps accepted
  Round  Step                PSNR   rms/thermal  Model flux  Gains/noise  Decision
  0      initial image       301.9  3.58         2.31 Jy     -            accepted
  1      Phasecal('inf')     331.0  3.39         2.217 Jy    3.2          accepted
  2      Phasecal('120s')    422.9  2.78         2.022 Jy    1.9          accepted
  3      AmpPhasecal('inf')  472.5  2.57         1.906 Jy    4.3 / 1.1    accepted
Final image: run/round3.image (round 3), PSNR 472.5, rms 2.57 x thermal
```

`selfcal.toml` records the same for each round (`solutions`, `gains`).

`--quiet` keeps the warnings only; `--debug` adds the details, e.g. all of Pyralysis's output.

CASA writes to the terminal on its own, alongside Wenuray's log. `wenuray run` keeps out two
notices every self-calibration gets, which say nothing about the data:
`Forcing use of OLD VisibilityIterator`, which every calibration task gives, and
`Deleting version: wenuray_selfcal_round...`, the flag versions the run discards when it ends.
`--debug` shows them. Everything else CASA says still shows, warnings and errors among them; a
few that look alarming and aren't:

- `The contents of measurespath do not appear to be casarundata`: CASA's measures data aren't
  where its configuration says, often a `HOME` without `~/.casa/config.py`. CASA still works if
  it finds them elsewhere; see [Installation](installation.md).
- `The following MS spws have no corresponding cal spws in ...`: see the solutions line above.

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
- **`Phasecal`, `Ampcal`, `AmpPhasecal`**: calibration steps, with a solint or `"auto"`.
- **`Repeat(step, min_shortening=0.25)`**: a step run again at `"auto"` solints, each at least
  `min_shortening` shorter than the last, until its gains are at the noise.
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
