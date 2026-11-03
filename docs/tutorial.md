# Tutorial: self-calibrating a simulated ALMA observation

This tutorial makes an observation whose answer we know, breaks it the way the atmosphere does, and
lets Wenuray repair it. You'll see what each self-calibration round does to the image, and check
that the gains Wenuray solves for are the gains that were put in.

Everything runs on a laptop in about ten minutes, in Wenuray's Docker image, which holds Wenuray
and [Pyralysis](https://pyralysis.readthedocs.io/) side by side (see
[Installation](installation.md#with-docker)). The files are in
[`examples/simulated/`](https://github.com/miguelcarcamov/wenuray/tree/development/examples/simulated):

```bash
git clone https://github.com/miguelcarcamov/wenuray && cd wenuray
docker run --rm -it -v "$PWD":/work -w /work ghcr.io/miguelcarcamov/wenuray:latest bash
```

## 1. Simulate the observation

[`simulate.py`](https://github.com/miguelcarcamov/wenuray/blob/development/examples/simulated/simulate.py)
uses Pyralysis's simulation framework. The sky is a protoplanetary disk: a ring, 0.45″ in radius
with a 0.3″ hole, inclined by 30°, and a compact source at its centre, 0.18 Jy in all. ALMA observes
it in configuration C-10.6 at 345 GHz (Band 7) for two hours, in 30 s integrations.

```{literalinclude} ../examples/simulated/simulate.py
:language: python
:start-at: "# ALMA in configuration"
:end-before: "def main"
```

The sky and the simulation itself come from
[`pyralysis_simulation.py`](https://github.com/miguelcarcamov/wenuray/blob/development/examples/pyralysis_simulation.py),
which other examples can share:

```{literalinclude} ../examples/pyralysis_simulation.py
:language: python
:pyobject: observe
```

Then the observation is corrupted with Pyralysis's noise injectors, as its documentation
recommends: the multiplicative errors first, then the additive ones, as in a receiver chain.

- **Antenna gain errors**: 40° of phase and 5% of amplitude per antenna, drawn anew for every
  integration, like fast atmospheric phase fluctuations.
- **Thermal noise** from the radiometer equation for a 150 K system. Pyralysis also writes the
  `SIGMA` and `WEIGHT` columns to match it.

```{literalinclude} ../examples/pyralysis_simulation.py
:language: python
:pyobject: gain_errors
```

```{literalinclude} ../examples/pyralysis_simulation.py
:language: python
:pyobject: thermal_noise
```

The script writes three things: `tutorial.ms`, the corrupted observation; `tutorial.ms.ideal`, the
same observation and the *same* thermal noise without gain errors, which is what a perfect
calibration would give; and `tutorial.ms.gains.npz`, the injected gains:

```bash
python examples/simulated/simulate.py tutorial.ms
```

## 2. Describe the self-calibration

The procedure is a short recipe,
[`recipe.toml`](https://github.com/miguelcarcamov/wenuray/blob/development/examples/simulated/recipe.toml):

```{literalinclude} ../examples/simulated/recipe.toml
:language: toml
```

A few choices, and why:

- **The reference antenna** is the one nearest the array's centre: it sees every other antenna on
  short, well-measured baselines.
- **`cell = "auto"`** takes the cell from the data: the Nyquist cell of the longest baseline,
  divided by 2.5, so about five pixels cover the beam. Here that's 0.0153″. 256 pixels hold the disk
  and some sky around it.
- **Three rounds.** A phase solution over the whole observation (`inf`) first: it can only remove
  the part of the errors that's constant, and here there's little. Then phases every integration
  (`int`), as fast as the errors change. Last, amplitudes and phases every integration, once the
  model is good.

The same run in Python is
[`selfcal.py`](https://github.com/miguelcarcamov/wenuray/blob/development/examples/simulated/selfcal.py);
a test checks that the two build the same run.

## 3. Run it

```bash
wenuray run examples/simulated/recipe.toml --ms tutorial.ms --output tutorial_selfcal
```

Each round images the data, solves the gains against that image's model, applies them and keeps
the round only if the image improves. The log tells the story:

| Round | Solution interval | Peak S/N | Image noise | Model flux | Gain corrections / their noise |
| --- | --- | --- | --- | --- | --- |
| initial | | 310 | 59.2 µJy/beam | 0.106 Jy | |
| phase | `inf` | 380 | 48.4 µJy/beam | 0.107 Jy | 6.3 |
| phase | `int` | 1215 | 24.7 µJy/beam | 0.179 Jy | 14.5 |
| amplitude and phase | `int` | 1241 | 24.3 µJy/beam | 0.179 Jy | phase 0.3, amplitude 1.5 |
| *ideal: no gain errors* | | *1245* | *24.1 µJy/beam* | *0.179 Jy* | |

Before calibration, the phase errors scatter the visibilities so much that they partly cancel out:
the image holds only 0.106 of the 0.18 Jy. The solution over the whole observation helps a little.
The solutions every integration recover the flux and bring the noise to within 1% of the ideal
image's. The last round's corrections are at the level of their own noise (0.3 and 1.5): there was
nothing left to correct, which is what convergence looks like.

## 4. Look at the results

Make the figures with
[`make_figures.py`](https://github.com/miguelcarcamov/wenuray/blob/development/examples/simulated/make_figures.py):

```bash
python examples/simulated/make_figures.py tutorial.ms tutorial_selfcal figures
```

```{figure} _static/tutorial/images.png
:alt: The disk before and after self-calibration, and with no gain errors, with their residuals

Top: the image before self-calibration, after it, and of the data without gain errors. Bottom:
their residuals. Before, the residual is dominated by the source and by stripes the gain errors
spread across the image; after, it matches the ideal one. The faint disk left in both is extended
flux the deconvolution hasn't cleaned yet, not a calibration error.
```

```{figure} _static/tutorial/convergence.png
:alt: Peak S/N, image noise and gain corrections over the rounds

Each round's peak S/N, its noise against the ideal image's (the floor any calibration can reach),
and its gain corrections against their own noise. A ratio near 1 in the last panel means a round
found nothing more to correct.
```

```{figure} _static/tutorial/gains.png
:alt: Injected and self-calibrated antenna phases

The phases Wenuray found against those injected, relative to the reference antenna. Left: one
antenna over the two hours, integration by integration. Right: every antenna at every integration.
They agree to 3.0° rms. The errors were 40° per antenna, so 56° rms between an antenna and the
reference.
```

## What the warnings mean

Wenuray warns twice during this run, and here both are false alarms:

- *The data's noise is 1.51 times the noise their weights state.*
- *The residual inside the source mask is below the thermal noise* (rounds 2 and 3).

Wenuray measures the noise from the difference between consecutive integrations, which cancels the
source as long as the gains change slowly. In this simulation they change completely every
integration, so the differences keep part of the source, and the noise measured comes out about
twice too high. The weights are right (`SIGMA` matches the noise of the ideal data exactly), and the
residual isn't below the real thermal noise. Real gains change more slowly, but fast phase
fluctuations do the same; making this estimate robust to them is
[#153](https://github.com/miguelcarcamov/wenuray/issues/153).

## What Pyralysis's simulation needs for CASA

A measurement set from Pyralysis 3.3's simulation doesn't open in CASA as it comes. `simulate.py`'s
`write()` patches three things, filed upstream:

- The pipeline's export writes only the main table, so every subtable is empty
  ([pyralysis#405](https://gitlab.com/clirai/pyralysis/-/issues/405)). Write with
  `DaskMS(...).write(dataset, tables="ALL")`.
- The spectral windows record REST frequencies, which tclean can't convert
  ([pyralysis#406](https://gitlab.com/clirai/pyralysis/-/issues/406)). Observatories record TOPO.
- The `FEED` table is empty, so tclean finds no receptors
  ([pyralysis#407](https://gitlab.com/clirai/pyralysis/-/issues/407)). Each antenna gets a row with
  ALMA's linear feeds, X and Y.

Two more don't stop CASA here: `STATE_ID` and `PROCESSOR_ID` point to empty tables
([pyralysis#408](https://gitlab.com/clirai/pyralysis/-/issues/408)), and the antenna gains are drawn
independently for every integration and channel
([pyralysis#409](https://gitlab.com/clirai/pyralysis/-/issues/409)). The latter is why this
tutorial uses one channel per spectral window, and why its gains change faster than real ones do.

## Try it yourself

- **Slower or stronger errors**: change `phase_sigma` and `amp_sigma` in `gain_errors()`.
- **A longer ladder**: once gains can be correlated in time (pyralysis#409), add `Phasecal("120s")`
  between `inf` and `int`, and watch the corrections fall round by round.
- **Pyralysis as the imager**: set `type = "pyralysis"` in `[imager]`, with a mask (see
  [Imagers](imagers.md#pyralysis)).
