# Gallery: self-calibration across arrays and frequencies

Self-calibration has to work whatever the array: a few dishes or a few dozen, metres or tens of
kilometres apart, at hundreds of megahertz or hundreds of gigahertz. The atmosphere changes with
each: the ionosphere disturbs low frequencies, the troposphere high ones, and both grow with
baseline length. This gallery puts Wenuray through nine such cases with one and the same
self-calibration, imaged with two imagers, tclean and Pyralysis, and shows each before and after.

```{figure} _static/gallery/gallery.webp
:alt: Nine simulated observations before and after self-calibration with tclean

Each scenario before (left) and after (right) self-calibration with tclean, on the same colour
scale. The peak S/N of each image is in its corner.
```

```{figure} _static/gallery/gallery-pyralysis.webp
:alt: The same nine observations before and after self-calibration with Pyralysis

The same observations self-calibrated with Pyralysis as the imager.
```

## The scenarios

Two arrays, the VLA and ALMA, each at a low and a high frequency, in a compact and an extended
configuration, and MeerKAT below 1 GHz. They're described in
[`examples/gallery/scenarios.toml`](https://github.com/miguelcarcamov/wenuray/blob/development/examples/gallery/scenarios.toml),
with the reasoning behind each value.

| Scenario | Target | Phase errors | Peak S/N with tclean | Peak S/N with Pyralysis |
| --- | --- | --- | --- | --- |
| MeerKAT · 816 MHz · up to 8 km | a giant radio galaxy and field sources | 33° → 6° | 106 → **719** (722) | 33 → **240** (273) |
| VLA D · 1.5 GHz · up to 1 km | a nearby spiral galaxy | 19° → 4° | 127 → **443** (435) | 74 → **466** (437) |
| VLA A · 1.5 GHz · up to 36 km | a powerful radio galaxy, Cygnus A-like | 53° → 8° | 53 → **688** (659) | 32 → **653** (667) |
| VLA D · 45 GHz · up to 1 km | compact HII regions in a star-forming region | 40° → 5° | 119 → **826** (826) | 73 → **860** (866) |
| VLA A · 45 GHz · up to 36 km | a young star's disk at 7 mm | 87° → 21° | 37 → **629** (674) | 18 → **604** (737) |
| ALMA C-3 · 100 GHz · up to 500 m | filaments meeting in a hub of dense cores | 21° → 5° | 161 → **566** (555) | 105 → **599** (582) |
| ALMA C-8 · 100 GHz · up to 8.5 km | a protoplanetary disk with rings and gaps | 58° → 12° | 100 → **997** (1007) | 64 → **1514** (1555) |
| ALMA C-3 · 345 GHz · up to 500 m | a gravitationally lensed dusty galaxy | 41° → 7° | 85 → **711** (701) | 66 → **777** (771) |
| ALMA C-8 · 345 GHz · up to 8.5 km | a protoplanetary disk with rings, gaps and spiral arms | 75° → 23° | 57 → **937** (951) | 28 → **945** (1268) |

Phase errors: the injected antenna phases relative to the reference antenna, rms, and those left
after self-calibration (with tclean; Pyralysis's are within a degree). Peak S/N: before → **after**
(without gain errors), each imager against its own ideal image.

**Each array observes a target typical for its band**: a giant radio galaxy among background
sources for MeerKAT; a spiral galaxy and a Cygnus A-like radio galaxy at 1.5 GHz; HII regions and a
young star's disk at 7 mm; and, for ALMA, filaments meeting in a hub of dense cores, a lensed dusty
galaxy, and protoplanetary disks with rings, gaps and spiral arms. They're built from Pyralysis's
analytic components, so their visibilities are exact
([`skies.py`](https://github.com/miguelcarcamov/wenuray/blob/development/examples/gallery/skies.py)):
points, elliptical Gaussians, disks and rings, laid along filaments and scattered in clumps (seeded,
so every run builds the same sky). Each spans tens of beams where the array allows it. A compact
configuration can't image emission larger than about 0.6 λ over its shortest baseline, so its
targets are kept within that scale. Fluxes are in units of the image noise, so every scenario has a
comparable S/N. Each target has a compact bright part, as real ones do, which the longest baselines
solve on.

**The gain errors follow the physics.** The ionosphere's phase grows towards low frequency and with
baseline length; the troposphere's grows with frequency and baseline length; amplitude errors grow
with frequency (pointing and opacity). As the atmosphere's, they drift smoothly, over a coherence
time of one to fifteen minutes, shorter at high frequency and on long baselines. Pyralysis's
`AntennaGainInjector` draws gains independently for every integration (pyralysis#409), and errors
that change that fast mostly average out: they blur and dim the image but hardly show. So the gallery
uses a subclass of it, `DriftingGainInjector` in
[`pyralysis_simulation.py`](https://github.com/miguelcarcamov/wenuray/blob/development/examples/pyralysis_simulation.py),
whose gains are smooth random processes in time, on top of thermal noise from the radiometer
equation.

**The self-calibration is the same everywhere**: phase solutions over the whole observation, then
over two minutes, then every 30 s integration, and last amplitude-and-phase solutions over five
minutes, with the antenna nearest the array's centre as reference. Both imagers use Briggs weighting
(robust 0.5) on a 512-pixel grid with an `"auto"` cell, about five pixels per beam:

- **tclean**: multiscale CLEAN (point sources, one and three beams) with an automatic mask.
- **Pyralysis**: χ² with a small total squared variation, optimised with L-BFGS-B, inside a mask.
  The mask is tclean's last one grown by a beam, as one would draw CLEAN boxes after a first look at
  the field: a model free in every pixel could absorb the gain errors self-calibration should solve.

## The results

```{figure} _static/gallery/summary.png
:alt: Peak S/N and phase errors before and after self-calibration, per scenario

Left: each image's peak S/N before self-calibration, after it, and without gain errors (the ideal
a perfect calibration reaches), with tclean (circles) and Pyralysis (squares). Right: the antenna
phase errors injected, and those left after self-calibration.
```

Every scenario reaches its ideal image with tclean, to within its noise (within 7% of the ideal's
peak S/N). With Pyralysis, two fall short: the young star's disk at 45 GHz and the spiral disk at
345 GHz end with 1.2 and 1.4 times the ideal noise, the latter's reconstruction not having converged
in its 100 iterations. At 45 GHz and 345 GHz on the longest baselines, about 20° of phase is left:
mostly the noise of solutions every 30 s at those frequencies, not error left uncorrected, as the
images show.

```{figure} _static/gallery/convergence.png
:alt: Image noise against the ideal image's, round by round, per scenario

The image noise over the ideal image's, round by round, with each imager. The solution over the
whole observation removes only the errors' mean; the solutions over two minutes follow the slow
drift, and those every integration the rest, which brings every scenario to its thermal floor.
```

## Each scenario

Each figure shows the image before self-calibration, after it, and without gain errors, with their
residuals below on one scale (±6σ of the ideal residual), first with tclean, then with Pyralysis.

```{figure} _static/gallery/meerkat-uhf.webp
:alt: MeerKAT at 816 MHz, a giant radio galaxy and field sources, with tclean

MeerKAT · 816 MHz · up to 8 km: a giant radio galaxy and field sources, with tclean.
```

```{figure} _static/gallery/meerkat-uhf-pyralysis.webp
:alt: MeerKAT at 816 MHz, a giant radio galaxy and field sources, with Pyralysis

MeerKAT · 816 MHz · up to 8 km: a giant radio galaxy and field sources, with Pyralysis.
```

```{figure} _static/gallery/vla-l-d.webp
:alt: VLA D at 1.5 GHz, a nearby spiral galaxy, with tclean

VLA D · 1.5 GHz · up to 1 km: a nearby spiral galaxy, with tclean.
```

```{figure} _static/gallery/vla-l-d-pyralysis.webp
:alt: VLA D at 1.5 GHz, a nearby spiral galaxy, with Pyralysis

VLA D · 1.5 GHz · up to 1 km: a nearby spiral galaxy, with Pyralysis.
```

```{figure} _static/gallery/vla-l-a.webp
:alt: VLA A at 1.5 GHz, a powerful radio galaxy, Cygnus A-like, with tclean

VLA A · 1.5 GHz · up to 36 km: a powerful radio galaxy, Cygnus A-like, with tclean.
```

```{figure} _static/gallery/vla-l-a-pyralysis.webp
:alt: VLA A at 1.5 GHz, a powerful radio galaxy, Cygnus A-like, with Pyralysis

VLA A · 1.5 GHz · up to 36 km: a powerful radio galaxy, Cygnus A-like, with Pyralysis.
```

```{figure} _static/gallery/vla-q-d.webp
:alt: VLA D at 45 GHz, compact HII regions in a star-forming region, with tclean

VLA D · 45 GHz · up to 1 km: compact HII regions in a star-forming region, with tclean.
```

```{figure} _static/gallery/vla-q-d-pyralysis.webp
:alt: VLA D at 45 GHz, compact HII regions in a star-forming region, with Pyralysis

VLA D · 45 GHz · up to 1 km: compact HII regions in a star-forming region, with Pyralysis.
```

```{figure} _static/gallery/vla-q-a.webp
:alt: VLA A at 45 GHz, a young star's disk at 7 mm, with tclean

VLA A · 45 GHz · up to 36 km: a young star's disk at 7 mm, with tclean.
```

```{figure} _static/gallery/vla-q-a-pyralysis.webp
:alt: VLA A at 45 GHz, a young star's disk at 7 mm, with Pyralysis

VLA A · 45 GHz · up to 36 km: a young star's disk at 7 mm, with Pyralysis.
```

```{figure} _static/gallery/alma-b3-c3.webp
:alt: ALMA C-3 at 100 GHz, filaments meeting in a hub of dense cores, with tclean

ALMA C-3 · 100 GHz · up to 500 m: filaments meeting in a hub of dense cores, with tclean.
```

```{figure} _static/gallery/alma-b3-c3-pyralysis.webp
:alt: ALMA C-3 at 100 GHz, filaments meeting in a hub of dense cores, with Pyralysis

ALMA C-3 · 100 GHz · up to 500 m: filaments meeting in a hub of dense cores, with Pyralysis.
```

```{figure} _static/gallery/alma-b3-c8.webp
:alt: ALMA C-8 at 100 GHz, a protoplanetary disk with rings and gaps, with tclean

ALMA C-8 · 100 GHz · up to 8.5 km: a protoplanetary disk with rings and gaps, with tclean.
```

```{figure} _static/gallery/alma-b3-c8-pyralysis.webp
:alt: ALMA C-8 at 100 GHz, a protoplanetary disk with rings and gaps, with Pyralysis

ALMA C-8 · 100 GHz · up to 8.5 km: a protoplanetary disk with rings and gaps, with Pyralysis.
```

```{figure} _static/gallery/alma-b7-c3.webp
:alt: ALMA C-3 at 345 GHz, a gravitationally lensed dusty galaxy, with tclean

ALMA C-3 · 345 GHz · up to 500 m: a gravitationally lensed dusty galaxy, with tclean.
```

```{figure} _static/gallery/alma-b7-c3-pyralysis.webp
:alt: ALMA C-3 at 345 GHz, a gravitationally lensed dusty galaxy, with Pyralysis

ALMA C-3 · 345 GHz · up to 500 m: a gravitationally lensed dusty galaxy, with Pyralysis.
```

```{figure} _static/gallery/alma-b7-c8.webp
:alt: ALMA C-8 at 345 GHz, a protoplanetary disk with rings, gaps and spiral arms, with tclean

ALMA C-8 · 345 GHz · up to 8.5 km: a protoplanetary disk with rings, gaps and spiral arms, with tclean.
```

```{figure} _static/gallery/alma-b7-c8-pyralysis.webp
:alt: ALMA C-8 at 345 GHz, a protoplanetary disk with rings, gaps and spiral arms, with Pyralysis

ALMA C-8 · 345 GHz · up to 8.5 km: a protoplanetary disk with rings, gaps and spiral arms, with Pyralysis.
```

## Make it yourself

In Wenuray's Docker image, which holds Pyralysis in its own environment
([Installation](installation.md#with-docker)):

```bash
python examples/gallery/make_gallery.py gallery docs/_static/gallery
```

In the `:gpu` image, run it with `--pyralysis-backend cupy` (and `docker run --gpus all`) for
Pyralysis on the GPU.

It simulates each scenario with Pyralysis
([`simulate_scenario.py`](https://github.com/miguelcarcamov/wenuray/blob/development/examples/gallery/simulate_scenario.py),
through
[`pyralysis_simulation.py`](https://github.com/miguelcarcamov/wenuray/blob/development/examples/pyralysis_simulation.py)),
self-calibrates it, and draws the figures
([`figures.py`](https://github.com/miguelcarcamov/wenuray/blob/development/examples/figures.py)).
Every scenario has its own seed and a fixed date, so each run gives the same figures. It resumes
where it stopped. `--only vla-q-a alma-b7-c8` runs some scenarios only, and `--imagers tclean` one
imager only.

Wenuray warns in some of these runs that the data's noise doesn't match their weights. It's a false
alarm: Wenuray measures the noise from the difference between consecutive integrations, which keeps
part of a bright source when the gains change within an integration's time, as they do here on the
longest baselines; [#153](https://github.com/miguelcarcamov/wenuray/issues/153) makes that estimate
robust. The simulated measurement sets also need the patches for Pyralysis 3.3 described in the
[tutorial](tutorial.md#what-pyralysiss-simulation-needs-for-casa).
