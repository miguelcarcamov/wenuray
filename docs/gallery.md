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
| MeerKAT · 816 MHz · up to 8 km | a giant radio galaxy and field sources | 33° → 6.6° / 5.6° | 83 → **1786** (1851) | 122 → **1704** (1806) |
| VLA D · 1.5 GHz · up to 1 km | a nearby spiral galaxy | 19° → 3.7° / 4.1° | 87 → **486** (489) | 87 → **611** (584) |
| VLA A · 1.5 GHz · up to 36 km | a powerful radio galaxy, Cygnus A-like | 53° → 9.1° / 8.2° | 40 → **815** (815) | 39 → **861** (909) |
| VLA D · 45 GHz · up to 1 km | compact HII regions in a star-forming region | 40° → 5.2° / 5.2° | 90 → **852** (846) | 98 → **1091** (1103) |
| VLA A · 45 GHz · up to 36 km | a young star's disk at 7 mm | 87° → 20° / 23° | 21 → **625** (766) | 23 → **638** (1006) |
| ALMA C-3 · 100 GHz · up to 500 m | filaments meeting in a hub of dense cores | 21° → 5.1° / 5.3° | 121 → **732** (716) | 125 → **761** (746) |
| ALMA C-8 · 100 GHz · up to 8.5 km | a protoplanetary disk with rings and gaps | 58° → 12° / 12° | 82 → **1501** (1574) | 84 → **2319** (2257) |
| ALMA C-3 · 345 GHz · up to 500 m | a gravitationally lensed dusty galaxy | 41° → 7.3° / 7.2° | 84 → **1068** (1036) | 81 → **1002** (1011) |
| ALMA C-8 · 345 GHz · up to 8.5 km | a protoplanetary disk with rings, gaps and spiral arms | 75° → 23° / 23° | 50 → **1284** (1532) | 44 → **2120** (2277) |

Phase errors: the injected antenna phases relative to the reference antenna, rms, and those left
after self-calibration with tclean / Pyralysis. Peak S/N: before → **after** (without gain
errors), each imager against its own ideal image. Compare each imager with its own ideal, not with
the other's: their noise differs (below), and so, but for MeerKAT, do their beams.

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
minutes, with the antenna nearest the array's centre as reference. Both imagers use natural weighting,
the most sensitive, so their images compare like for like, on a 512-pixel grid with an `"auto"`
cell, about five pixels per beam:

- **tclean**: multiscale CLEAN (point sources, one and three beams) with an automatic mask.
- **Pyralysis**: χ² with a small total squared variation, optimised with L-BFGS-B, inside a mask.

```{important}
**Pyralysis's mask comes from tclean.** It is the mask of tclean's last round in the same
scenario, grown by a beam, and Pyralysis uses it from its first round on. So Pyralysis starts
knowing where the emission is, which tclean only found by the end of its own self-calibration: its
first rounds are helped, and the two imagers don't start from the same knowledge.

Pyralysis needs a mask. Without one, its model, free and positive in every pixel, absorbs the gain
errors and the noise. On the 45 GHz VLA D scenario, its image before self-calibration already had a
peak S/N of 445, against tclean's 90, and 11° of phase error was left after self-calibration,
against 5° with this mask. A mask derived from the data each round, e.g. from the dirty image above
its noise and sidelobes, grown by a beam, would make Pyralysis independent of tclean; until then,
the gallery uses tclean's.
```

```{note}
**Pyralysis fits part of the noise inside its mask.** Its model is free in every masked pixel, so
it takes up the noise there too: its residual inside the mask is 0.1 to 0.7 times the thermal
noise, and on the extended arrays, whose sidelobes carry that fit beyond the mask, 0.6 to 0.8
times it outside. CLEAN's residual is 1.3 to 3 times the thermal noise inside its mask, where it
leaves deconvolution errors, and the thermal noise outside. So Pyralysis's peak S/N is higher than
tclean's on the same data, and its ideal image's too: the noise it reports is lower than the
data's. The gallery was made before Wenuray measured Pyralysis's rounds outside and inside its mask
([#167](https://github.com/miguelcarcamov/wenuray/issues/167), 1.2.1): its Pyralysis noise is
the whole residual's, with sigma clipping.

| Residual noise over the thermal noise | tclean, in / out of the mask | Pyralysis, in / out of the mask |
| --- | --- | --- |
| MeerKAT · 816 MHz | 1.65 / 1.06 | 0.74 / 1.07 |
| VLA D · 1.5 GHz | 1.36 / 1.02 | 0.20 / 0.80 |
| VLA A · 1.5 GHz | 1.30 / 1.03 | 0.33 / 0.98 |
| VLA D · 45 GHz | 1.46 / 1.01 | 0.10 / 0.80 |
| VLA A · 45 GHz | 2.47 / 1.27 | 0.21 / 1.23 |
| ALMA C-3 · 100 GHz | 1.37 / 0.99 | 0.49 / 0.96 |
| ALMA C-8 · 100 GHz | 2.75 / 1.05 | 0.13 / 0.61 |
| ALMA C-3 · 345 GHz | 1.88 / 1.03 | 0.51 / 1.01 |
| ALMA C-8 · 345 GHz | 3.20 / 1.23 | 0.11 / 0.66 |

The noise is the MAD standard deviation of each self-calibrated image's residual, in the mask
Pyralysis used (tclean's, grown by a beam) and outside it, over that of tclean's ideal residual
outside the mask, the thermal noise.
```

### Where each imager grids

Neither imager computes the Fourier transform of its model visibility by visibility, as a direct
sum would; both go through a padded grid and an FFT, and how accurately they interpolate between the
grid and the visibilities decides how faithful their models and residuals are.

| Step | tclean | Pyralysis |
| --- | --- | --- |
| Fitting the model | Major cycles: the residual visibilities gridded into a dirty image, CLEANed in the image | χ² **on the visibilities** (row by row), never gridded |
| Predicting the model's visibilities (and, for Pyralysis, the χ² gradient) | FFT of the model on a grid padded by 1.2, degridded with a prolate spheroidal kernel 7 cells wide, tabulated 100 times per cell | FFT of the model on a padded grid, interpolated at each visibility: by default bilinearly, on a grid padded by 1.2 |
| Residual image | The residual visibilities gridded with the same kernel | The residual visibilities gridded, by Pyralysis 3.3.0 always without a kernel on a grid padded by 1.2 |
| Restored image | The model convolved with a Gaussian fitted to the PSF's main lobe, plus the residual image | The same, with a Gaussian from the second moments of the weighted uv coverage |

CASA's kernel is within 0.02% of the exact dirty image. Pyralysis's defaults are off by a few
percent of the peak around bright emission ([clirai/pyralysis#414](https://gitlab.com/clirai/pyralysis/-/work_items/414)):
the model, fitted through that prediction, makes up for it, and its residual image shows bowls
around bright emission. So the gallery sets Pyralysis's prediction to a Gaussian-sinc kernel
tabulated 8 times per cell, on a grid padded by 2 (`estimator="degridding"`,
`ckernel="gaussian-sinc"`, `ckernel_oversampling=8`, `padding_factor=2.0`), within 0.06% of the
exact dirty image. Pyralysis's options build kernels 3 cells wide only, which leaves its prolate
spheroidal kernel, tclean's, less accurate than this one. Pyralysis 3.3.0 grids its residual image
with its defaults whatever these options say, so the gallery rebuilds each round's residual and
restored images from the round's model with the same kernel and padding
([`restore_pyralysis.py`](https://github.com/miguelcarcamov/wenuray/blob/development/examples/gallery/restore_pyralysis.py)).
The round's noise and peak S/N in the figures are the rebuilt images'; the self-calibration's
decisions were taken on Pyralysis's own. The rebuilt images come from a copy of the data,
calibrated round by round as the run did; where a round's solutions failed and flagged data, the
copy's flags differ from the run's by those few percent of the data.

The two imagers' beams differ too. The moments' Gaussian is the one that matches the PSF's
curvature at its peak, which the longest baselines decide; CASA's follows the main lobe down to
its sidelobes. They agree to a few percent where the PSF is close to a Gaussian (the VLA's, ALMA's
compact configurations), but MeerKAT's PSF, a narrow core on broad shoulders from its dense core of
antennas, gives Pyralysis a beam half the size of CASA's on each axis, and ALMA's long
configurations one 20% smaller. Pyralysis 3.3.0's position angles are also about 90° from CASA's.
A smaller beam gives a lower peak in Jy/beam over the same noise, so for MeerKAT the gallery
restores Pyralysis's images with tclean's beam of the same round (`tclean_beam = true` in
`scenarios.toml`), and their peak S/N compare; elsewhere it keeps Pyralysis's own beam. The beam
does not touch the model or the residual image, nor the self-calibration.

To use the same gridding in your runs, pass these options to the imager:

```python
Pyralysis(geometry, weighting, terms=[...], optimizer="lbfgsb",
          estimator="degridding", ckernel="gaussian-sinc", ckernel_oversampling=8,
          padding_factor=2.0)
```

Its residual images keep Pyralysis's own gridding until Pyralysis grids them with these options.

## The results

```{figure} _static/gallery/summary.png
:alt: Peak S/N and phase errors before and after self-calibration, per scenario

Left: each image's peak S/N before self-calibration, after it, and without gain errors (the ideal
a perfect calibration reaches), with tclean (circles) and Pyralysis (squares). Right: the antenna
phase errors injected, and those left after self-calibration.
```

Both imagers leave the same phase errors, within a degree or two, in every scenario: the
self-calibration doesn't depend on the imager. Six scenarios reach their ideal image with both, to
within 5% of its peak S/N (MeerKAT and the 1.5 GHz VLA A with Pyralysis to within 6%). Three
don't, the three with the most phase error at the longest baselines:

- **The young star's disk, VLA A at 45 GHz**, ends at 82% of the ideal's peak S/N with tclean and
  63% with Pyralysis, with about 20° of phase left. Its errors, 87° rms drifting over two
  minutes, are more than four steps of solutions remove on a source this faint per integration: a
  longer schedule, or solutions combined over spectral windows, would go further.
- **The disks with ALMA C-8**, at 100 and 345 GHz, end at 95% and 84% of the ideal with tclean,
  and within 3% and 7% with Pyralysis. The 12° and 23° left are mostly the noise of solutions
  every 30 s at those frequencies, not error left uncorrected, as the images show. At 345 GHz,
  CLEAN leaves rings around the disk's rings in its residual, 1.2 times the ideal's noise outside
  the mask, which its ideal image doesn't have.

```{figure} _static/gallery/convergence.png
:alt: Image noise against the ideal image's, round by round, per scenario

The image noise over the ideal image's, round by round, with each imager. The solution over the
whole observation removes only the errors' mean; the solutions over two minutes follow the slow
drift, and those every integration the rest, which brings every scenario to its thermal floor.
```

## Each scenario

Each figure shows the image before self-calibration, after it, and without gain errors, with their
residuals below on one scale (±6σ of the ideal image's noise, measured with the source masked),
first with tclean, then with Pyralysis.

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

In an environment with Wenuray and Pyralysis, e.g. Wenuray's Docker image or a conda environment
([Installation](installation.md)):

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

Before 1.2.1, Wenuray warned in some of these runs that the data's noise didn't match their
weights. It was a false alarm: it measured the noise from the difference between consecutive
integrations, which keeps part of a bright source when the gains change within an integration's
time, as they do here on the longest baselines. It now also measures it from the difference of the
parallel hands, which those gains don't touch
([#153](https://github.com/miguelcarcamov/wenuray/issues/153)). The simulated measurement sets also need the patches for Pyralysis 3.3 described in the
[tutorial](tutorial.md#what-pyralysiss-simulation-needs-for-casa).
