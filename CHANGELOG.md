# Changelog

All notable changes to Wenuray. The entries are built with
[towncrier](https://towncrier.readthedocs.io/) from the news fragments in
[`changes/`](https://github.com/miguelcarcamov/wenuray/tree/development/changes); see its README
to add one.

<!-- towncrier release notes start -->

## [1.1.0](https://github.com/miguelcarcamov/wenuray/releases/tag/v1.1.0) (2026-10-29)

### Features

- Support Python 3.11 (with CASA 6.7.0, the newest CASA with 3.11 wheels), and test every supported Python version with tox. ([#99](https://github.com/miguelcarcamov/wenuray/issues/99))

### Bug fixes

- An empty spw selection now means the spectral windows that hold data, so combined-spw solutions are applied when the measurement set has unused `SPECTRAL_WINDOW` rows. ([#102](https://github.com/miguelcarcamov/wenuray/issues/102))
- Relative file options of an imager, e.g. Pyralysis's `mask_path`, are resolved against the recipe file that names them; given from Python, Pyralysis finds them from the caller's working directory. ([#103](https://github.com/miguelcarcamov/wenuray/issues/103))

### Documentation

- Document the release naming convention: each major version is named after a woman who shaped radio astronomy. ([#97](https://github.com/miguelcarcamov/wenuray/issues/97))
- A friendlier README, introducing self-calibration and why Wenuray exists, with new installation, guide and imagers pages, and a `CITATION.cff` for GitHub's "Cite this repository" and Zenodo. ([#106](https://github.com/miguelcarcamov/wenuray/issues/106))
- The README badge, the BibTeX entry and `CITATION.cff` give Wenuray's Zenodo DOI, [10.5281/zenodo.23775089](https://doi.org/10.5281/zenodo.23775089).

### Packaging and maintenance

- CI installs what the CASA data fetch needs before running it, so pull requests into `master` can run the tests, and a fallback to cached measures data is reported as a warning. ([#109](https://github.com/miguelcarcamov/wenuray/issues/109))
- Dependency updates: multimethod 2.1, python-dateutil 3.0, newer Sphinx extensions and theme, and newer GitHub Actions.
- Keep a changelog, `CHANGELOG.md`, built with towncrier from a news fragment per pull request.

## [1.0.0](https://github.com/miguelcarcamov/wenuray/releases/tag/v1.0.0) (2026-10-26)

**"Payne-Scott"**, after Ruby Payne-Scott (1912–1981), the first woman radio astronomer and a
pioneer of radio interferometry.

The first release as Wenuray, previously *snow*: a redesign around one self-calibration loop
(#22). See the [release notes](https://github.com/miguelcarcamov/wenuray/releases/tag/v1.0.0) and
the [migration guide](https://github.com/miguelcarcamov/wenuray/blob/development/docs/migration.md).
