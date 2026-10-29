"""Frozen configuration objects (#22): validated once, derived with replace, round-trip via TOML."""
import dataclasses
import tomllib

import pytest
import tomli_w

from wenuray.config import (
    ApplyOptions, ApplyTarget, ImageGeometry, Selection, SolveOptions, Weighting, to_dict
)

CONFIGS = [Selection, SolveOptions, ApplyOptions, ImageGeometry, Weighting]


@pytest.mark.parametrize("config", CONFIGS)
def test_configs_are_frozen(config):
    instance = config()
    name = dataclasses.fields(instance)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(instance, name, None)


@pytest.mark.parametrize("config", CONFIGS)
def test_replace_rejects_unknown_names(config):
    with pytest.raises(TypeError):
        dataclasses.replace(config(), not_a_parameter=1)


@pytest.mark.parametrize(
    "instance", [
        Selection(field="target", spw="1~3", uvrange=">50klambda"),
        SolveOptions(refant="DA51", minblperant=6, minsnr=2.5, gaintype="G", combine="spw"),
        ApplyOptions(applymode="calonly", interp="nearest"),
        ApplyOptions(targets=(ApplyTarget("line.ms", Selection(spw="4")), ApplyTarget("b.ms"))),
        ImageGeometry(imsize=(1024, 512), cell="0.05arcsec", phasecenter="J2000 10h00m +30d00m"),
        ImageGeometry(),
        Weighting(weighting="briggs", robust=-0.5, uvtaper=("0.1arcsec", )),
    ]
)
def test_toml_round_trip(instance):
    text = tomli_w.dumps(to_dict(instance))

    assert type(instance)(**tomllib.loads(text)) == instance


def test_weighting_defaults_to_natural():
    assert Weighting().weighting == "natural"


def test_square_imsize_from_one_number():
    assert ImageGeometry(imsize=512).imsize == (512, 512)


def test_unset_imsize_is_left_out_of_the_dictionary():
    assert "imsize" not in to_dict(ImageGeometry())


def test_single_uvtaper_string_becomes_a_tuple():
    assert Weighting(uvtaper="0.1arcsec").uvtaper == ("0.1arcsec", )


@pytest.mark.parametrize(
    "build, match", [
        (lambda: Selection(spw=1), "Selection.spw must be a string"),
        (lambda: SolveOptions(minblperant=0), "minblperant"),
        (lambda: SolveOptions(minblperant=True), "minblperant"),
        (lambda: SolveOptions(minsnr=-1), "minsnr"),
        (lambda: SolveOptions(gaintype="K"), "gaintype"),
        (lambda: ApplyOptions(applymode="apply"), "applymode"),
        (lambda: ApplyOptions(interp=""), "interp"),
        (lambda: ImageGeometry(imsize=(512, )), "imsize"),
        (lambda: ImageGeometry(imsize=(512, -1)), "imsize"),
        (lambda: ImageGeometry(imsize=512.0), "imsize"),
        (lambda: Weighting(weighting="robust"), "weighting"),
        (lambda: Weighting(robust=3), "robust"),
        (lambda: Weighting(uvtaper=(0.1, )), "uvtaper"),
    ]
)
def test_invalid_values_raise_at_construction(build, match):
    with pytest.raises((TypeError, ValueError), match=match):
        build()


def test_apply_targets_are_built_from_tables():
    options = ApplyOptions(targets=[{"ms": "line.ms", "selection": {"spw": "4"}}])

    assert options.targets == (ApplyTarget("line.ms", Selection(spw="4")), )


@pytest.mark.parametrize(
    "build, match",
    [
        (lambda: ApplyTarget(""), "measurement set"),
        (lambda: ApplyOptions(targets=["line.ms"]), "ApplyTarget"),
        (lambda: ApplyTarget("line.ms", selection="4"), "Selection"),
    ],
)
def test_invalid_apply_targets_raise(build, match):
    with pytest.raises((TypeError, ValueError), match=match):
        build()
