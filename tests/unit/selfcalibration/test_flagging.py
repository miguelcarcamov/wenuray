"""Optional residual flagging (#57): flagdata's own names, Wenuray's settings in one place."""
from unittest import mock

import pytest

from wenuray.config import Selection
from wenuray.selfcalibration import flagging
from wenuray.selfcalibration.flagging import NoFlagging, ResidualFlagging
from wenuray.selfcalibration.steps import AmpPhasecal, Phasecal

FLAGDATA_PARAMETERS = frozenset(
    {
        "vis", "mode", "datacolumn", "field", "spw", "uvrange", "action", "flagbackup", "savepars",
        "display", "timedevscale", "freqdevscale", "ntime", "extendflags", "clipminmax"
    }
)


@pytest.fixture(autouse=True)
def flagdata_signature(monkeypatch):
    monkeypatch.setattr(flagging, "_flagdata_parameters", lambda: FLAGDATA_PARAMETERS)


@pytest.fixture
def flagdata(monkeypatch):
    """flagdata whose summary reports 10% flagged before and 15% after."""
    summaries = iter([{"flagged": 10, "total": 100}, {"flagged": 15, "total": 100}])
    task = mock.Mock(
        side_effect=lambda **kwargs: next(summaries) if kwargs["mode"] == "summary" else None
    )
    monkeypatch.setattr(flagging, "flagdata", task)
    return task


class TestOptions:

    def test_options_keep_flagdatas_names(self):
        flagger = ResidualFlagging("rflag", timedevscale=4.0, ntime="scan")

        assert flagger.mode == "rflag" and dict(flagger.options) == {
            "timedevscale": 4.0,
            "ntime": "scan"
        }
        assert repr(flagger) == "ResidualFlagging('rflag', timedevscale=4.0, ntime='scan')"

    @pytest.mark.parametrize("mode", ["manual", "quack", "summary"])
    def test_only_outlier_modes(self, mode):
        with pytest.raises(ValueError, match="rflag"):
            ResidualFlagging(mode)

    @pytest.mark.parametrize(
        "name, source",
        [("datacolumn", "residual"), ("spw", "Selection"), ("flagbackup", "snapshots")]
    )
    def test_settings_wenuray_controls_are_rejected(self, name, source):
        with pytest.raises(TypeError, match=source):
            ResidualFlagging("rflag", **{name: "x"})

    def test_names_that_are_not_flagdata_parameters_are_rejected(self):
        with pytest.raises(TypeError, match="timedev_scale"):
            ResidualFlagging("rflag", timedev_scale=3.0)

    def test_no_default_thresholds_of_wenurays_own(self):
        """A good threshold depends on the data, so flagdata's defaults apply."""
        assert dict(ResidualFlagging().options) == {}


def test_flags_the_residuals_of_the_selection(flagdata):
    selection = Selection(field="target", spw="0~3", uvrange=">10klambda")

    fractions = ResidualFlagging("tfcrop", ntime="scan").flag("target.ms", selection)

    (applied, ) = [c.kwargs for c in flagdata.call_args_list if c.kwargs["mode"] != "summary"]
    assert applied == {
        "vis": "target.ms",
        "mode": "tfcrop",
        "datacolumn": "residual",
        "field": "target",
        "spw": "0~3",
        "uvrange": ">10klambda",
        "action": "apply",
        "flagbackup": False,
        "ntime": "scan"
    }
    assert fractions == (0.1, 0.15)


def test_no_flagging_does_nothing(flagdata):
    assert NoFlagging().flag("target.ms", Selection()) is None
    flagdata.assert_not_called()


class TestSteps:

    def test_steps_flag_nothing_by_default(self):
        assert Phasecal("inf").flagging == NoFlagging()

    def test_a_step_takes_residual_flagging(self):
        step = AmpPhasecal("7min", flagging=ResidualFlagging("rflag", timedevscale=4.0))

        assert step.flagging == ResidualFlagging("rflag", timedevscale=4.0)

    def test_flagging_must_be_a_flagging(self):
        with pytest.raises(TypeError, match="ResidualFlagging"):
            Phasecal("inf", flagging="rflag")
