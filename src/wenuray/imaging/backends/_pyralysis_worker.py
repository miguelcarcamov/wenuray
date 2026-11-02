"""
Runs one Pyralysis reconstruction for Wenuray's Pyralysis imager (#24), in a process of its own.

casatools and python-casacore (which Pyralysis reads measurement sets with) each bundle a casacore
build, and the two crash in one process. Wenuray therefore runs this file as a script, so it
imports neither wenuray nor casatools. It reads a JSON request from stdin and prints a JSON reply
as the last line of its output (Pyralysis prints its progress before it):

    {"check": true, "options": {...}}                    -> {"error": ""} (validates the options)
    {"versions": true}                                   -> {"error": "", "versions": {...}}
    {"ms": ..., "taql": ..., "options": {...}, "outputs": {"model": ..., "residual": ..., "restored": ...}}
                                                         -> {"error": ""} after writing MODEL_DATA
"""
import argparse
import contextlib
import io
import json
import os
import re
import sys

# Wenuray's pyralysis.py lives next to this file and would shadow the pyralysis package
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [path for path in sys.path if os.path.abspath(path or ".") != _HERE]
from dataclasses import dataclass
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    from pyralysis.cli.optimization import add_optimization_cli_args

    result = argparse.ArgumentParser(prog="pyralysis", exit_on_error=False)
    add_optimization_cli_args(result)
    return result


def to_argv(cli: argparse.ArgumentParser, options: dict) -> list:
    """
    Pyralysis CLI arguments from options named like the CLI's destinations (max_iter=100, ...)
    """
    actions = {action.dest: action for action in cli._actions if action.option_strings}
    argv = []
    for name, value in options.items():
        if name not in actions:
            raise ValueError("{0!r} is not a Pyralysis option".format(name))
        action = actions[name]
        flags = action.option_strings
        if isinstance(action, argparse.BooleanOptionalAction):
            argv.append(flags[0] if value else "--no-" + flags[0][2:])
        elif isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
            if bool(value) == isinstance(action, argparse._StoreTrueAction):
                argv.append(flags[0])
        elif isinstance(value, (list, tuple)):
            if action.nargs in (None, "?", 1):  # a repeated option, e.g. --term a --term b
                for item in value:
                    argv += [flags[0], str(item)]
            else:
                argv += [flags[0]] + [str(item) for item in value]
        else:
            argv += [flags[0], str(value)]
    return argv


def parse(options: dict, extra: list) -> argparse.Namespace:
    """
    Parses the options with Pyralysis's own CLI parser, so they are checked by Pyralysis
    """
    cli = parser()
    messages = io.StringIO()
    with contextlib.redirect_stderr(messages):
        try:
            return cli.parse_args(to_argv(cli, options) + extra)
        except argparse.ArgumentError as error:
            raise ValueError(str(error)) from None
        except SystemExit:  # argparse reports some errors by exiting
            lines = messages.getvalue().strip().splitlines()
            raise ValueError(lines[-1] if lines else "invalid Pyralysis options") from None


def run(request: dict) -> None:
    from pyralysis.cli.optimization import (
        build_optimizer, build_pipeline_imager_config, normalize_dask_runtime_for_gpu,
        optimize_kwargs_from_args, parse_dask_runtime_options, parse_terms
    )
    from pyralysis.optimization import ObjectiveFunction
    from pyralysis.pipelines.base import Step
    from pyralysis.pipelines.imager import (
        ApplyVisibilityWeighting, CheckpointModelResiduals, ExportOptimizedModel, ExportResults,
        ImagerContext, ImagerPipeline, LoadData, RestoreImage, SetupDaskCluster, SetupImageGrid,
        SetupMeasurementOperator, TeardownDaskCluster
    )

    ms, outputs = request["ms"], request["outputs"]
    args = parse(
        request["options"], [
            "--input-path", ms, "--output-model", outputs["model"], "--output-residual",
            outputs["residual"], "--output-restored", outputs["restored"], "--restore",
            "--no-hermitian-symmetry"
        ]
    )
    for name in ("input_path", "output_model", "output_residual", "output_restored"):
        setattr(args, name, Path(getattr(args, name)))
    dask_options = normalize_dask_runtime_for_gpu(parse_dask_runtime_options(args))
    config = build_pipeline_imager_config(args, dask_options.to_pipeline_config())
    # Self-calibration fits the calibrated data of the request's selection (LoadData raises if
    # CORRECTED_DATA isn't loaded), and the next gaincal reads only MODEL_DATA
    config.update(
        data_column="CORRECTED_DATA",
        taql_query=request.get("taql") or None,
        checkpoint_columns=["MODEL_DATA"],
    )

    @dataclass
    class Reconstruct(Step[ImagerContext]):

        def execute(self, context):
            terms = parse_terms(args.term, context.measurement_operator, dataset=context.dataset)
            context.objective = ObjectiveFunction(
                terms=terms, parameter=context.image, persist_gradient=True
            )
            optimizer = build_optimizer(
                args.optimizer,
                context.objective,
                context.image,
                args,
                mask=context.mask,
                dataset=context.dataset
            )
            context.image = optimizer.optimize(**optimize_kwargs_from_args(args, optimizer))
            context.optimizer = optimizer
            return context

    steps = [
        SetupDaskCluster(),
        LoadData(),
        ApplyVisibilityWeighting(),
        SetupImageGrid(),
        SetupMeasurementOperator(),
        Reconstruct(),
        ExportOptimizedModel(),
        CheckpointModelResiduals(),
        RestoreImage(),
        ExportResults(),
        TeardownDaskCluster(),
    ]
    context = ImagerContext(config=config)
    try:
        ImagerPipeline(steps=steps).run(context)
    finally:
        TeardownDaskCluster().execute(context)


# The first Pyralysis whose pipeline loads CORRECTED_DATA with a selection and writes MODEL_DATA
# (pyralysis#358, #359), leaves visibilities off the uv grid out of fits (#369) and follows the
# loss's gradient with degridding (#377)
MINIMUM_PYRALYSIS = (3, 2, 1)


class PyralysisVersionError(ImportError):
    pass


def require_pyralysis(minimum: tuple = MINIMUM_PYRALYSIS) -> None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        found = version("pyralysis")
    except PackageNotFoundError:
        found = None
    numbers = tuple(int(part) for part in re.findall(r"\d+", found or "")[:3])
    if found is None or numbers < minimum:
        raise PyralysisVersionError(
            "The Pyralysis imager needs Pyralysis {0} or later ({1}): pip install "
            "'wenuray[pyralysis]' --extra-index-url "
            "https://artefact.skao.int/repository/pypi-internal/simple".format(
                ".".join(map(str, minimum)), "found " + found if found else "not installed"
            )
        )


# What a run record names of the worker's environment, when installed: Pyralysis, and the packages
# its results depend on, on CPU and GPU
VERSIONED = (
    "pyralysis", "numpy", "numba", "numba-cuda", "dask", "python-casacore", "cupy-cuda12x",
    "cupy-cuda13x"
)


def versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version

    found = {}
    for distribution in VERSIONED:
        try:
            found[distribution] = version(distribution)
        except PackageNotFoundError:
            pass
    return found


class ThreadingLayerError(RuntimeError):
    pass


def require_thread_safe_numba() -> None:
    """
    Dask runs Pyralysis's parallel Numba kernels from several threads, one per chunk. Only Numba's
    TBB and OpenMP layers allow that; without their libraries Numba falls back to workqueue, which
    aborts the process when a measurement set is read in more than one chunk.
    """
    import numba

    if "NUMBA_THREADING_LAYER" not in os.environ:
        numba.config.THREADING_LAYER = "threadsafe"

    @numba.njit(parallel=True, cache=False)
    def probe(n):
        total = 0
        for i in numba.prange(n):
            total += i
        return total

    try:
        probe(2)
        layer = numba.threading_layer()
    except ValueError as error:  # no layer could be loaded
        layer, reason = None, str(error).strip().splitlines()[0]
    else:
        reason = "Numba uses the {0} layer".format(layer)
    if layer not in ("tbb", "omp"):
        raise ThreadingLayerError(
            "Pyralysis needs Numba's OpenMP or TBB threading layer ({0}). Install OpenMP's "
            "runtime (e.g. apt install libgomp1), or TBB (e.g. conda install tbb)".format(reason)
        )


def main() -> int:
    request = json.load(sys.stdin)
    try:
        require_pyralysis()
        require_thread_safe_numba()
        if request.get("versions"):
            print(json.dumps({"error": "", "versions": versions()}))
            return 0
        if request.get("check"):
            parse(request["options"], ["--input-path", "check.ms"])
        else:
            run(request)
    except Exception as error:  # reported to Wenuray, which raises it
        print(json.dumps({"error": "{0}: {1}".format(type(error).__name__, error)}))
        return 1
    print(json.dumps({"error": ""}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
