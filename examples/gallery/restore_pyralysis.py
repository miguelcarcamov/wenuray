"""
Rebuilds the residual and restored images of a Pyralysis model, gridding the residual with a
convolution kernel. Runs with Pyralysis's Python (WENURAY_PYRALYSIS_PYTHON), reading a request on
stdin like Wenuray's Pyralysis worker, whose option parser it uses:

    {"worker": <_pyralysis_worker.py>, "ms": ..., "taql": ..., "options": {...},
     "outputs": {"model": ..., "residual": ..., "restored": ...},
     "beam": [major, minor, position angle]}

"beam" (arcsec, arcsec, degrees East of North; optional) restores the model with that beam instead
of Pyralysis's. Pyralysis 3.3.0 derives its beam from the second moments of the weighted uv
coverage, not from a fit to the PSF as CASA does: on a PSF with a narrow core and broad shoulders
(MeerKAT's, ALMA's long configurations) its beam is up to half the size of CASA's.

The model is options["initial_model"]; it is predicted with options["estimator"], "ckernel",
"ckernel_oversampling" and "padding_factor", and the residual is gridded with the same. Pyralysis
3.3.0 grids the residual image at padding 1.2 without a kernel whatever those options say, which
puts bowls of a few percent of the peak around bright emission; this script puts them through
until Pyralysis does (clirai/pyralysis#414).
"""
import importlib.util
import json
import sys
from pathlib import Path


def main() -> None:
    request = json.load(sys.stdin)
    spec = importlib.util.spec_from_file_location("worker", request["worker"])
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)

    from pyralysis.cli.optimization import (
        build_pipeline_imager_config, normalize_dask_runtime_for_gpu, parse_dask_runtime_options
    )
    from pyralysis.cli.optimization.registries import CKERNELS
    from pyralysis.pipelines.imager import (
        ApplyVisibilityWeighting, ExportResults, ImagerContext, ImagerPipeline, LoadData,
        RestoreImage, SetupDaskCluster, SetupImageGrid, SetupMeasurementOperator,
        TeardownDaskCluster
    )
    import astropy.units as un
    from pyralysis.reconstruction import restore
    from pyralysis.reconstruction.psf import PSF
    from pyralysis.transformers import DirtyMapper

    outputs = request["outputs"]
    args = worker.parse(
        request["options"], [
            "--input-path", request["ms"], "--output-model", outputs["model"], "--output-residual",
            outputs["residual"], "--output-restored", outputs["restored"], "--restore",
            "--no-hermitian-symmetry"
        ]
    )
    for name in ("input_path", "output_model", "output_residual", "output_restored"):
        setattr(args, name, Path(getattr(args, name)))
    dask_options = normalize_dask_runtime_for_gpu(parse_dask_runtime_options(args))
    config = build_pipeline_imager_config(args, dask_options.to_pipeline_config())
    config.update(data_column="CORRECTED_DATA", taql_query=request.get("taql") or None)

    def grid_residual_image(
        dataset, model_image, *, restore_stokes, hermitian_symmetry, imsize=None, w_term=None
    ):
        ny, nx = restore._resolve_imsize(model_image, imsize)
        kernel = CKERNELS[args.ckernel](
            cellsize=model_image.cellsize, oversampling_factor=args.ckernel_oversampling
        )
        mapper = DirtyMapper(
            input_data=dataset,
            imsize=[ny, nx],
            cellsize=model_image.cellsize,
            stokes=restore_stokes,
            data_column="RESIDUAL_DATA",
            hermitian_symmetry=hermitian_symmetry,
            w_term=w_term,
            padding_factor=args.padding_factor,
            ckernel_object=kernel,
        )
        images, _ = mapper.transform()
        return images

    # restore_images looks the function up when it runs
    restore.grid_residual_image = grid_residual_image

    class RestoreWithBeam(RestoreImage):
        """
        RestoreImage with the request's beam, which also goes in the images' headers
        """

        def execute(self, context: ImagerContext) -> ImagerContext:
            if request.get("beam") and context.dataset is not None:
                major, minor, angle = request["beam"]
                context.dataset.psf = PSF(
                    major=[major] * un.arcsec, minor=[minor] * un.arcsec, pa=[angle] * un.deg
                )
            return super().execute(context)

    steps = [
        SetupDaskCluster(),
        LoadData(),
        ApplyVisibilityWeighting(),
        SetupImageGrid(),
        SetupMeasurementOperator(),
        RestoreWithBeam(),
        ExportResults(),
        TeardownDaskCluster(),
    ]
    context = ImagerContext(config=config)
    try:
        ImagerPipeline(steps=steps).run(context)
    finally:
        TeardownDaskCluster().execute(context)


if __name__ == "__main__":
    main()
