import os
from . import inject_keras_modules, inject_tfkeras_modules
from . import init_keras_custom_objects, init_tfkeras_custom_objects
from .solvers import *
from .lr_scheduler import learning_rate_schedulers as lr_sch
from .kmodel.data import get_dataloader
from .kmodel.PlotLosses import get_plot_losses  # todo: actually inherit pipeline SHOULD NOT use this
from .tools.predict_contours import predict_contours as _predict_contours


def _get_keras_modules(**kwargs):
    return kwargs


bim_framework = 'keras'
if 'BIM_FRAMEWORK' in os.environ:
    bim_framework = os.environ['BIM_FRAMEWORK']

if bim_framework == 'keras':
    init_keras_custom_objects()
    PlotLosses = inject_keras_modules(get_plot_losses)()
    Dataloder = inject_keras_modules(get_dataloader)()
    SegmSolver = inject_keras_modules(get_segm_solver)()
    RegrSolver = inject_keras_modules(get_regr_solver)()
    RegrRocksSolver = inject_keras_modules(get_regrrocks_solver)()
    LR_StepDecay = inject_keras_modules(lr_sch.get_step_decay)
    LR_PolynomialDecay = inject_keras_modules(lr_sch.get_polynomial_decay)
    LR_SnapshotEnsembleDecay = inject_keras_modules(lr_sch.get_snapshot_ensemble_decay)()
    #
    predict_contours = inject_keras_modules(_predict_contours)()
    modules = inject_keras_modules(_get_keras_modules)()
else:
    init_tfkeras_custom_objects()
    PlotLosses = inject_tfkeras_modules(get_plot_losses)()
    Dataloder = inject_tfkeras_modules(get_dataloader)()
    SegmSolver = inject_tfkeras_modules(get_segm_solver)()
    RegrSolver = inject_tfkeras_modules(get_regr_solver)()
    RegrRocksSolver = inject_tfkeras_modules(get_regrrocks_solver)()
    LR_StepDecay = inject_tfkeras_modules(lr_sch.get_step_decay)
    LR_PolynomialDecay = inject_tfkeras_modules(lr_sch.get_polynomial_decay)
    LR_SnapshotEnsembleDecay = inject_tfkeras_modules(lr_sch.get_snapshot_ensemble_decay)()
    #
    predict_contours = inject_tfkeras_modules(_predict_contours)()
    modules = inject_tfkeras_modules(_get_keras_modules)()
