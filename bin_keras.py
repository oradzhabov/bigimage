import os
from . import inject_keras_modules, inject_tfkeras_modules
from . import init_keras_custom_objects, init_tfkeras_custom_objects
from .solvers import *
from .lr_scheduler import learning_rate_schedulers as lr_sch
from .lr_finder import LRFinder
from .kmodel.data import get_dataloader
from .kmodel.PlotLosses import get_plot_losses  # todo: actually inherit pipeline SHOULD NOT use this
from .tools.predict_contours import predict_field as _predict_field


def _get_keras_modules(**kwargs):
    return kwargs


bim_framework = 'keras'
if 'BIM_FRAMEWORK' in os.environ:
    bim_framework = os.environ['BIM_FRAMEWORK']

if True:  # Automatic environment setup
    import logging
    from distutils.sysconfig import get_python_lib

    # Setup projections environment
    site_packages_path = get_python_lib()
    python_env_root = '\\'.join(site_packages_path.split('\\')[:-2])
    proj_lib_path = os.path.join(python_env_root, 'Library\\share\\proj')
    if not os.path.isdir(proj_lib_path):
        # Try to find installed separately by pip
        proj_lib_path = os.path.join(site_packages_path, 'pyproj\\proj_dir\\share\\proj')
    if os.path.isdir(proj_lib_path):
        logging.info('Required data folder has been found: \"{}\"'.format(proj_lib_path))
        os.environ['PROJ_LIB'] = proj_lib_path
    else:
        logging.warning('There is no required installed package PYPROJ. You need install the latest GDAL package')
    gdal_lib_path = os.path.join(python_env_root, 'Library\\share\\gdal')
    if os.path.isdir(gdal_lib_path):
        logging.info('Required data folder has been found: \"{}\"'.format(gdal_lib_path))
        os.environ['GDAL_DATA'] = gdal_lib_path
    else:
        logging.warning('There is no required installed package GDAL')

if bim_framework == 'keras':
    init_keras_custom_objects()
    AccumGradOptimizer = inject_keras_modules(AccumOptimizer.tf1)()
    LRFinder = inject_keras_modules(LRFinder.tf1)()
    PlotLosses = inject_keras_modules(get_plot_losses)()
    Dataloder = inject_keras_modules(get_dataloader)()
    NativeUnet = inject_keras_modules(get_native_unet)()
    SegmSolver = inject_keras_modules(get_segm_solver)()
    RegrSolver = inject_keras_modules(get_regr_solver)()
    RegrRocksSolver = inject_keras_modules(get_regrrocks_solver)()
    LR_StepDecay = inject_keras_modules(lr_sch.get_step_decay)
    LR_PolynomialDecay = inject_keras_modules(lr_sch.get_polynomial_decay)
    LR_SnapshotEnsembleDecay = inject_keras_modules(lr_sch.get_snapshot_ensemble_decay)()
    #
    predict_field = inject_keras_modules(_predict_field)()
    modules = inject_keras_modules(_get_keras_modules)()
else:
    init_tfkeras_custom_objects()
    AccumGradOptimizer = inject_tfkeras_modules(AccumOptimizer.tf2)()
    LRFinder = inject_tfkeras_modules(LRFinder.tf2)()
    PlotLosses = inject_tfkeras_modules(get_plot_losses)()
    Dataloder = inject_tfkeras_modules(get_dataloader)()
    NativeUnet = inject_tfkeras_modules(get_native_unet)()
    SegmSolver = inject_tfkeras_modules(get_segm_solver)()
    RegrSolver = inject_tfkeras_modules(get_regr_solver)()
    RegrRocksSolver = inject_tfkeras_modules(get_regrrocks_solver)()
    LR_StepDecay = inject_tfkeras_modules(lr_sch.get_step_decay)
    LR_PolynomialDecay = inject_tfkeras_modules(lr_sch.get_polynomial_decay)
    LR_SnapshotEnsembleDecay = inject_tfkeras_modules(lr_sch.get_snapshot_ensemble_decay)()
    #
    predict_field = inject_tfkeras_modules(_predict_field)()
    modules = inject_tfkeras_modules(_get_keras_modules)()
