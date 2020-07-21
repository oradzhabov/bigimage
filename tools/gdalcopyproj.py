from osgeo import gdal


def get_projection(inp):
    dataset = gdal.Open(inp)
    if dataset is None:
        print('Unable to open ', inp, ' for reading')
        return None

    projection = dataset.GetProjection()
    geotransform = dataset.GetGeoTransform()

    if projection is None and geotransform is None:
        print('No projection or geotransform found on file ' + inp)
        return None

    gcp_count = dataset.GetGCPCount()
    gcps = dataset.GetGCPs()
    gcp_proj = dataset.GetGCPProjection()

    del dataset

    result = dict()
    result['projection'] = projection
    result['geotransform'] = geotransform
    result['gcp_count'] = gcp_count
    result['gcps'] = gcps
    result['gcp_proj'] = gcp_proj

    return result


def apply_projection(inp_proj, output):
    dataset2 = gdal.Open(output, gdal.GA_Update)

    if dataset2 is None:
        print('Unable to open ', output, ' for writing')
        return -1

    geotransform = inp_proj['geotransform']
    projection = inp_proj['projection']
    gcp_count = inp_proj['gcp_count']
    gcps = inp_proj['gcps']
    gcp_proj = inp_proj['gcp_proj']

    if geotransform is not None and geotransform != (0, 1, 0, 0, 0, 1):
        dataset2.SetGeoTransform(geotransform)

    if projection is not None and projection != '':
        dataset2.SetProjection(projection)

    if gcp_count != 0:
        dataset2.SetGCPs(gcps, gcp_proj)

    del dataset2

    return 0


def copy_projection(inp, output):
    # Copy GDAL projection metadata from one file into other.

    dataset = gdal.Open(inp)
    if dataset is None:
        print('Unable to open', inp, 'for reading')
        return -1

    projection = dataset.GetProjection()
    geotransform = dataset.GetGeoTransform()

    if projection is None and geotransform is None:
        print('No projection or geotransform found on file ' + inp)
        return -1

    dataset2 = gdal.Open(output, gdal.GA_Update)

    if dataset2 is None:
        print('Unable to open', output, 'for writing')
        return -1

    if geotransform is not None and geotransform != (0, 1, 0, 0, 0, 1):
        dataset2.SetGeoTransform(geotransform)

    if projection is not None and projection != '':
        dataset2.SetProjection(projection)

    gcp_count = dataset.GetGCPCount()
    if gcp_count != 0:
        dataset2.SetGCPs(dataset.GetGCPs(), dataset.GetGCPProjection())

    del dataset
    del dataset2

    return 0


def create_tiles(args):
    from .gdal2tiles import GDAL2Tiles

    err_code = 0
    argv = gdal.GeneralCmdLineProcessor(args)
    try:
        if argv:
            gdal2tiles = GDAL2Tiles(argv[1:])
            gdal2tiles.process()
    except Exception:
        err_code = -1

    return err_code
