from osgeo import gdal


def copy_projection(inp, output):
    # Copy GDAL projection metadata from one file into other.

    dataset = gdal.Open(inp)
    if dataset is None:
        print('Unable to open', inp, 'for reading')
        return -1

    projection = dataset.GetProjection()
    geotransform = dataset.GetGeoTransform()

    if projection is None and geotransform is None:
        print('No projection or geotransform found on file' + input)
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

    dataset = None
    dataset2 = None

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
