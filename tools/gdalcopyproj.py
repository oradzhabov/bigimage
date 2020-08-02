import logging
from osgeo import gdal


def create_empty(fname, inp_proj, clear_img=False):
    """
    Create empty geoTiff-image based on metadata from another geoTiff image. Metadata could be obtained by method
    get_projection().
    :param fname: Full path to new created image
    :param inp_proj: Projection dictionary metadata from another geoTiff image
    :param clear_img: If True, fill new image by 0(spends much more time). Default is False.
    :return:
    """
    driver = gdal.GetDriverByName('GTiff')

    outfn = fname
    dtype = gdal.GDT_Byte
    inp_shape = inp_proj['shape']
    nbands = 4
    nodata = 255

    ds = driver.Create(outfn, inp_shape[1], inp_shape[0], nbands, dtype, options=['COMPRESS=LZW', 'TILED=YES'])
    ds.SetProjection(inp_proj['projection'])
    ds.SetGeoTransform(inp_proj['geotransform'])

    if clear_img:
        for i in range(nbands):
            ds.GetRasterBand(1+i).Fill(0)
            ds.GetRasterBand(1+i).SetNoDataValue(nodata)
        ds.FlushCache()

    del ds  # close file


def put_into_image(fname, start_xy, bgra):
    """
    Put into already created geoTiff image BGRA patch by specific XY-position.
    :param fname: Full path to existing geoTiff image
    :param start_xy: X/Y pixel position where patch will be placed
    :param bgra: BGRA (8-bit per channel)
    :return:
    """
    rgba = [2, 1, 0, 3]
    ds = gdal.Open(fname, gdal.GA_Update)
    for band_ind in range(4):
        out_band = ds.GetRasterBand(band_ind + 1)
        # GDAL operate with bands first dim
        out_band.WriteArray(bgra[..., rgba[band_ind]], xoff=int(start_xy[0]), yoff=int(start_xy[1]))

    del ds  # close file


def get_projection(inp):
    dataset = gdal.Open(inp)
    if dataset is None:
        logging.error('Unable to open {}'.format(inp))
        return None

    projection = dataset.GetProjection()
    geotransform = dataset.GetGeoTransform()

    if projection is None and geotransform is None:
        logging.error('No projection or geotransform found on file {}'.format(inp))
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
        logging.error('Unable to open {}'.format(output))
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
    src_projection = get_projection(inp)

    if src_projection is None:
        return -1

    ret_code = apply_projection(src_projection, output)

    return ret_code


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
