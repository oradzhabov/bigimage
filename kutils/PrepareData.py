import logging
import os
import re
import json
import numpy as np
import cv2
import time
from .color2height import color2height
from .mapLoc2EPSG import map_epsg2loc
from osgeo import gdal, osr, ogr
import affine


DATETIME_FORMAT = '%Y-%m-%d'  # '%Y-%m-%d_%H-%M-%S'


def map_contour_meter_pix(img_shape, contours_m, x0_m, y0_m, x1_m, y1_m):
    imgw_m = x1_m - x0_m
    imgh_m = y1_m - y0_m

    imgh_px = img_shape[0]
    imgw_px = img_shape[1]

    # Map pixel coords to local coords
    maxx_px = imgw_px - 1
    maxy_px = imgh_px - 1
    contours_px = []
    coef_w = imgw_m/maxx_px
    coef_h = imgh_m/maxy_px
    for cntr in contours_m:
        cntr_px = []
        for pnt in cntr:
            x_m = pnt[0]
            y_m = pnt[1]
            # x_m = x0_m + x_px*coef_w
            x_px = int((x_m - x0_m)/coef_w)
            # y_m = y1_m - y_px*coef_h # Y-axis in image is opposite to Y-axis in geoloc CS
            y_px = int((y1_m - y_m)/coef_h)
            cntr_px.append([x_px, y_px])
        contours_px.append(cntr_px)
    return contours_px


def retrieve_pixel_value(geo_coords, data_source):
    """ Map points from CRS to image space """

    # Get matrix to transform from CRS to image/pixel space
    forward_transform = affine.Affine.from_gdal(*data_source.GetGeoTransform())
    reverse_transform = ~forward_transform

    result = []
    for p in geo_coords:
        ppx = reverse_transform * p
        result.append((int(ppx[0] + 0.5), int(ppx[1] + 0.5)))

    return result


def get_via_item(proj_dir, mppx, src_shp_fname='shp.shp', src_epsg=4326):
    """ Create VIA item based on project content and geometry poygones stored in shape-file """

    geotiff_fname = os.path.relpath(os.path.join(proj_dir, './orthophoto/orthophoto_export.tif'))
    in_shape_file = os.path.join(proj_dir, src_shp_fname)

    for f in [in_shape_file, geotiff_fname]:
        if not os.path.isfile(f):
            logging.error('There is no input file {}'.format(f))
            return None

    ds = gdal.Open(geotiff_fname)
    ds_epsg = int(osr.SpatialReference(wkt=ds.GetProjection()).GetAttrValue('AUTHORITY', 1))

    ds_transofrmation = ds.GetGeoTransform()
    ds_mppx = abs(ds_transofrmation[1])
    coords_scale = ds_mppx / mppx

    # input SpatialReference
    in_spat_ref = osr.SpatialReference()
    in_spat_ref.ImportFromEPSG(src_epsg)

    # output SpatialReference
    out_spat_ref = osr.SpatialReference()
    out_spat_ref.ImportFromEPSG(ds_epsg)

    # create the CoordinateTransformation
    coord_trans = osr.CoordinateTransformation(in_spat_ref, out_spat_ref)

    # get the input layer
    driver = ogr.GetDriverByName('ESRI Shapefile')
    in_dataset = driver.Open(in_shape_file, 0)  # 0 means read-only. 1 means writeable.
    in_layer = in_dataset.GetLayer()
    # inFeatureCount = in_layer.GetFeatureCount()

    attribute_names = [field.name for field in in_layer.schema]

    size = int(-1)
    regions = list()
    file_attributes = dict()
    for inFeature in in_layer:
        # get the input geometry
        geom = inFeature.GetGeometryRef()
        if geom is None:
            logging.warning('Feature has no geometry. Skip feature')
            continue

        # Map the geometry from shape-file space to geo-tiff space
        geom.Transform(coord_trans)

        pts = geom.GetGeometryRef(0).GetPoints()

        # Map points from geotiff space into pixel space
        contour_px = retrieve_pixel_value(pts, ds)

        # Store results to VIA sub-structure
        shape_attributes = dict()
        shape_attributes['name'] = 'polygon'
        shape_attributes['all_points_x'] = [int(px[0] * coords_scale) for px in contour_px]
        shape_attributes['all_points_y'] = [int(px[1] * coords_scale) for px in contour_px]
        if len(attribute_names) > 0:
            region_attributes = dict(zip(attribute_names, [inFeature[field_name] for field_name in attribute_names]))
        else:
            region_attributes = dict()
        attributes = dict({'shape_attributes': shape_attributes, 'region_attributes': region_attributes})
        regions.append(attributes)
    in_layer.ResetReading()  # You must call ResetReading if you want to start iterating over the layer again.

    f_item = dict({'filename': geotiff_fname, 'size': size, 'regions': regions, 'file_attributes': file_attributes})

    return tuple((geotiff_fname, f_item))


def map_shapefiles_to_via(rootdir, customer_list, proj_list, src_shp_fname, mppx):
    via_items = dict()
    for customer in customer_list:
        for proj_id in proj_list:
            proj_path = os.path.join(rootdir, customer, str(proj_id))

            item = get_via_item(proj_path, mppx, src_shp_fname=src_shp_fname, src_epsg=4326)
            if item is not None:
                # Overwrite special field by uniq id
                item[1]['filename'] = '../imgs/{}_{}.png'.format(customer, proj_id)
                via_items[item[0]] = item[1]

    return via_items


def get_geoloc_gata(gdalinfo, mapdata):
    # Get absolute values
    with open(gdalinfo, "r") as f:
        for line in f:
            if line.startswith('Upper Left'):
                _ = [float(value.strip()) for value in re.search(r'\((.*?)\)', line).group(1).split(',')]
            if line.startswith('Lower Left'):
                ll = [float(value.strip()) for value in re.search(r'\((.*?)\)', line).group(1).split(',')]
            if line.startswith('Upper Right'):
                ur = [float(value.strip()) for value in re.search(r'\((.*?)\)', line).group(1).split(',')]
            if line.startswith('Lower Right'):
                lr = [float(value.strip()) for value in re.search(r'\((.*?)\)', line).group(1).split(',')]

    odm_center = mapdata['ODMCenter']

    # Relocate coordinates to the ODMCenter
    # x0_m = ul[0] - odm_center[0]
    # y1_m = ul[1] - odm_center[1]
    x0_m = ll[0] - odm_center[0]
    # y0_m = ll[1] - odm_center[1]
    # x1_m = ur[0] - odm_center[0]
    y1_m = ur[1] - odm_center[1]
    x1_m = lr[0] - odm_center[0]
    y0_m = lr[1] - odm_center[1]

    return x0_m, y0_m, x1_m, y1_m


def get_raster_info(img_fname):
    gtif = gdal.Open(img_fname)
    img_shape = [gtif.RasterYSize, gtif.RasterXSize]

    # Load as a gdal image to get geotransform (world file) info
    geo_trans = gtif.GetGeoTransform()

    mppx = np.fabs((geo_trans[1], geo_trans[5]))
    # tiepoint = np.array((geo_trans[0], geo_trans[3]))

    # Obtain length UNITS
    prj = gtif.GetProjection()
    srs = osr.SpatialReference(wkt=prj)
    unit = srs.GetAttrValue('unit')  # todo: could return None. In that way behavior is not correct
    if unit is None:
        logging.error('Try to read raster info from file {} which does not contain these data'.format(img_fname))
        return None, None
    # logging.info('GeoTIFF length units: {}'.format(unit))
    scale_to_meter = 1.0
    if unit != 'metre':
        scale_to_meter = 0.3048

    mppx = mppx * scale_to_meter

    return img_shape, mppx


def create_orthophoto(dataset_path, dst_mppx, dest_img_fname):
    logging.info('Mapping orthographic image into {}'.format(dest_img_fname))

    recreated = False
    src_img_fname = os.path.join(dataset_path, 'orthophoto/orthophoto_export.tif')
    if not os.path.isfile(src_img_fname):
        logging.error('File {} does not exist'.format(src_img_fname))
        return False, recreated

    if not os.path.isfile(dest_img_fname):
        src_img_shape, src_mppx = get_raster_info(src_img_fname)
        if src_img_shape is None:
            logging.error('Cannot create orthophoto because file {} does not have raster info'.format(src_img_fname))
            return False, recreated

        rescale = src_mppx / dst_mppx
        gdal.Translate(dest_img_fname, src_img_fname,
                       options="-outsize {} {} -ot Byte -r bilinear".
                       format(int(src_img_shape[1] * rescale[0]), int(src_img_shape[0] * rescale[1])))
        recreated = True

    return True, recreated


def create_heightmap_color(dataset_path, dst_img_shape, dest_himg_fname):
    fname = os.path.join(dataset_path, 'dem/color_relief/color_relief.tif')
    if not os.path.isfile(fname):
        logging.error('File {} does not exist'.format(fname))
        return False
    himg_bgr = cv2.imread(fname)[:, :, :3]
    if himg_bgr is None:
        logging.error('Cannot read file {}'.format(fname))
        return False
    himg_gray = color2height(os.path.join(dataset_path, 'dem/color_relief/color_relief.txt'), himg_bgr)
    himg_gray_resized = cv2.resize(himg_gray, (dst_img_shape[1], dst_img_shape[0]))
    cv2.imwrite(dest_himg_fname, himg_gray_resized)
    return True


def create_heightmap_dsm(dataset_path, dst_img_shape, dest_himg_fname):
    src_himg_fname = os.path.join(dataset_path, 'dem/dsm.tif')
    if not os.path.isfile(src_himg_fname):
        logging.error('File {} does not exist'.format(src_himg_fname))
        return False
    # Param '-scale' without parameters rescales the value's range from min/max to 0/255
    gdal.Translate(dest_himg_fname, src_himg_fname,
                   options="-outsize {} {} -ot Byte -scale -r bilinear".
                   format(int(dst_img_shape[1]), int(dst_img_shape[0])))
    return True


def create_heightmap(dataset_path, dst_img_shape, dest_himg_fname):
    logging.info('Mapping heightmap image into {}'.format(dest_himg_fname))

    # Using DSM(even u8c1) provides better smoothed results rather using colored height-map
    if create_heightmap_dsm(dataset_path, dst_img_shape, dest_himg_fname):
        return True
    logging.info('There are no dsm-file. Try to operate with colored depth map')
    return create_heightmap_color(dataset_path, dst_img_shape, dest_himg_fname)


def build_from_project(dataset_path, dst_mppx, dest_img_fname, dest_himg_fname):
    is_success, bgr_recreated = create_orthophoto(dataset_path, dst_mppx, dest_img_fname)
    if not is_success:
        return False

    if dest_himg_fname is not None:
        if not os.path.isfile(dest_himg_fname) or bgr_recreated:
            dst_img_shape, _ = get_raster_info(dest_img_fname)
            if dst_img_shape is None:
                logging.error('Cannot create heightmap because file {} does not have raster info'.
                              format(dest_img_fname))
                return False

            is_success = create_heightmap(dataset_path, dst_img_shape, dest_himg_fname)
            if not is_success:
                return False

    return True


def prepare_dataset(rootdir, destdir, dst_mppx, data_subset, img_fnames=None):
    logging.info('Prepare dataset...')

    # Create destination folders
    dest_img_folder = os.path.join(destdir, 'imgs')
    dest_himg_folder = os.path.join(destdir, 'himgs')
    dest_mask_folder = os.path.join(destdir, 'masks.{}'.format(data_subset))
    if not os.path.exists(destdir):
        os.makedirs(destdir)
    if not os.path.exists(dest_img_folder):
        os.makedirs(dest_img_folder)
    if not os.path.exists(dest_himg_folder):
        os.makedirs(dest_himg_folder)
    if not os.path.exists(dest_mask_folder):
        os.makedirs(dest_mask_folder)

    # Collect customers
    customers = []
    filenames = os.listdir(rootdir)  # get all files' and folders'
    for filename in filenames:  # loop through all the files and folders
        if os.path.isdir(os.path.join(rootdir, filename)):  # check whether the current object is a folder or not
            customers.append(filename)

    # Iterate customers
    img_fname_list = []
    dataset_strings_collection = []
    for customer in customers:
        customer_folder = os.path.join(rootdir, customer)
        # Collect datasets
        filenames = os.listdir(customer_folder)  # get all files' and folders'
        datasets = []
        for filename in filenames:  # loop through all the files and folders
            if os.path.isdir(os.path.join(customer_folder, filename)):
                datasets.append(filename)

        for dataset in datasets:
            dataset_path = os.path.join(customer_folder, dataset)
            uniq_fname = customer + '_' + dataset  # ATTENTION: do not use DOTS '.' in filename
            uniq_fname = uniq_fname.replace('.', '_')

            if img_fnames is not None:
                if uniq_fname + '.png' not in img_fnames:
                    continue

            logging.info('Iterate dataset {}'.format(dataset_path))
            #
            dest_img_fname = os.path.join(dest_img_folder, uniq_fname + '.png')
            dest_himg_fname = os.path.join(dest_himg_folder, uniq_fname + '.png')

            is_success = build_from_project(dataset_path, dst_mppx, dest_img_fname, dest_himg_fname)
            if not is_success:
                continue

            contour_fname = os.path.join(dataset_path, 'orthophoto/user_muckpile.json')
            if os.path.isfile(contour_fname):
                gdalinfo = os.path.join(dataset_path, 'orthophoto/tiles/gdalinfo.txt')
                if not os.path.isfile(gdalinfo):
                    logging.error('File {} does not exist'.format(gdalinfo))
                    continue
                mapdata_fname = os.path.join(dataset_path, 'orthophoto/tiles/mapdata.json')
                if not os.path.isfile(mapdata_fname):
                    logging.error('File {} does not exist'.format(mapdata_fname))
                    continue
                # Get mapdata
                with open(mapdata_fname, 'r') as f:
                    mapdata = json.load(f)
                x0_m, y0_m, x1_m, y1_m = get_geoloc_gata(gdalinfo, mapdata)

                # Get last modification date of contours
                mod_timesince_epoc = os.path.getmtime(contour_fname)
                # Convert seconds since epoch to readable timestamp
                mod_time = time.strftime(DATETIME_FORMAT, time.localtime(mod_timesince_epoc))
                with open(contour_fname) as f:
                    json_data = json.load(f)
                    json_contours = json_data['contours']
                    dst_img_shape, dst_f_mppx = get_raster_info(dest_img_fname)
                    if dst_img_shape is None:
                        logging.error('File {} does not have raster info'.format(dest_img_fname))
                        continue

                    mask = np.zeros(shape=(dst_img_shape[0], dst_img_shape[1], 1), dtype=np.uint8)

                    for json_contour in json_contours:
                        pts_wmerc = json_contour['pts_m']
                        pts_m = map_epsg2loc(mapdata, np.array(pts_wmerc), 3857)
                        pts_px = map_contour_meter_pix(dst_img_shape, [pts_m], x0_m, y0_m, x1_m, y1_m)
                        pts_px = np.asarray(pts_px)
                        cv2.fillPoly(mask, [pts_px], color=(255,))

                    dest_mask_fname = os.path.join(dest_mask_folder, uniq_fname + '.png')
                    cv2.imwrite(dest_mask_fname, mask)
                    #
                    dataset_strings_collection.append(customer + '/' + dataset + ',' + mod_time)
            img_fname_list.append('../imgs/{}'.format(os.path.basename(dest_img_fname)))  # todo: /imgs ?
    with open(os.path.join(destdir, 'dataset_list.txt'), 'w') as f:
        for item in dataset_strings_collection:
            f.write("%s\n" % item)
    with open(os.path.join(dest_mask_folder, "image_list.txt"), 'w') as f:
        for item in img_fname_list:
            f.write("%s\n" % item)
