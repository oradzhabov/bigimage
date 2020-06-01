import os
import re
import json
import numpy as np
import cv2
import time
from .color2height import color2height
from .mapLoc2EPSG import mapEPSG2Loc
from osgeo import gdal, osr


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


def get_geoloc_gata(gdalinfo, mapdata):
    # Get absolute values
    with open(gdalinfo, "r") as f:
        for line in f:
            if line.startswith('Upper Left'):
                ul = [float(value.strip()) for value in re.search(r'\((.*?)\)', line).group(1).split(',')]
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
    unit = srs.GetAttrValue('unit')
    # print('GeoTIFF length units: {}'.format(unit))
    scale_to_meter = 1.0
    if unit != 'metre':
        scale_to_meter = 0.3048

    mppx = mppx * scale_to_meter

    return img_shape, mppx


def create_orthophoto(dataset_path, dst_mppx, dest_img_fname):
    recreated = False
    src_img_fname = os.path.join(dataset_path, 'orthophoto/orthophoto_export.tif')
    if not os.path.isfile(src_img_fname):
        print('ERROR: File {} does not exist'.format(src_img_fname))
        return False, recreated

    if not os.path.isfile(dest_img_fname):
        src_img_shape, src_mppx = get_raster_info(src_img_fname)
        rescale = src_mppx / dst_mppx
        gdal.Translate(dest_img_fname, src_img_fname,
                       options="-outsize {} {} -ot Byte -r bilinear".
                       format(int(src_img_shape[1] * rescale[0]), int(src_img_shape[0] * rescale[1])))
        recreated = True

    return True, recreated


def create_heightmap(dataset_path, dst_img_shape, dest_himg_fname):
    fname = os.path.join(dataset_path, 'dem/color_relief/color_relief.tif')
    if not os.path.isfile(fname):
        print('ERROR: File {0} does not exist'.format(fname))
        return False
    himg_bgr = cv2.imread(fname)[:, :, :3]
    if himg_bgr is None:
        print('ERROR: Cannot read file {0}'.format(fname))
        return False
    himg_gray = color2height(os.path.join(dataset_path, 'dem/color_relief/color_relief.txt'), himg_bgr)
    himg_gray_resized = cv2.resize(himg_gray, (dst_img_shape[1], dst_img_shape[0]))
    cv2.imwrite(dest_himg_fname, himg_gray_resized)
    return True


def build_from_project(dataset_path, dst_mppx, dest_img_fname, dest_himg_fname):
    is_success, bgr_recreated = create_orthophoto(dataset_path, dst_mppx, dest_img_fname)
    if not is_success:
        return False

    if not os.path.isfile(dest_himg_fname) or bgr_recreated:
        dst_img_shape, dst_f_mppx = get_raster_info(dest_img_fname)
        is_success = create_heightmap(dataset_path, dst_img_shape, dest_himg_fname)
        if not is_success:
            return False

    return True


def prepare_dataset(rootdir, destdir, dst_mppx, data_subset):
    print('Prepare dataset...')

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
            print('Iterate dataset {0}'.format(dataset_path))
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
                    print('ERROR: File {0} does not exist'.format(gdalinfo))
                    continue
                mapdata_fname = os.path.join(dataset_path, 'orthophoto/tiles/mapdata.json')
                if not os.path.isfile(mapdata_fname):
                    print('ERROR: File {0} does not exist'.format(mapdata_fname))
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
                    mask = np.zeros(shape=(dst_img_shape[0], dst_img_shape[1], 1), dtype=np.uint8)

                    for json_contour in json_contours:
                        pts_wmerc = json_contour['pts_m']
                        pts_m = mapEPSG2Loc(mapdata, np.array(pts_wmerc), 3857)
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
