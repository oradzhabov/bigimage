import pyproj
from osgeo import gdal
import numpy as np


def map_px_to_epsg(raster_path, pix_coords, dest_epsg):
    # Map pixel coordinates(according to image space from raster_path) to destination EPSG

    pix_coords = np.hstack([np.array(pix_coords), np.ones([len(pix_coords), 1], dtype=np.int32)])
    data_source = gdal.Open(raster_path)

    c, a, b, f, d, e = data_source.GetGeoTransform()
    m = np.array([[a, b, (a+b)/2+c],
                  [d, e, (d+e)/2+f],
                  [0, 0, 1]])
    crs_coords = m.dot(pix_coords.T).T[:, :2]

    src_proj = data_source.GetProjection()
    fx, fy = pyproj.transform(pyproj.Proj(src_proj), pyproj.Proj(init='epsg:' + str(dest_epsg)),
                              crs_coords[:, 0], crs_coords[:, 1])
    result = np.dstack([fx, fy])[0]
    return result.tolist()


def map_epsg_to_px(src_epsg, crs_coords, raster_path):
    # Map points from source EPSG to image space of raster_path

    crs_coords = np.array(crs_coords)
    data_source = gdal.Open(raster_path)

    c, a, b, f, d, e = data_source.GetGeoTransform()
    m = np.array([[a, b, (a+b)/2+c],
                  [d, e, (d+e)/2+f],
                  [0, 0, 1]])

    dest_proj = data_source.GetProjection()
    fx, fy = pyproj.transform(pyproj.Proj(init='epsg:' + str(src_epsg)), pyproj.Proj(dest_proj),
                              crs_coords[:, 0], crs_coords[:, 1])
    crs_coords = np.dstack([fx, fy])[0]

    crs_coords = np.hstack([np.array(crs_coords), np.ones([len(crs_coords), 1])])
    result = np.linalg.inv(m).dot(crs_coords.T).T[:, :2]
    result = np.rint(result).astype(np.int32)

    return result.tolist()

