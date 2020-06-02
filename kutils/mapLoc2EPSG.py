import pyproj
import numpy as np


def get_value(line, val_from, val_to):
    pos1 = line.find(val_from)
    pos2 = -1
    if pos1 != -1:
        pos1 += len(val_from)
        pos2 = line.find(val_to, pos1)

    if pos1 == -1 or pos2 == -1:
        raise Exception("Not Found value: {}".format(val_from))

    return line[pos1: pos2]


def map_loc2epsg(mapdata, pts_loc_cs, dest_epsg):
    projstring = mapdata['PROJ4']
    lat_0 = get_value(projstring, '["latitude_of_origin",', ']')
    lon_0 = get_value(projstring, '["central_meridian",', ']')
    k = get_value(projstring, '["scale_factor",', ']')
    x_0 = get_value(projstring, '["false_easting",', ']')
    y_0 = get_value(projstring, '["false_northing",', ']')

    pr = "+proj=tmerc +lat_0="+str(lat_0)+" +lon_0="+str(lon_0)+" +k=" + \
         str(k)+" +x_0="+str(x_0)+" +y_0="+str(y_0)+" +datum=WGS84 +units=m +no_defs"
    fx, fy = pyproj.transform(pyproj.Proj(pr),
                              pyproj.Proj(init='epsg:'+str(dest_epsg)),
                              pts_loc_cs[:, 0],
                              pts_loc_cs[:, 1])
    result = np.dstack([fx, fy])[0]
    return result.tolist()


def map_epsg2loc(mapdata, pts_wmerc, src_epsg):
    projstring = mapdata['PROJ4']
    lat_0 = get_value(projstring, '["latitude_of_origin",', ']')
    lon_0 = get_value(projstring, '["central_meridian",', ']')
    k = get_value(projstring, '["scale_factor",', ']')
    x_0 = get_value(projstring, '["false_easting",', ']')
    y_0 = get_value(projstring, '["false_northing",', ']')

    pr = "+proj=tmerc +lat_0="+str(lat_0)+" +lon_0="+str(lon_0)+" +k="+str(k)+" +x_0="+str(x_0) + \
         " +y_0="+str(y_0)+" +datum=WGS84 +units=m +no_defs"
    fx, fy = pyproj.transform(pyproj.Proj(init='epsg:'+str(src_epsg)),
                              pyproj.Proj(pr),
                              pts_wmerc[:, 0],
                              pts_wmerc[:, 1])
    result = np.dstack([fx, fy])[0]
    return result.tolist()


"""
if __name__ == "__main__":
    import json
    p = np.array([[-8.439429058249516, -12.04654554463923], [1, 1], [2, 2]])
    with open('mapdata.json', 'r') as f:
        mapdata = json.load(f)
    r = map_loc2epsg(mapdata, p, 3857)
    print(r)
"""
