import pyproj
import numpy as np

def getValue(line, val_from, val_to):
    pos1 = line.find(val_from)
    if pos1 != -1:
        pos1 += len(val_from)
        pos2 = line.find(val_to, pos1)

    if pos1 == -1 or pos2 == -1:
        raise Exception("Not Found value: {}".format())

    return line[pos1 : pos2]

def mapLoc2EPSG(mapdata, pts_locCS, destEPSG):
    projstring = mapdata['PROJ4']
    lat_0 = getValue(projstring, '["latitude_of_origin",', ']')
    lon_0 = getValue(projstring, '["central_meridian",', ']')
    k = getValue(projstring, '["scale_factor",', ']')
    x_0 = getValue(projstring, '["false_easting",', ']')
    y_0 = getValue(projstring, '["false_northing",', ']')

    pr = "+proj=tmerc +lat_0="+str(lat_0)+" +lon_0="+str(lon_0)+" +k="+str(k)+" +x_0="+str(x_0)+" +y_0="+str(y_0)+" +datum=WGS84 +units=m +no_defs"
    fx,fy = pyproj.transform(pyproj.Proj(pr), pyproj.Proj(init='epsg:'+str(destEPSG)), pts_locCS[:,0],pts_locCS[:,1])
    result = np.dstack([fx,fy])[0]
    return result.tolist()

def mapEPSG2Loc(mapdata, pts_wmerc, srcEPSG):
    projstring = mapdata['PROJ4']
    lat_0 = getValue(projstring, '["latitude_of_origin",', ']')
    lon_0 = getValue(projstring, '["central_meridian",', ']')
    k = getValue(projstring, '["scale_factor",', ']')
    x_0 = getValue(projstring, '["false_easting",', ']')
    y_0 = getValue(projstring, '["false_northing",', ']')

    pr = "+proj=tmerc +lat_0="+str(lat_0)+" +lon_0="+str(lon_0)+" +k="+str(k)+" +x_0="+str(x_0)+" +y_0="+str(y_0)+" +datum=WGS84 +units=m +no_defs"
    fx,fy = pyproj.transform(pyproj.Proj(init='epsg:'+str(srcEPSG)), pyproj.Proj(pr), pts_wmerc[:,0],pts_wmerc[:,1])
    result = np.dstack([fx,fy])[0]
    return result.tolist()

if __name__ == "__main__":
    import json
    p = np.array([[-8.439429058249516,-12.04654554463923],[1,1],[2,2]])
    with open('mapdata.json', 'r') as f:
        mapdata = json.load(f)
    r = mapLoc2EPSG(mapdata, p, 3857)
    print(r)