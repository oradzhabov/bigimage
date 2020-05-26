import os
import shutil
#from PIL import Image
import cv2
import sys
#append the relative location you want to import from
#sys.path.append("../keras/tools")
from color2height import color2height

#Image.MAX_IMAGE_PIXELS = 99331200000

rootdir = './'
destdir = './Result'
#

class Sample:
    subdir = ''
    imgFile = ''
    colorHillFile = ''

samples = []
# Iterate subfolders
for subdir, dirs, files in os.walk(rootdir):
    if subdir == './' or subdir == destdir:
        continue

    sample = Sample()
    sample.subdir = subdir
    for file in files:
        fileFullPath = os.path.join(subdir, file)
        if file == 'orthophoto_export.tif':
            sample.imgFile = fileFullPath
        if file == 'color_relief.tif':
            sample.colorHillFile = fileFullPath
    if len(sample.imgFile) > 0 and len(sample.colorHillFile):
        samples.append(sample)

# Create destenation folders
destImgFolder = destdir + '/imgs'
destHImgFolder = destdir + '/himgs'
if not os.path.exists(destdir):
    os.makedirs(destdir)
if not os.path.exists(destImgFolder):
    os.makedirs(destImgFolder)
if not os.path.exists(destHImgFolder):
    os.makedirs(destHImgFolder)

# Iterate samples
for sample in samples:
    subdir = sample.subdir

    print(subdir)
    #
    destImgFileName = destImgFolder + '/' + subdir+'.png'
    destHImgFileName = destHImgFolder + '/' + subdir+'.png'
    if os.path.isfile(destImgFileName) == True:
        continue
    if os.path.isfile(destHImgFileName) == True:
        continue

    img = cv2.imread(sample.imgFile)
    himgBGR = cv2.imread(sample.colorHillFile)[:,:,:3]

    if img.shape[0]*img.shape[1] < 10000*10000 and himgBGR.shape[0]*himgBGR.shape[1] < 10000*10000:
        grayImg = color2height('color_relief.txt', himgBGR)
        resized_grayImg = cv2.resize(grayImg, (img.shape[1], img.shape[0])) 
        cv2.imwrite(destHImgFileName, resized_grayImg)
        cv2.imwrite(destImgFileName, img)

    else:
        print('File {0} dropped by size'.format(subdir))
