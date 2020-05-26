import os
from kutils import PrepareData
from datetime import datetime
from kutils import VIAConverter

if __name__ == "__main__":
    rootdir = 'F:/DATASET/Strayos/MuckPileDatasets.outputs'
    destdir = 'F:/DATASET/Strayos/MuckPileDatasets.outputs.Result.4'

    mppx = 0.05
    destdir = os.path.join(destdir, datetime.now().strftime(PrepareData.DATETIME_FORMAT), 'mppx{:.2f}'.format(mppx))

    PrepareData.prepare_dataset(rootdir, destdir, mppx)

    VIAConverter.convert(os.path.join(destdir, "masks", "via_muckpile_contours.json"), 0.05)
