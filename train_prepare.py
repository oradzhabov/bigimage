import os
from utils import PrepareData
from datetime import datetime
from utils import VIAConverter

if __name__ == "__main__":
    rootdir = 'F:/DATASET/Strayos/MuckPileDatasets.outputs'
    destdir = 'F:/DATASET/Strayos/MuckPileDatasets.outputs.Result.4'

    destdir = os.path.join(destdir, datetime.now().strftime(PrepareData.DATETIME_FORMAT))

    PrepareData.prepare_dataset(rootdir, destdir, 0.25)

    VIAConverter.convert(os.path.join(destdir, "masks", "via_muckpile_contours.json"), 0.05)
