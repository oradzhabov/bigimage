import os
from kutils import PrepareData
from datetime import datetime
from kutils import VIAConverter
from kmodel import config

if __name__ == "__main__":
    rootdir = 'F:/DATASET/Strayos/MuckPileDatasets.outputs'
    destdir = 'F:/DATASET/Strayos/MuckPileDatasets.outputs.Result.4'

    mppx = 0.1
    # destdir = os.path.join(destdir, datetime.now().strftime(PrepareData.DATETIME_FORMAT), 'mppx{:.2f}'.format(mppx))
    destdir = os.path.join(destdir, '2020-05-24a', 'mppx{:.2f}'.format(mppx))

    PrepareData.prepare_dataset(rootdir, destdir, mppx, config.cfg.data_subset)

    # Find json-file in mask-subdir
    via_annotation_file = None
    maskdir = os.path.join(destdir, 'masks.{}'.format(config.cfg.data_subset))
    for file in os.listdir(maskdir):
        if os.path.splitext(file)[1].lower() == '.json':
            via_annotation_file = os.path.join(maskdir, file)
            break
    VIAConverter.convert(via_annotation_file, 0.05)
