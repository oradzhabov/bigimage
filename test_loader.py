import os
from kmodel.data import get_data, Dataset, Dataloder
from kmodel.config import cfg
from kmodel.train import read_sample, get_training_augmentation, visualize, denormalize
import time


if __name__ == "__main__":
    # Check folder path
    if not os.path.exists(cfg.data_dir):
        print('There are no such data folder {}'.format(cfg.data_dir))
        exit(-1)

    # Get all data into test-set
    # data_dir, ids_test, _ = get_data(cfg, 0.0)
    data_dir, ids_train, ids_test = get_data(cfg, 0.33)

    data_reader = read_sample

    # Dataset for train images
    train_dataset = Dataset(data_reader, data_dir, ids_train,
                            min_mask_ratio=0.01,
                            augmentation=get_training_augmentation(cfg),
                            preprocessing=None)

    train_dataloader = Dataloder(train_dataset, batch_size=200, shuffle=True)

    t1 = time.time()
    a = train_dataloader[0]
    a = train_dataloader[1]
    a = train_dataloader[2]
    print("1 argument test spends:", time.time() - t1)
    pass

