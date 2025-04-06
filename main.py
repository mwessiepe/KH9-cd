import os
import argparse

from src.train import train
from src.test import test

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "test"], help="Choose whether to train or test the model.")
    args = parser.parse_args()

    # Common settings
    old_images_dir = 'C:/masterarbeit/raster/KH-9/amsterdam/georeferenced'
    new_images_dir = 'C:/masterarbeit/raster/aerial_images_2023'
    bag_buildings_dir = 'C:/masterarbeit/vector/buildings'

    experiment_name = 'first_test'
    experiment_dir = os.path.join('C:/masterarbeit/code/results', experiment_name)
    log_dir = 'C:/masterarbeit/code/logs'
    checkpoint_name = 'last'

    # Parameters shared by train and test
    config = {
        "old_images_dir": old_images_dir,
        "new_images_dir": new_images_dir,
        "bag_buildings_dir": bag_buildings_dir,
        "experiment_name": experiment_name,
        "experiment_dir": experiment_dir,
        "log_dir": log_dir,
        "batch_size": 8,
        "patch_size": 256,
        "learning_rate": 0.0001,
        "num_dataloader_workers": 4,
        "val_split_pct": 0.1,
        "checkpoint_name": checkpoint_name,
    }

    if args.mode == "train":
        train(model='unet', backbone='resnet18', **config)
    elif args.mode == "test":
        test(**config)
