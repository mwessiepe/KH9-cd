import os
import argparse
import geopandas as gpd
from torchgeo.datasets.utils import BoundingBox

from src.train import train
from src.test import test


def read_aoi(path):
    gdf = gpd.read_file(path)
    minx, miny, maxx, maxy = gdf.total_bounds
    return BoundingBox(minx, maxx, miny, maxy, 0, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "test"],
                        help="Choose whether to train or test the model.")
    args = parser.parse_args()

    experiment_name = 'with_aois'
    experiment_dir = os.path.join('C:/masterarbeit/code/results', experiment_name)
    aoi = read_aoi('C:/masterarbeit/vector/aois.gpkg')

    # aoi_city_centre = BoundingBox(minx=118637.7735100609570509, maxx=123819.5269524390459992,
    #                               miny=483705.8844411586178467, maxy=488492.1146832318045199, mint=0, maxt=1)

    # Parameters shared by train and test
    config = {
        "old_images_dir": 'C:/masterarbeit/raster/KH-9/amsterdam/georeferenced',
        "new_images_dir": 'C:/masterarbeit/raster/aerial_images_2023',
        "bag_buildings_dir": 'C:/masterarbeit/vector/buildings',
        "experiment_name": experiment_name,
        "experiment_dir": experiment_dir,
        "log_dir": 'C:/masterarbeit/code/logs',
        "batch_size": 8,
        "patch_size": 256,
        "learning_rate": 0.0001,
        "num_dataloader_workers": 4,
        "val_split_pct": 0.2,
        "test_split_pct": 0.1,
        "checkpoint_name": None,
        "aoi": aoi,
    }

    if args.mode == "train":
        train(
            model='unet',
            backbone='resnet18',
            **config)
    elif args.mode == "test":
        test(**config)
