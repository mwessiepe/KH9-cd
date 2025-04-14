import os
from functools import reduce
import argparse
import geopandas as gpd
from torchgeo.datasets.utils import BoundingBox

from src.train import train
from src.test import test
from src.predict import predict


def read_aoi(path):
    gdf = gpd.read_file(path)
    bboxes = []

    for geom in gdf.geometry:
        minx, miny, maxx, maxy = geom.bounds
        bbox = BoundingBox(minx, maxx, miny, maxy, 0, 1)
        bboxes.append(bbox)

    union_aoi = reduce(lambda a, b: a | b, bboxes) if bboxes else None

    return union_aoi


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "test", "predict"])
    parser.add_argument("task", choices=['baseline', 'ChangeStarFarSeg'], default='baseline')
    parser.add_argument("--checkpoint_name", default=None, help="Checkpoint name to load (e.g. 'last' or a specific file name)")
    parser.add_argument("--patch_size", default=256)
    args = parser.parse_args()

    experiment_name = 'ChangeStarFarSeg_test'
    experiment_dir = os.path.join('C:/masterarbeit/early_fusion/results', experiment_name)

    # aoi = None
    aoi = read_aoi('C:/masterarbeit/vector/aois.gpkg')
    # aoi = aoi_city_centre = BoundingBox(minx=118637.7735100609570509, maxx=123819.5269524390459992,
                                #   miny=483705.8844411586178467, maxy=488492.1146832318045199, mint=0, maxt=1)
    # aoi = aoi_test_small = BoundingBox(119736.8306800332211424,120771.4748695194866741,487192.3094612115528435,487754.0150268116849475,0,1)

    # Parameters shared by train and test
    config = {
        "old_images_dir": 'C:/masterarbeit/raster/KH-9/amsterdam/georeferenced',
        "new_images_dir": 'C:/masterarbeit/raster/aerial_images_2023',
        "bag_buildings_dir": 'C:/masterarbeit/vector/buildings',
        "experiment_name": experiment_name,
        "experiment_dir": experiment_dir,
        "log_dir": 'C:/masterarbeit/early_fusion/logs',
        "batch_size": 10,
        "patch_size": args.patch_size,
        "learning_rate": 0.0001,
        "num_dataloader_workers": 11,
        "val_split_pct": 0.2,
        "test_split_pct": 0.1,
        "checkpoint_name": args.checkpoint_name,
        "aoi": aoi,
        "task": args.task,
    }

    if args.mode == "train":
        train(
            model='unet',
            backbone='resnet50',
            **config)
    elif args.mode == "test":
        test(**config)
    elif args.mode == "predict":
        predict(**config)
