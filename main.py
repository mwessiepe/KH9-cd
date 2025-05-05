import os
from functools import reduce
import argparse
import geopandas as gpd
from torchgeo.datasets.utils import BoundingBox

from src.train import train
from src.test import test
from src.predict import predict


def read_rois_from_geopackage(path, layer=None):
    """
    Reads geometries from a GeoPackage and converts each geometry into a TorchGeo BoundingBox.

    Args:
        path (str): Path to the GeoPackage file.
        layer (str, optional): Name of the layer to read. If None, reads the default layer.

    Returns:
        List[BoundingBox]: A list of BoundingBox objects corresponding to each geometry.
    """
    gdf = gpd.read_file(path, layer=layer)
    bboxes = []

    for geom in gdf.geometry:
        minx, miny, maxx, maxy = geom.bounds
        bbox = BoundingBox(minx, maxx, miny, maxy, 0, 1)  # Temporal bounds set to 0 and 1
        bboxes.append(bbox)

    return bboxes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "test", "predict"])
    parser.add_argument("task", choices=['baseline', 'ChangeStarFarSeg'], default='baseline')
    parser.add_argument("experiment_name" )
    parser.add_argument("--checkpoint_name", default=None, help="Checkpoint name to load (e.g. 'last' or a specific file name)")
    parser.add_argument("--patch_size", default=256)
    args = parser.parse_args()

    experiment_dir = os.path.join('C:/masterarbeit/early_fusion/results', args.experiment_name)

    predictions_dir = 'C:/masterarbeit/predictions'
    
    rois = read_rois_from_geopackage('C:/masterarbeit/vector/dataset_splits.gpkg')


    # Parameters shared by train and test
    config = {
        "old_images_dir": 'C:/masterarbeit/raster/KH-9/amsterdam/georeferenced/filtered',
        "new_images_dir": 'C:/masterarbeit/raster/aerial_images_2023',
        "bag_buildings_dir": 'C:/masterarbeit/vector/buildings',
        "experiment_name": args.experiment_name,
        "experiment_dir": experiment_dir,
        "log_dir": 'C:/masterarbeit/early_fusion/logs',
        "batch_size": 10,
        "patch_size": int(args.patch_size),
        "learning_rate": 0.0001,
        "num_dataloader_workers": 11,
        "val_split_pct": 0.2,
        "test_split_pct": 0.1,
        "checkpoint_name": args.checkpoint_name,
        "rois": rois,
        "aoi": None,
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
        predict(predictions_dir=os.path.join(predictions_dir, args.experiment_name),
                **config)
