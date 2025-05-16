import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchgeo.samplers import RandomGeoSampler, GridGeoSampler, RandomBatchGeoSampler
from torchgeo.datasets import IntersectionDataset, stack_samples
from torchgeo.datasets import roi_split
from torchgeo.datasets.utils import BoundingBox
from torchgeo.datamodules import GeoDataModule
from lightning.pytorch import LightningDataModule

from kornia.augmentation import (
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomRotation90,
    ColorJitter,
    AugmentationSequential,
)
from kornia.constants import Resample

from .datasets import KH9Images, AerialImages, BagBuildings, BitemporalIntersectionDataset


class KH9CdDataModule(LightningDataModule):
    def __init__(self, old_images_dir, new_images_dir, bag_buildings_dir, batch_size,
                 num_workers, val_split_pct, test_split_pct, patch_size, rois, aoi: BoundingBox = None,):
        super().__init__()
        self.old_images_dir = old_images_dir
        self.new_images_dir = new_images_dir
        self.bag_buildings_dir = bag_buildings_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split_pct = val_split_pct
        self.test_split_pct = test_split_pct
        self.patch_size = patch_size
        self.rois = rois
        self.aoi = aoi
        self.predict_dataset = None
        self.train_transform = AugmentationSequential(
            RandomHorizontalFlip(p=0.5, same_on_batch=True),
            RandomVerticalFlip(p=0.5, same_on_batch=True),
            RandomRotation90(times=(0,3), resample=Resample.NEAREST, 
                keepdim=False, p=0.5, same_on_batch=True),
            # ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5),
            data_keys=["image", "mask"],
        )

    def setup(self, stage=None):
        old = KH9Images(self.old_images_dir, res=1)
        new = AerialImages(self.new_images_dir, res=1)
        bag_buildings = BagBuildings(
            paths=self.bag_buildings_dir, res=1, label_name="change_class")
        # combined_dataset = IntersectionDataset(old, new) & bag_buildings
        combined_dataset = BitemporalIntersectionDataset(old, new, bag_buildings, transforms=self.train_transform)
        
        self.train_dataset, self.val_dataset, self.test_dataset = roi_split(combined_dataset, rois=self.rois)
        self.val_dataset.transforms = None
        self.test_dataset.transforms = None
        if stage == "predict":
            self.predict_dataset = BitemporalIntersectionDataset(old, new, bag_buildings)

    def transfer_batch_to_device(self, batch, device, dataloader_idx=0):
        # Move tensor data to the specified device
        batch_on_device = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        return batch_on_device

    def train_dataloader(self):
        sampler = RandomBatchGeoSampler(
            self.train_dataset, size=self.patch_size, length=2000, batch_size=self.batch_size, roi=self.aoi)
        return DataLoader(self.train_dataset, batch_sampler=sampler,
                          num_workers=self.num_workers, collate_fn=stack_samples, persistent_workers=True)

    def val_dataloader(self):
        sampler = RandomBatchGeoSampler(
            self.val_dataset, size=self.patch_size, length=500, batch_size=self.batch_size, roi=self.aoi)
        return DataLoader(self.val_dataset, batch_sampler=sampler,
                          num_workers=self.num_workers, collate_fn=stack_samples, persistent_workers=True)

    def test_dataloader(self):
        sampler = RandomBatchGeoSampler(
            self.test_dataset, size=self.patch_size, length=500, batch_size=self.batch_size)
        return DataLoader(self.test_dataset, batch_sampler=sampler,
                          num_workers=self.num_workers, collate_fn=stack_samples, persistent_workers=True)

    def predict_dataloader(self):
        sampler = GridGeoSampler(
            self.predict_dataset, size=self.patch_size, stride=self.patch_size)
        return DataLoader(self.predict_dataset, batch_size=2, sampler=sampler,
                          num_workers=self.num_workers, collate_fn=stack_samples, persistent_workers=True)
