import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchgeo.samplers import RandomGeoSampler, GridGeoSampler
from torchgeo.datasets import IntersectionDataset, stack_samples
from torchgeo.datasets import random_grid_cell_assignment
from torchgeo.datasets.utils import BoundingBox
from torchgeo.datamodules import GeoDataModule
from lightning.pytorch import LightningDataModule

from .datasets import KH9Images, AerialImages, BagBuildings


class KH9CdDataModule(LightningDataModule):
    def __init__(self, old_images_dir, new_images_dir, bag_buildings_dir, batch_size,
                 num_workers, val_split_pct, test_split_pct, patch_size, aoi: BoundingBox = None,):
        super().__init__()
        self.old_images_dir = old_images_dir
        self.new_images_dir = new_images_dir
        self.bag_buildings_dir = bag_buildings_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split_pct = val_split_pct
        self.test_split_pct = test_split_pct
        self.patch_size = patch_size
        self.aoi = aoi
        self.predict_dataset = None

    def setup(self, stage=None):
        old = KH9Images(self.old_images_dir)
        new = AerialImages(self.new_images_dir)
        bag_buildings = BagBuildings(
            paths=self.bag_buildings_dir, res=0.25, label_name="change_class")
        combined_dataset = IntersectionDataset(old, new) & bag_buildings
        fractions = [1 - self.val_split_pct - self.test_split_pct,
                     self.val_split_pct, self.test_split_pct]
        self.train_dataset, self.val_dataset, self.test_dataset = random_grid_cell_assignment(
            dataset=combined_dataset,
            fractions=fractions,
            grid_size=8,
            generator=torch.Generator().manual_seed(42)
        )

        print(f"Training set size: {len(self.train_dataset)} samples")
        print(f"Validation set size: {len(self.val_dataset)} samples")
        print(f"Test set size: {len(self.test_dataset)} samples")

        if stage == "predict":
            self.predict_dataset = IntersectionDataset(old, new)
            print(
                f"Prediction dataset size: {len(self.predict_dataset)} samples")

    def transfer_batch_to_device(self, batch, device, dataloader_idx=0):
        # Move tensor data to the specified device
        batch_on_device = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        return batch_on_device

    def train_dataloader(self):
        sampler = RandomGeoSampler(
            self.train_dataset, size=self.patch_size, length=len(self.train_dataset), roi=self.aoi)
        return DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=sampler,
                          num_workers=self.num_workers, collate_fn=stack_samples, persistent_workers=True)

    def val_dataloader(self):
        sampler = RandomGeoSampler(
            self.val_dataset, size=self.patch_size, length=len(self.val_dataset), roi=self.aoi)
        return DataLoader(self.val_dataset, batch_size=self.batch_size, sampler=sampler,
                          num_workers=self.num_workers, collate_fn=stack_samples, persistent_workers=True)

    def test_dataloader(self):
        sampler = RandomGeoSampler(
            self.test_dataset, size=self.patch_size, length=len(self.test_dataset), roi=self.aoi)
        return DataLoader(self.test_dataset, batch_size=self.batch_size, sampler=sampler,
                          num_workers=self.num_workers, collate_fn=stack_samples, persistent_workers=True)

    def predict_dataloader(self):
        sampler = GridGeoSampler(
            self.predict_dataset, size=self.patch_size, stride=self.patch_size, roi=self.aoi)
        return DataLoader(self.predict_dataset, batch_size=1, sampler=sampler,
                          num_workers=self.num_workers, collate_fn=stack_samples, persistent_workers=True)
