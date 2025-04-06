import torch
from torch.utils.data import DataLoader
from torchgeo.samplers import RandomGeoSampler
from torchgeo.datasets import IntersectionDataset, stack_samples
from torchgeo.datasets import random_grid_cell_assignment
from lightning.pytorch import LightningDataModule
from .datasets import KH9Images, AerialImages, BagBuildings

class KH9CdDataModule(LightningDataModule):
    def __init__(self, old_images_dir, new_images_dir, bag_buildings_dir, batch_size=32, num_workers=4, val_split_pct=0.2, test_split_pct=0.1, patch_size=256):
        super().__init__()
        self.old_images_dir = old_images_dir
        self.new_images_dir = new_images_dir
        self.bag_buildings_dir = bag_buildings_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split_pct = val_split_pct
        self.test_split_pct = test_split_pct
        self.patch_size = patch_size

    def setup(self, stage=None):
        old = KH9Images(self.old_images_dir)
        new = AerialImages(self.new_images_dir)
        bag_buildings = BagBuildings(paths=self.bag_buildings_dir, res=0.25, label_name="change_class")
        combined_dataset = IntersectionDataset(old, new) & bag_buildings
        fractions = [1 - self.val_split_pct - self.test_split_pct, self.val_split_pct, self.test_split_pct]
        self.train_dataset, self.val_dataset, self.test_dataset = random_grid_cell_assignment(
            dataset=combined_dataset,
            fractions=fractions,
            grid_size=6,
            generator=torch.Generator().manual_seed(42)
        )

    def transfer_batch_to_device(self, batch, device, dataloader_idx=0):
        return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}

    def train_dataloader(self):
        sampler = RandomGeoSampler(self.train_dataset, size=self.patch_size, length=len(self.train_dataset))
        return DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=sampler, num_workers=self.num_workers, collate_fn=stack_samples)

    def val_dataloader(self):
        sampler = RandomGeoSampler(self.val_dataset, size=self.patch_size, length=len(self.val_dataset))
        return DataLoader(self.val_dataset, batch_size=self.batch_size, sampler=sampler, num_workers=self.num_workers, collate_fn=stack_samples)

    def test_dataloader(self):
        sampler = RandomGeoSampler(self.test_dataset, size=self.patch_size, length=len(self.test_dataset))
        return DataLoader(self.test_dataset, batch_size=self.batch_size, sampler=sampler, num_workers=self.num_workers, collate_fn=stack_samples)
