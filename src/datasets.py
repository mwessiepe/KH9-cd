import torch
import matplotlib.pyplot as plt
from torchgeo.datasets import GeoDataset, RasterDataset, VectorDataset, IntersectionDataset

class AerialImages(RasterDataset):
    filename_glob = 'RGB*.tiff'
    is_image = True

    def plot(self, sample):
        image = sample['image'].permute(1, 2, 0)
        image = torch.clamp(image/300, min=0, max=1).numpy()
        fig, ax = plt.subplots()
        ax.imshow(image)
        return fig


class KH9Images(RasterDataset):
    filename_glob = 'output_feature_*.tif'
    # filename_glob = '*clip.tif'
    is_image = True


class BagBuildings(VectorDataset):
    filename_glob = 'classified_buildings_binary.gpkg'
    is_image = False
    
class BitemporalIntersectionDataset(GeoDataset):
    def __init__(self, dataset_old, dataset_new, vector_dataset, transforms=None):
        # We don't call super().__init__ to avoid default index assignment.
        self.transforms = transforms
        # Create individual intersections for the old and new datasets.
        self.dataset_old = IntersectionDataset(dataset_old, vector_dataset)
        self.dataset_new = IntersectionDataset(dataset_new, vector_dataset)
        
        # Set the internal index (you probably already have this)
        self._index = IntersectionDataset(dataset_old, dataset_new).index
    
    def __len__(self):
        return len(self.dataset_new)
    
    def __getitem__(self, query):
        # Fetch the old & new samples
        sample_old = self.dataset_old[query]
        sample_new = self.dataset_new[query]

        # Grab their images & mask
        old_img = sample_old["image"]               # [C, H, W] (panchromatic)
        new_img = sample_new["image"]               # [3, H, W] (RGB)
        mask    = sample_old["mask"]                # [H, W] or [1, H, W]

        # Expand the panchromatic to 3 channels, stack along time
        old_img = old_img.repeat(3, 1, 1)           # [3, H, W]
        image_pair = torch.stack([old_img, new_img], dim=0)  # [2, 3, H, W]

        mask = mask.unsqueeze(0)            # [1, 1, H, W]
        mask = mask.repeat(image_pair.size(0), 1, 1, 1)  # → [2, 1, H, W]

        # Build your output dict
        out = {
            "image": image_pair,  # shape [2, 3, H, W]
            "mask":  mask.unsqueeze(0) if mask.ndim==2 else mask,  # ensure [1,H,W]
            "bounds": query,
            **({"crs": sample_old["crs"]} if "crs" in sample_old else {})
        }

        print(self.transforms)
        # Apply transforms
        if self.transforms is not None:
            # Kornia’s AugmentationSequential with data_keys=["image","mask"]
            # expects you to pass positional args in the order of data_keys:
            # i.e. (image_tensor, mask_tensor) → returns (aug_image, aug_mask)
            aug_image, aug_mask = self.transforms(out["image"], out["mask"])
            out["image"], out["mask"] = aug_image, aug_mask

        return out


    @property
    def bounds(self):
        return self._index.bounds

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        self._index = value

    @property
    def res(self):
        return self.dataset_old.res