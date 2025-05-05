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
        # Use query (a BoundingBox) to index both the old and new datasets.
        sample_old = self.dataset_old[query]
        sample_new = self.dataset_new[query]
        
        # Retrieve images.
        old_img = sample_old["image"]
        new_img = sample_new["image"]
        # Transform panchromatic image to 3 channel to allow stacking
        old_img = old_img.repeat(3, 1, 1)  # Now shape [3, H, W]
           
        # Stack the images along a new temporal dimension (T=2).
        image_pair = torch.stack([old_img, new_img], dim=0)  # shape: [2, C, H, W]
        
        # Build output sample with the change mask and geospatial metadata.
        out_sample = {"image": image_pair, "mask": sample_old["mask"]}
        out_sample["bounds"] = query  # use query as bounds
        if "crs" in sample_old:
            out_sample["crs"] = sample_old["crs"]
        return out_sample

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