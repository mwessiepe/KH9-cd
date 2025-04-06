import torch
import matplotlib.pyplot as plt
from torchgeo.datasets import RasterDataset, VectorDataset

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
    filename_glob = '*clip.tif'
    is_image = True

    def plot(self, sample):
        image = sample['image'].permute(1, 2, 0)
        image = torch.clamp(image/400, min=0, max=1).numpy()
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='Reds')
        return fig

class BagBuildings(VectorDataset):
    filename_glob = 'classified_buildings.gpkg'
    is_image = False