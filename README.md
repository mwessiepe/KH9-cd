
# Building Change Detection with TorchGeo

## Overview

This project detects **building changes** over time by comparing historical KH-9 satellite images with recent aerial imagery. It uses **semantic segmentation** with a **UNet** model trained using **TorchGeo** and **PyTorch Lightning**.

---

## Project Structure

```
.
├── main.py                 # Entry point for training
├── src/
│   ├── datasets.py         # Custom Raster/Vector dataset definitions
│   ├── datamodule.py       # LightningDataModule setup
│   ├── task.py             # Custom LightningModule (UNet model)
│   ├── train.py            # Training function
│   └── test.py             # Test function with metric evaluation
├── results/                # Model checkpoints
└── logs/                   # TensorBoard logs
```

---

## Dataset Description

Three main dataset classes are used:

| Name           | Type           | Description                                     |
|----------------|----------------|-------------------------------------------------|
| `KH9Images`    | `RasterDataset`| Grayscale KH-9 satellite imagery               |
| `AerialImages` | `RasterDataset`| RGB aerial imagery (modern)                    |
| `BagBuildings` | `VectorDataset`| Building polygons labeled with `change_class`  |

The `change_class` field uses the following labels:
- `0` – No change  
- `1` – Change  
- `99` – Ignore (excluded from loss and metrics)

These datasets are combined using `IntersectionDataset`.

---

## DataModule (`KH9CdDataModule`)

Defined in `src/datamodule.py`. This module:

- Loads and intersects raster/vector datasets
- Splits them using `random_grid_cell_assignment`
- Uses `RandomGeoSampler` to sample spatial patches
- Returns PyTorch `DataLoader`s

### Configured Hyperparameters
- `patch_size`: 256 × 256
- `batch_size`: 8
- `val_split_pct`: 0.1
- `test_split_pct`: 0.1
- `grid_size`: 6
- `ignore_index`: 99 (excluded during training/evaluation)

---

## Model (`CustomSemanticSegmentationTask`)

A custom subclass of `SemanticSegmentationTask` located in `src/task.py`. Key components:

- **Model**: UNet with ResNet-18 backbone
- **Input Channels**: 4 (1 grayscale + 3 RGB)
- **Loss**: CrossEntropyLoss with `ignore_index=99`
- **Classes**: 2 (change / no change)
- **Precision**: Mixed (`16-mixed`)
- **Visualization**: Adds input/mask/prediction samples to TensorBoard

---

## Training (`train.py`)

The training loop is encapsulated in a `train()` function and launched from `main.py`.

### Features:
- Auto checkpoint saving (`ModelCheckpoint`)
- Early stopping on validation loss
- Resumes from last checkpoint if available
- TensorBoard logging at `C:/masterarbeit/logs/<experiment_name>/`

### Launch example
```bash
& C:/Users/Anwender/miniforge3/envs/torchgeo/python.exe C:\masterarbeit\code\main.py train
```

---

## Testing (`test.py`)

The test script loads a model from checkpoint and computes:
- **Precision**
- **Recall**
- **F1 score**

It uses the Lightning `trainer.test()` and performs additional per-pixel sampling:

```python
y_trues.append(batch["mask"].cpu().numpy().ravel()[::500])
y_pred = model(images).argmax(dim=1).cpu().numpy().ravel()[::500]
```

### Test function
```python
test(
    experiment_dir="C:/masterarbeit/results/first_test",
    batch_size=8,
    patch_size=256,
    num_dataloader_workers=4,
    checkpoint_name="last.ckpt"
)
```

---

## Checkpointing

During training, models are saved to:
```
C:/masterarbeit/results/<experiment_name>/
```

Files:
- `last.ckpt` – Latest model
- `epoch=XX-step=YY.ckpt` – Best model by validation loss

To load a model:
```python
task = CustomSemanticSegmentationTask.load_from_checkpoint("path/to/last.ckpt")
```

---

## TensorBoard

To monitor training visually:

```bash
tensorboard --logdir C:/masterarbeit/logs
```

Logged:
- Loss curves
- Visual patches from input, masks, predictions

---
