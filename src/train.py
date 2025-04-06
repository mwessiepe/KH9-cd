import os
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from .task import CustomSemanticSegmentationTask
from .datamodule import KH9CdDataModule

gpu_id = 0

def train(experiment_name, experiment_dir, batch_size, patch_size, learning_rate, num_dataloader_workers, val_split_pct, checkpoint_name):
    
    os.makedirs(experiment_dir, exist_ok=True)
    torch.set_float32_matmul_precision('medium')
    datamodule = KH9CdDataModule(
        old_images_dir = 'C:/masterarbeit/raster/KH-9/amsterdam/georeferenced',
        new_images_dir = 'C:/masterarbeit/raster/aerial_images_2023',
        bag_buildings_dir = 'C:/masterarbeit/vector/buildings',
        batch_size = batch_size,
        num_workers = num_dataloader_workers,
        patch_size = patch_size,
        val_split_pct = val_split_pct
    )

    task = CustomSemanticSegmentationTask(
        model="unet",
        backbone="resnet18",
        weights=True,
        in_channels=4,
        num_classes=2,
        loss="ce",
        ignore_index=99,
        lr=learning_rate,
        patience=10
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=experiment_dir,
        save_top_k=1,
        save_last=True,
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=10,
    )

    tb_logger = TensorBoardLogger(
        save_dir="C:/masterarbeit/logs",
        name=experiment_name
    )

    trainer = Trainer(
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=[tb_logger],
        default_root_dir=experiment_dir,
        min_epochs=10,
        max_epochs=50,
        accelerator='gpu',
        devices=[gpu_id],
        precision="16-mixed"
    )

    trainer.fit(model=task, datamodule=datamodule, ckpt_path="last")