import os
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from .task import CustomSemanticSegmentationTask
from .datamodule import KH9CdDataModule


def train(old_images_dir, new_images_dir, bag_buildings_dir, experiment_name,
          experiment_dir, log_dir, model, backbone, batch_size, patch_size,
          learning_rate, num_dataloader_workers, val_split_pct, test_split_pct,
          checkpoint_name, aoi):

    os.makedirs(experiment_dir, exist_ok=True)
    torch.set_float32_matmul_precision('medium')
    datamodule = KH9CdDataModule(
        old_images_dir=old_images_dir,
        new_images_dir=new_images_dir,
        bag_buildings_dir=bag_buildings_dir,
        batch_size=batch_size,
        num_workers=num_dataloader_workers,
        patch_size=patch_size,
        val_split_pct=val_split_pct,
        test_split_pct=test_split_pct,
        aoi=aoi,
    )

    task = CustomSemanticSegmentationTask(
        model=model,
        backbone=backbone,
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
        save_dir=log_dir,
        name=experiment_name
    )

    trainer = Trainer(
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=[tb_logger],
        default_root_dir=experiment_dir,
        min_epochs=10,
        max_epochs=50,
        accelerator='gpu',
        devices=[0],
        precision="16-mixed"
    )

    trainer.fit(model=task, datamodule=datamodule, ckpt_path=checkpoint_name)
