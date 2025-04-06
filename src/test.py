import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from lightning.pytorch import Trainer
from src.task import CustomSemanticSegmentationTask
from src.datamodule import KH9CdDataModule

def test(experiment_dir, batch_size, patch_size, num_dataloader_workers, checkpoint_name):
    # Initialize the data module
    datamodule = KH9CdDataModule(
        old_images_dir='C:/masterarbeit/raster/KH-9/amsterdam/georeferenced',
        new_images_dir='C:/masterarbeit/raster/aerial_images_2023',
        bag_buildings_dir='C:/masterarbeit/vector/buildings',
        batch_size=batch_size,
        num_workers=num_dataloader_workers,
        patch_size=patch_size,
        val_split_pct=0.1  # Ensure this matches your training setup
    )

    # Load the trained model from the checkpoint
    checkpoint_path = os.path.join(experiment_dir, checkpoint_name)
    task = CustomSemanticSegmentationTask.load_from_checkpoint(checkpoint_path)
    task.eval()  # Set the model to evaluation mode

    # Initialize the Trainer
    trainer = Trainer(accelerator='gpu', devices=[0])

    # Run the test set
    trainer.test(model=task, datamodule=datamodule)

    # Compute additional metrics
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = task.model.to(device).eval()

    y_preds = []
    y_trues = []

    for batch in tqdm(datamodule.test_dataloader()):
        images = batch["image"].to(device)
        y_trues.append(batch["mask"].cpu().numpy().ravel()[::500])  # Sample every 500th element
        with torch.no_grad():
            y_pred = model(images).argmax(dim=1).cpu().numpy().ravel()[::500]
        y_preds.append(y_pred)

    y_preds = np.concatenate(y_preds)
    y_trues = np.concatenate(y_trues)

    precision = precision_score(y_trues, y_preds, average='binary')
    recall = recall_score(y_trues, y_preds, average='binary')
    f1 = f1_score(y_trues, y_preds, average='binary')

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
