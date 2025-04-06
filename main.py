import os

from src.train import train

if __name__ == "__main__":

    experiment_name = 'first_test'
    experiment_dir = os.path.join('C:/masterarbeit/results', experiment_name)

    train(
        experiment_name = experiment_name,
        experiment_dir = experiment_dir,
        batch_size = 8,
        patch_size = 256,
        learning_rate = 0.0001,
        num_dataloader_workers = 4,
        val_split_pct = 0.1,
        checkpoint_name = 'last.ckpt'
        )
