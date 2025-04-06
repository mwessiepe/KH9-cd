from torchgeo.trainers import SemanticSegmentationTask
import matplotlib.pyplot as plt

from torchgeo.datasets import unbind_samples


class CustomSemanticSegmentationTask(SemanticSegmentationTask):
    
    def plot(self, sample):
        image = sample['image']  # shape: [C, H, W] or [1, H, W]
        mask = sample["mask"]
        prediction = sample["prediction"]

        # Split the image into two parts for visualization
        image1 = image[0:1]  # First channel
        image2 = image[1:4] if image.shape[0] >= 4 else image  # RGB or as many as possible

        fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(4 * 5, 5))

        # Display image1 (single channel)
        axs[0].imshow(image1.squeeze(0), cmap="gray")  # shape [1, H, W] -> [H, W]
        axs[0].axis("off")
        axs[0].set_title("Image 1")

        # Display image2 (RGB or 3-band)
        if image2.shape[0] == 3:
        # Normalize to [0, 1] if not already in that range
            if image2.max() > 1.0:
                image2 = image2 / 255.0
                # Convert to numpy array and ensure data type is float
                image2 = image2.permute(1, 2, 0).cpu().numpy().astype(float)
                axs[1].imshow(image2)
            elif image2.shape[0] == 1:
                axs[1].imshow(image2.squeeze(0), cmap="gray")
            else:
                axs[1].imshow(image2[0], cmap="gray")
        axs[1].axis("off")
        axs[1].set_title("Image 2")

        # Display ground truth mask
        axs[2].imshow(mask, cmap="gray")
        axs[2].axis("off")
        axs[2].set_title("Mask")

        # Display prediction
        axs[3].imshow(prediction, cmap="gray")
        axs[3].axis("off")
        axs[3].set_title("Prediction")

        plt.tight_layout()
        return fig


    # The only difference between this code and the same from SemanticSegmentationTask is our redirect to use our own plotting function
    def training_step(self, *args, **kwargs):
        batch = args[0]
        batch_idx = args[1]

        x = batch["image"]
        y = batch["mask"]
        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.criterion(y_hat, y)

        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.train_metrics(y_hat_hard, y)

        if batch_idx < 10:
            batch["prediction"] = y_hat_hard
            for key in ["image", "mask", "prediction"]:
                batch[key] = batch[key].cpu()
            sample = unbind_samples(batch)[0]
            fig = self.plot(sample)
            summary_writer = self.logger.experiment
            summary_writer.add_figure(
                f"image/train/{batch_idx}", fig, global_step=self.global_step
            )
            plt.close()

        return loss

    # The only difference between this code and the same from SemanticSegmentationTask is our redirect to use our own plotting function
    def validation_step(self, *args, **kwargs):
        batch = args[0]
        batch_idx = args[1]
        x = batch["image"]
        y = batch["mask"]
        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.criterion(y_hat, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.val_metrics(y_hat_hard, y)

        if batch_idx < 10:
            batch["prediction"] = y_hat_hard
            for key in ["image", "mask", "prediction"]:
                batch[key] = batch[key].cpu()
            sample = unbind_samples(batch)[0]
            fig = self.plot(sample)
            summary_writer = self.logger.experiment
            summary_writer.add_figure(
                f"image/val/{batch_idx}", fig, global_step=self.global_step
            )
            plt.close()
