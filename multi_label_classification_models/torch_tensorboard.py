import numpy as np
import torch

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from keras.callbacks import Callback
from keras.src import backend, ops


class TorchTensorBoard(Callback):
    def __init__(self, log_dir: Path, write_images: bool = False):
        super().__init__()
        self.log_dir = log_dir
        self.writers: dict = {}
        self.write_images = write_images

    def _get_writer(self, name) -> SummaryWriter:
        if name not in self.writers:
            self.writers[name] = SummaryWriter(self.log_dir / name)
        return self.writers[name]

    def _log_epoch_metrics(self, writer_name, logs, step):
        writer = self._get_writer(writer_name)
        for key, value in logs.items():
            writer.add_scalar(key, value, step)

    def _log_weight_as_image(self, writer_name, weight, weight_name, epoch):
        w_img = ops.squeeze(weight)
        # Detach, move to CPU, then convert to numpy:
        if isinstance(w_img, torch.Tensor):
            w_img = w_img.detach().cpu().numpy()

        shape = w_img.shape

        if len(shape) == 1:
            length = shape[0]
            size = int(np.ceil(np.sqrt(length)))
            img = np.zeros((size, size), dtype=w_img.dtype)
            img.flat[:length] = w_img
            img = img[np.newaxis, :, :]  # Add channel dim

        elif len(shape) == 2:
            img = w_img
            if shape[0] > shape[1]:
                img = img.T
            img = img[np.newaxis, :, :]  # Add channel dim

        elif len(shape) == 3:
            if backend.image_data_format() == "channels_last":
                img = np.transpose(w_img, (2, 0, 1))
            else:
                img = w_img
        else:
            return  # Unsupported shape for image logging

        # Convert to torch tensor for tensorboard logging
        img_tensor = torch.tensor(img, dtype=torch.float32)

        # Normalize image between 0 and 1
        img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min() + 1e-5)

        writer = self._get_writer(writer_name)
        if img_tensor.dim() == 3:
            writer.add_image(weight_name, img_tensor, epoch)
        elif img_tensor.dim() == 4:
            writer.add_images(weight_name, img_tensor, epoch)

    def _log_weights(self, writer_name, train_logs, epoch):
        """Logs the weights of the Model to TensorBoard."""
        writer = self._get_writer(writer_name)
        for layer in self.model.layers:
            for weight in layer.weights:
                weight_name = weight.name.replace(":", "_")
                # Add a suffix to prevent summary tag name collision.
                histogram_weight_name = weight_name + "/histogram"
                writer.add_histogram(histogram_weight_name, backend.convert_to_numpy(weight), epoch)
                if self.write_images:
                    # Add a suffix to prevent summary tag name
                    # collision.
                    image_weight_name = weight_name + "/image"
                    self._log_weight_as_image(
                        writer_name, weight, image_weight_name, epoch
                    )

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            train_logs = {k: v for k, v in logs.items() if not k.startswith("val_")}
            val_logs = {k[4:]: v for k, v in logs.items() if k.startswith("val_")}
            self._log_weights("train", train_logs, epoch + 1)
            self._log_epoch_metrics("train", train_logs, epoch + 1)
            self._log_epoch_metrics("val", val_logs, epoch + 1)

    def on_train_end(self, logs=None):
        for writer_name in ["train", "val"]:
            writer = self._get_writer(writer_name)
            writer.flush()
            writer.close()
