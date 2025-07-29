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
        """Logs a weight as a TensorBoard image."""
        w_img = ops.squeeze(weight)
        shape = w_img.shape
        if len(shape) == 1:  # Bias case
            w_img = ops.reshape(w_img, [1, shape[0], 1, 1])
        elif len(shape) == 2:  # Dense layer kernel case
            if shape[0] > shape[1]:
                w_img = ops.transpose(w_img)
                shape = w_img.shape
            w_img = ops.reshape(w_img, [1, shape[0], shape[1], 1])
        elif len(shape) == 3:  # ConvNet case
            if backend.image_data_format() == "channels_last":
                # Switch to channels_first to display every kernel as a separate
                # image.
                w_img = ops.transpose(w_img, [2, 0, 1])
                shape = w_img.shape
            w_img = ops.reshape(w_img, [shape[0], shape[1], shape[2], 1])

        w_img = backend.convert_to_numpy(w_img)
        shape = w_img.shape
        # Not possible to handle 3D convnets etc.
        if len(shape) == 4 and shape[-1] in [1, 3, 4]:
            writer = self._get_writer(writer_name)
            writer.add_images(weight_name, w_img, epoch, dataformats='NHWC')

    def _log_weights(self, epoch):
        """Logs the weights of the Model to TensorBoard."""
        writer = self._get_writer('weights')
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
                        'weights_image', weight, image_weight_name, epoch
                    )

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            train_logs = {k: v for k, v in logs.items() if not k.startswith("val_")}
            val_logs = {k[4:]: v for k, v in logs.items() if k.startswith("val_")}
            self._log_epoch_metrics("train", train_logs, epoch + 1)
            self._log_epoch_metrics("val", val_logs, epoch + 1)

        self._log_weights(epoch + 1)

    def on_train_end(self, logs=None):
        for writer_name in ["train", "val"]:
            writer = self._get_writer(writer_name)
            writer.flush()
            writer.close()
