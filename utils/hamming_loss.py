import os
os.environ["KERAS_BACKEND"] = "torch"

import torch
from keras.metrics import BinaryAccuracy


class HammingLoss(BinaryAccuracy):
    def __init__(self, threshold=0.5, name="hamming_loss", **kwargs):
        super().__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Apply threshold to get binary predictions
        y_pred_bin = (y_pred > self.threshold).float()

        y_true = y_true.to('cuda')

        # Calculate element-wise XOR (prediction != truth)
        xor = torch.ne(y_pred_bin, y_true).float()

        # Mean hamming loss per sample
        hamming_loss = torch.mean(xor)

        # Store the sum and count for averaging across batches
        self.total.assign_add(float(hamming_loss * y_true.size(0)))
        self.count.assign_add(y_true.size(0))

    def result(self):
        # Return the average Hamming loss across all batches
        return self.total / self.count

    def reset_states(self):
        self.total.assign(0.0)
        self.count.assign(0)
