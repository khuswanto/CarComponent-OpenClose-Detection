import os
os.environ["KERAS_BACKEND"] = "torch"

import json
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from keras.callbacks import ModelCheckpoint

from utils.dataset import CarDataset
from utils.torch_tensorboard import TorchTensorBoard
from utils.hamming_loss import HammingLoss
from multi_label_classification_models.resnet18 import ResNet18
from multi_label_classification_models.resnet50v2 import ResNet50V2


if __name__ == '__main__':
    dataset = CarDataset(variant='224_', use_case='multi-label')
    print("Number of images: ", len(dataset))
    print("Number of classes: ", dataset.num_classes)
    batch_size = 128
    train_dataset, test_dataset, val_dataset = random_split(dataset, [.64, .16, .20])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    model_name = "resnet50v2-multi-label"
    model = ResNet50V2(dataset.num_classes, use_case='multi-label')
    model.build(input_shape=(None, 3, 224, 224))
    model.compile(
        loss='binary_crossentropy',  # multi-class classification: categorical_crossentropy
        optimizer='adam',
        metrics=[
            # 'binary_accuracy',
            HammingLoss()
        ],  # multi-class classification: accuracy
    )
    model.summary()

    # train
    model.fit(
        train_loader,
        validation_data=val_dataloader,
        batch_size=batch_size,
        epochs=50,
        callbacks=[
            TorchTensorBoard(write_images=True),
            ModelCheckpoint(
                f'ckpt/{model_name}.weights.h5',
                save_best_only=True,
                save_weights_only=True
            )
        ]
    )
    model.save(f'models/{model_name}.keras')

    # evaluation
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    score = model.evaluate(test_loader, batch_size=batch_size, return_dict=True)
    print('Score:', json.dumps(score, indent=4))
