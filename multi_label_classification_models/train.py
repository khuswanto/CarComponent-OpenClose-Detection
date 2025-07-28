import os
os.environ["KERAS_BACKEND"] = "torch"

import json
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping

from multi_label_classification_models.car_dataset import CarDataset
from multi_label_classification_models.hamming_loss import HammingLoss
from multi_label_classification_models.torch_tensorboard import TorchTensorBoard

THIS_PATH = Path(os.path.abspath(os.path.dirname(__file__)))


if __name__ == '__main__':
    dataset = CarDataset(variants=('white-224', 'dark-224'), use_case='multi-label')
    batch_size = 64
    proportions = [.64, .16, .20]

    # split by turn
    train_idx, val_idx, test_idx = [], [], []
    all_subset = [train_idx, val_idx, test_idx]
    i_sub = 0
    for i in range(len(dataset)):
        j = i_sub % 3
        if len(all_subset[j]) / (i + 1) < proportions[j]:
            all_subset[i_sub % 3].append(i)
        else:
            i_sub += 1

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    print(f"Number of images: {len(dataset)}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Number of data per subset: {len(train_dataset)} | {len(val_dataset)} | {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    model_name = "cnn-2"
    model_dir = THIS_PATH / model_name / 'models'
    try:
        model = load_model(model_dir / f'{model_name}.keras')
        last_epoch = 286  # manual set
        print(f"Continue training from {last_epoch} epoch")
    except FileNotFoundError:
        last_epoch = 0
        print("Model not found, creating new model")
        from multi_label_classification_models.cnn import create_model

        model = create_model(small=False)
        model.build(input_shape=(None, 3, 224, 224))
        model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=[
                'binary_accuracy',
                # HammingLoss()
            ],
        )
    model.summary()

    # train
    model.fit(
        train_loader,
        validation_data=val_dataloader,
        batch_size=batch_size,
        initial_epoch=last_epoch,
        epochs=1000,
        callbacks=[
            TorchTensorBoard(THIS_PATH / model_name / 'logs', write_images=True),
            ModelCheckpoint(
                THIS_PATH / model_name / 'ckpt' / f'{model_name}.keras',
                save_best_only=True,
                save_weights_only=False
            ),
            EarlyStopping(patience=20)
        ]
    )
    os.makedirs(model_dir, exist_ok=True)
    model.save(model_dir / f'{model_name}.keras')

    # evaluation
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    score = model.evaluate(test_loader, batch_size=batch_size, return_dict=True)
    print('Score:', json.dumps(score, indent=4))
