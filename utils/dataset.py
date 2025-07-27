import os
import torch

from typing import Literal
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

THIS_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(THIS_PATH, '..', 'data')


class CarDataset(Dataset):
    def __init__(self, variant: str = '224', use_case: Literal['multi-class', 'multi-label'] = 'multi-class'):
        self.use_case = use_case
        self.training_dir = os.path.join(DATA_PATH, variant)

        self.image_paths = []
        self.classes = sorted(entry.name for entry in os.scandir(self.training_dir) if entry.is_dir())
        if use_case == 'multi-label':
            self.classes = sorted(list(set(label for cls_name in self.classes for label in cls_name.split('-'))))
            self.classes.pop(0)  # remove 'AllClose'

        self.class2idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        for folder_name in os.listdir(self.training_dir):
            folder_path = os.path.join(self.training_dir, folder_name)
            for fname in os.listdir(folder_path):
                if fname.endswith(".png"):
                    if use_case == 'multi-class':
                        label = self.class2idx[folder_name]
                    else:
                        label = [int(cls_name in folder_name) for cls_name in self.classes]
                    self.image_paths.append((os.path.join(folder_path, fname), label))

        self.transform = transforms.ToTensor()

    def one_hot_encode(self, y):
        return torch.zeros(self.num_classes, dtype=torch.float).scatter_(0, torch.tensor(y), 1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, label = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        return (
            self.transform(image),
            self.one_hot_encode(label) if self.use_case == 'multi-class' else torch.tensor(label)
        )

    @property
    def num_classes(self) -> int:
        return len(self.classes)
