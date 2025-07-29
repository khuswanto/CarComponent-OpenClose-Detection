import os
import json

from pathlib import Path
from PIL import Image
from datasets import Dataset

THIS_PATH = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = THIS_PATH / "data"
MAIN_DATA_PATH = THIS_PATH / ".." / "data"

with open(DATA_PATH / 'annotation.json') as f:
    annotation = json.load(f)


def construct_dataset(folder_name, dir_path):
    data = []
    n = len(annotation[folder_name])
    for i, fname in enumerate(os.listdir(dir_path)):
        image = Image.open(dir_path / fname).convert("RGB")
        data.append({
            'image': image,
            'question': 'Which door is open and closed?',
            'description': annotation[folder_name][i % n]
        })
    return data


def create_dataset(use_all: bool = False) -> Dataset:
    data = []
    if use_all:
        for variant in ['dark-224', 'white-224']:
            for folder_name in os.listdir(MAIN_DATA_PATH / variant):
                dir_path = MAIN_DATA_PATH / variant / folder_name
                if not os.path.isfile(dir_path):
                    data.extend(construct_dataset(folder_name, dir_path))

    else:
        for folder_name in os.listdir(DATA_PATH):
            dir_path = DATA_PATH / folder_name
            if not os.path.isfile(dir_path):
                data.extend(construct_dataset(folder_name, dir_path))

    return Dataset.from_list(data)
