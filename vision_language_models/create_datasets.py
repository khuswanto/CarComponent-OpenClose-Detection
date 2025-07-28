import os
import json

from pathlib import Path
from PIL import Image
from datasets import Dataset

THIS_PATH = Path(os.path.dirname(os.path.abspath(__file__)))


def create_dataset() -> Dataset:
    data_dir = THIS_PATH / 'data'
    with open(data_dir / 'annotation.json') as f:
        annotation = json.load(f)

    data = []
    for folder_name in os.listdir(data_dir):
        dir_path = data_dir / folder_name
        if os.path.isfile(dir_path): continue

        for i, fname in enumerate(os.listdir(dir_path)):
            fpath = dir_path / fname
            image = Image.open(fpath).convert("RGB")
            data.append({
                'image': image,
                'description': annotation[folder_name][i]
            })

    return Dataset.from_list(data)
