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


def interleaved_select(lst, fraction):
    n = len(lst)
    k = round(n * fraction)
    step = n / k
    indices = [round(i * step) for i in range(k)]
    # Ensure indices are within bounds and unique
    indices = sorted(set(min(idx, n - 1) for idx in indices))
    return [lst[i] for i in indices]


def construct_dataset(folder_name, dir_path, fraction: float = None):
    data = []
    arr = os.listdir(dir_path)
    if fraction:
        # select % of the data interleave-ly
        arr = interleaved_select(arr, fraction)

    n = len(annotation[folder_name])
    for i, fname in enumerate(arr):
        image = Image.open(dir_path / fname).convert("RGB")
        data.append({
            'image': image,
            'question': 'Which door is open and closed?',
            'description': annotation[folder_name][i % n]
        })
    return data


def create_dataset(portion: str | float = 'curated') -> Dataset:
    data = []
    if portion == 'all' or isinstance(portion, float):
        for variant in ['dark-224', 'white-224']:
            for folder_name in os.listdir(MAIN_DATA_PATH / variant):
                dir_path = MAIN_DATA_PATH / variant / folder_name
                if not os.path.isfile(dir_path):
                    data.extend(construct_dataset(folder_name, dir_path, portion if isinstance(portion, float) else None))

    elif portion == 'curated':
        for folder_name in os.listdir(DATA_PATH):
            dir_path = DATA_PATH / folder_name
            if not os.path.isfile(dir_path):
                data.extend(construct_dataset(folder_name, dir_path))

    else:
        raise ValueError(f"Unknown portion: {portion}")

    return Dataset.from_list(data)
