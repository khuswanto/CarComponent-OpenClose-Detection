import os
from pathlib import Path, WindowsPath
from PIL import Image

THIS_PATH = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = THIS_PATH / '..' / 'data'


source = 'dark'
target = 'dark-224'
dim = 224
for file in (DATA_PATH / source).glob('**/*.png'):  # type: WindowsPath
    print(file)
    parts = list(file.parts)
    parts[-3] = target
    new_path = Path(*parts)
    os.makedirs(new_path.parent, exist_ok=True)
    im = Image.open(file)
    im.resize((dim, dim)).save(new_path)
