import os
from pathlib import Path, WindowsPath
from PIL import Image

THIS_PATH = Path(os.path.dirname(os.path.abspath(__file__)))


dim = 224
for file in (THIS_PATH / "800").glob('**/*.png'):  # type: WindowsPath
    print(file)
    parts = list(file.parts)
    parts[-3] = str(dim)
    new_path = Path(*parts)
    os.makedirs(new_path.parent, exist_ok=True)
    im = Image.open(file)
    im.resize((dim, dim)).save(new_path)
