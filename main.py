import os
os.environ["KERAS_BACKEND"] = "torch"

import sys
import time
import asyncio
import keras
import torchvision
import numpy as np

from io import BytesIO
from utils.car3d import Car3D
from PIL import Image
from multi_label_classification_models.resnet18 import ResNet18
from utils.dataset import DATA_PATH



async def main():
    classes = sorted(entry.name for entry in os.scandir(os.path.join(DATA_PATH, str(224))) if entry.is_dir())
    classes = sorted(list(set(label for cls_name in classes for label in cls_name.split('-'))))
    classes.pop(0)  # remove 'AllClose'
    idx2class = {i: name for i, name in enumerate(classes)}

    car3d = Car3D()
    model = keras.models.load_model(
        'models/resnet18-multi-label.keras',
        custom_objects={'ResNet18': ResNet18}
    )
    transform = torchvision.transforms.ToTensor()

    async with car3d.show_page():
        while True:
            frame_count = 0
            start_time = time.time()
            img = Image.open(BytesIO(await car3d.screenshot())).resize((224, 224)).convert("RGB")
            arr = np.array(transform(img))
            arr = arr[np.newaxis, ...]
            prediction = model.predict(arr, verbose=0)

            frame_count += 1
            elapsed = time.time() - start_time
            print(f"Open: {' | '.join(idx2class[i] for i, pred in enumerate(prediction[0]) if pred)}")
            print(f"QoS : {frame_count / elapsed:.2f} it/s")
            sys.stdout.flush()


if __name__ == '__main__':
    asyncio.run(main())
