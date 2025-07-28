import os
import asyncio

from pathlib import Path
from utils.car3d import Car3D

THIS_PATH = Path(os.path.dirname(os.path.abspath(__file__)))
CLEAR_LINE = '\x1b[2K'


async def main(objective: int):
    car3d = Car3D()

    async with car3d.show_page():
        await car3d.hide_button()

        match objective:
            case 1:
                os.environ["KERAS_BACKEND"] = "torch"

                import time
                import numpy as np
                import torchvision

                from keras.models import load_model
                from multi_label_classification_models.car_dataset import CarDataset

                model_name = "cnn"
                model_dir = THIS_PATH / 'multi_label_classification_models' / model_name / 'models'
                model = load_model(model_dir / f'{model_name}.keras')
                transform = torchvision.transforms.ToTensor()
                idx2class = {i: name for i, name in enumerate(CarDataset(use_case='multi-label').classes)}

                while True:
                    threshold = 0.5
                    frame_count = 0
                    start_time = time.time()
                    img = await car3d.screenshot()
                    img = img.resize((224, 224)).convert("RGB")
                    arr = np.array(transform(img))
                    arr = arr[np.newaxis, ...]
                    prediction = model.predict(arr, verbose=0)
                    prediction = (prediction[0] >= threshold).astype(int)

                    frame_count += 1
                    elapsed = time.time() - start_time
                    # print(f"{CLEAR_LINE}{frame_count / elapsed:.2f} it/s - {' | '.join(idx2class[i] for i, pred in enumerate(prediction) if pred)}", end='\r', flush=True)
                    print(f"{frame_count / elapsed:.2f} it/s - {' | '.join(idx2class[i] for i, pred in enumerate(prediction) if pred)}")

            case 2:
                import torch
                from vision_language_models.load_model import processor
                from transformers import AutoModelForVision2Seq

                device = "cuda" if torch.cuda.is_available() else "cpu"

                model = AutoModelForVision2Seq.from_pretrained(
                    "HuggingFaceTB/SmolVLM-Base",
                    torch_dtype=torch.bfloat16,
                    _attn_implementation="flash_attention_2" if device == "cuda" else "eager",
                ).to(device)

                while True:
                    input(f"Press <enter> to describe")
                    img = await car3d.screenshot()
                    img = img.resize((224, 224)).convert("RGB")


if __name__ == '__main__':
    asyncio.run(main(1))
