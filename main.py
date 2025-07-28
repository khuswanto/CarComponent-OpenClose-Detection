import os
import asyncio

from time import time
from pathlib import Path
from utils.car3d import Car3D

THIS_PATH = Path(os.path.dirname(os.path.abspath(__file__)))
CLEAR_LINE = '\x1b[2K'


async def main(objective: int):
    car3d = Car3D()
    match objective:
        case 1:
            os.environ["KERAS_BACKEND"] = "torch"

            import numpy as np
            import torchvision
            from keras.models import load_model
            from multi_label_classification_models.car_dataset import CarDataset

            model_name = "cnn"
            model_dir = THIS_PATH / 'multi_label_classification_models' / model_name / 'models'
            model = load_model(model_dir / f'{model_name}.keras')
            transform = torchvision.transforms.ToTensor()
            idx2class = {i: name for i, name in enumerate(CarDataset(use_case='multi-label').classes)}

            async with car3d.show_page():
                await car3d.hide_button()
                while True:
                    threshold = 0.5
                    img = await car3d.screenshot()

                    start_time = time()
                    img = img.resize((224, 224)).convert("RGB")
                    arr = np.array(transform(img))
                    arr = arr[np.newaxis, ...]
                    prediction = model.predict(arr, verbose=0)
                    prediction = (prediction[0] >= threshold).astype(int)

                    elapsed = time() - start_time
                    print(
                        # f"{CLEAR_LINE}"
                        f"{(1 / elapsed) if elapsed else 0 :.2f} it/s - "
                        f"{' | '.join(idx2class[i] for i, pred in enumerate(prediction) if pred)}"
                    )

        case 2:
            import torch
            from transformers import AutoModelForImageTextToText
            from vision_language_models.load_model import processor

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_path = THIS_PATH / 'vision_language_models' / 'full-training'

            model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                _attn_implementation="flash_attention_2" if device == "cuda" else "eager",
            ).to(device)

            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "Which door is open and closed?"}
            ]}]
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

            async with car3d.show_page():
                await car3d.hide_button()
                while True:
                    input(f"Press <enter> to describe")
                    img = await car3d.screenshot()

                    start_time = time()
                    img = img.resize((224, 224)).convert("RGB")
                    inputs = processor(text=prompt, images=[img], return_tensors="pt")
                    inputs = inputs.to(device)

                    generated_ids = model.generate(**inputs, max_new_tokens=500)
                    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
                    elapsed = time() - start_time
                    print(f"{elapsed} s - {generated_texts[0]}")


if __name__ == '__main__':
    asyncio.run(main(2))
