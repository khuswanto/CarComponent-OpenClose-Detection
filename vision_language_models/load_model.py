import torch

from transformers import AutoProcessor
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import BitsAndBytesConfig, Idefics3ForConditionalGeneration

# HuggingFaceTB/SmolVLM-Base, SmolVLM-500M-Base, SmolVLM-500M-Instruct, SmolVLM-256M-Base, SmolVLM-256M-Instruct
model_id = "HuggingFaceTB/SmolVLM-500M-Base"
device = "cuda" if torch.cuda.is_available() else "cpu"


processor = AutoProcessor.from_pretrained(model_id)
image_token_id = processor.tokenizer.additional_special_tokens_ids[
    processor.tokenizer.additional_special_tokens.index("<image>")]


def load_pretrained_model(lora_training: bool = False, qlora_training: bool = True):
    if lora_training or qlora_training:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules=['down_proj', 'o_proj', 'k_proj', 'q_proj', 'gate_proj', 'up_proj', 'v_proj'],
            use_dora=not qlora_training,
            init_lora_weights="gaussian"
        )
        lora_config.inference_mode = False
        model = Idefics3ForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            ) if qlora_training else None,
            _attn_implementation="flash_attention_2",
            device_map="auto"
        )
        model.add_adapter(lora_config)
        model.enable_adapters()
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        print(model.get_nb_trainable_parameters())

    else:
        model = Idefics3ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            _attn_implementation="flash_attention_2",
        ).to(device)

        # if you'd like to only fine-tune LLM
        for param in model.model.vision_model.parameters():
            param.requires_grad = False

    return model


def collate_fn(examples):
    texts = []
    images = []
    for example in examples:
        image = example["image"]
        if image.mode != 'RGB':
            image = image.convert('RGB')
        question = example["question"]
        answer = example["description"]
        messages = [
            {
                "role": "user",
                "content": [
                    # {"type": "text", "text": "Answer briefly."},
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer}
                ]
            }
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=False)
        texts.append(text.strip())
        images.append([image])

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == image_token_id] = -100
    batch["labels"] = labels

    return batch
