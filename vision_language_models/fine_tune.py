from vlm_dataset import create_dataset
from load_model import load_pretrained_model, collate_fn
from transformers import TrainingArguments, Trainer


if __name__ == '__main__':
    lora_training = False
    qlora_training = True
    dirname = 'SmolVLM-Instruct-qlora-30'
    train_ds = create_dataset(portion=.3)
    model = load_pretrained_model(lora_training, qlora_training)

    # Training Process
    training_args = TrainingArguments(
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=50,
        learning_rate=1e-4,
        weight_decay=0.01,
        logging_steps=25,
        save_strategy="steps",
        save_steps=250,
        save_total_limit=1,
        optim="paged_adamw_8bit",  # for 8-bit, keep this, else adamw_hf
        bf16=True,  # underlying precision for 8bit
        output_dir=f"./{dirname}",
        report_to="tensorboard",
        remove_unused_columns=False,
        gradient_checkpointing=True
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=train_ds,
    )
    trainer.train()
    trainer.save_model()
