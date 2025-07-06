from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer
import os

# Load tokenizer and base model (DistilGPT2 is small and fast)
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Ensure tokenizer handles padding
tokenizer.pad_token = tokenizer.eos_token

# Load your DPO dataset
dataset = load_dataset("json", data_files="../data/dpo_format.json", split="train")

# Define training arguments
training_args = TrainingArguments(
    output_dir="../models/distilgpt2-dpo-checkpoint",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=5e-5,
    report_to="none",
    bf16=False,  # change to True if using bf16-compatible GPU
    fp16=True,
)

# Initialize DPOTrainer
trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

# Train
trainer.train()

# Save final model
trainer.save_model("../models/distilgpt2-dpo-checkpoint")
