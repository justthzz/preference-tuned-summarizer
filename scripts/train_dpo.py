from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer, DPOConfig
import os

# Load tokenizer and base model (DistilGPT2 is small and fast)
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
ref_model = AutoModelForCausalLM.from_pretrained(model_name)

# Ensure tokenizer handles padding
tokenizer.pad_token = tokenizer.eos_token

# Load your DPO dataset
dataset = load_dataset("json", data_files="/Users/thanuja/Desktop/preference-tuned-summarizer/data/dpo_format.json", split="train")

# Define training arguments (use DPOConfig instead!)
training_args = DPOConfig(
    output_dir="../models/distilgpt2-dpo-checkpoint",
    per_device_train_batch_size=4,
    learning_rate=5e-5,
    num_train_epochs=3,
    logging_dir="./logs",
    save_strategy="epoch",
    bf16=False,   # Disable bfloat16 (MPS usually does NOT support this fully yet)
    fp16=False,   # Disable fp16 (not supported on MPS)
    remove_unused_columns=False,
    report_to="none",
    padding_value=tokenizer.pad_token_id,
)

# Initialize DPOTrainer
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=dataset,
)

# Train
trainer.train()

# Save final model
trainer.save_model("../models/distilgpt2-dpo-checkpoint")
