from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset
from rouge_score import rouge_scorer
import evaluate
import torch
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and models
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
dpo_model = pipeline("text-generation", model="../models/distilgpt2-dpo-checkpoint/checkpoint-2154")
grpo_model = pipeline("text-generation", model="../models/distilgpt2-grpo-checkpoint/checkpoint-1000")

# Load test samples
dataset = load_dataset("cnn_dailymail", "3.0.0", split="test[:10]")

# Metrics
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
bleu = evaluate.load("bleu")

results = []

def generate_summary(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        do_sample=True
    )
    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated.replace(prompt, "").strip()

for item in dataset:
    prompt = "Summarize: " + item["article"]
    reference = item["highlights"]

    base_summary = generate_summary(prompt, base_model, tokenizer)
    dpo_summary = generate_summary(prompt, base_model, tokenizer)
    grpo_summary = generate_summary(prompt, base_model, tokenizer)

    # ROUGE
    rouge_base = scorer.score(reference, base_summary)
    rouge_dpo = scorer.score(reference, dpo_summary)
    rouge_grpo = scorer.score(reference, grpo_summary)

    # BLEU
    bleu_base = bleu.compute(predictions=[base_summary], references=[[reference]])["bleu"] if base_summary else 0.0
    bleu_dpo = bleu.compute(predictions=[dpo_summary], references=[[reference]])["bleu"] if dpo_summary else 0.0
    bleu_grpo = bleu.compute(predictions=[grpo_summary], references=[[reference]])["bleu"] if grpo_summary else 0.0

    results.append({
        "prompt": prompt[:300] + "...",
        "reference": reference,
        "base_summary": base_summary,
        "dpo_summary": dpo_summary,
        "grpo_summary": grpo_summary,
        "rouge_base": rouge_base,
        "rouge_dpo": rouge_dpo,
        "rouge_grpo": rouge_grpo,
        "bleu_base": bleu_base,
        "bleu_dpo": bleu_dpo,
        "bleu_grpo": bleu_grpo
    })

# Save results
with open("../outputs/evaluation_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Evaluation complete! Results saved to `outputs/evaluation_results_grpo.json`")
