from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from rouge_score import rouge_scorer
import evaluate
import json

# Load models
base_model = pipeline("text-generation", model="distilgpt2")
dpo_model = pipeline("text-generation", model="../models/distilgpt2-dpo-checkpoint/checkpoint-2154")

# Load some prompts for testing
dataset = load_dataset("cnn_dailymail", "3.0.0", split="test[:10]")

# Load metrics
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
bleu = evaluate.load("bleu")

results = []

for item in dataset:
    prompt = "Summarize: " + item["article"]
    reference = item["highlights"]

    base_output = base_model(prompt, max_new_tokens=80, do_sample=False)[0]['generated_text']
    dpo_output = dpo_model(prompt, max_new_tokens=80, do_sample=False)[0]['generated_text']

    # Cut out the actual summary only
    base_summary = base_output.replace(prompt, "").strip()
    dpo_summary = dpo_output.replace(prompt, "").strip()

    # ROUGE
    rouge_base = scorer.score(reference, base_summary)
    rouge_dpo = scorer.score(reference, dpo_summary)

    # BLEU
    if base_summary.strip() == "" or dpo_summary.strip() == "":
        bleu_base = 0.0
        bleu_dpo = 0.0
    else:
        bleu_base = bleu.compute(predictions=[base_summary], references=[[reference]])["bleu"]
        bleu_dpo = bleu.compute(predictions=[dpo_summary], references=[[reference]])["bleu"]


    results.append({
        "prompt": prompt[:300] + "...",
        "reference": reference,
        "base_summary": base_summary,
        "dpo_summary": dpo_summary,
        "rouge_base": rouge_base,
        "rouge_dpo": rouge_dpo,
        "bleu_base": bleu_base,
        "bleu_dpo": bleu_dpo
    })

# Save results
with open("../outputs/evaluation_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Evaluation complete! Results saved to outputs/evaluation_results.json")