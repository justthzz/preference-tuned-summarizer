
# Preference-Tuned Summarizer using DPO and GRPO

This project implements a summarization system fine-tuned using Direct Preference Optimization (DPO) and Gaussian Reward Preference Optimization (GRPO) on a small language model (DistilGPT2). Built using Hugging Face's `trl` library, it demonstrates how preference-based learning techniques can guide models to generate summaries that better align with human preferences.

## Project Structure

```
preference-tuned-summarizer/
├── configs/
│   └── dpo_config.json              # Hyperparameters and settings for DPO fine-tuning
│   └── grpo_config.json             # Hyperparameters and settings for GRPO fine-tuning
├── data/
│   └── dpo_format.json              # Dataset formatted with preference pairs (`prompt`, `chosen`, and `rejected`)
├── notebooks/
│   ├── data_preparation.ipynb       # Dataset preprocessing and preference formatting
│   ├── train_dpo.ipynb              # Fine-tuning using DPO
│   └── train_grpo.ipynb             # Fine-tuning using GRPO
├── models/
│   ├── distilgpt2-dpo-checkpoint/   # DPO fine-tuned model artifacts
│   └── distilgpt2-grpo-checkpoint/  # GRPO fine-tuned model artifacts
├── outputs/
│   └── evaluation_results.json      # Evaluation scores and sample generations comparing models
├── scripts/
│   └── run_evaluation.py            # Evaluation on multiple examples
```

## Model Overview

- **Base Model**: `distilgpt2` (Hugging Face)
- **Fine-Tuning Methods**: Direct Preference Optimization (DPO) and Gaussian Reward Preference Optimization (GRPO)
- **Objective**: Leverage pairwise preference data and Gaussian-based reward shaping to improve summary quality beyond standard likelihood training.

## Dataset

A preference-formatted summarization dataset in Hugging Face format. Each sample contains:

- `prompt`: Input text to summarize
- `chosen`: Preferred summary (positive example)
- `rejected`: Less-preferred summary (negative example)

## Training Details

- **Frameworks**: `transformers`, `trl` (`DPOTrainer`, `GRPOTrainer`), `accelerate`
- **Batch Size**: 4
- **Max Sequence Length**: 512 tokens
- **Optimizer**: AdamW
- **Mixed precision enabled** (FP16) for efficient training

## Evaluation Results

Metrics comparing the baseline (untrained), DPO, and GRPO fine-tuned models:

| Metric   | Base Summary (avg) | DPO Summary (avg) | GRPO Summary (avg) |
|----------|--------------------|-------------------|--------------------|
| ROUGE-1  | 0.0442             | 0.2841            | 0.3157             |
| ROUGE-L  | 0.0366             | 0.2247            | 0.2501             |
| BLEU     | 0.0000             | 0.0286            | 0.0342             |

These results demonstrate consistent improvements with preference-based fine-tuning, with GRPO providing an additional performance boost through Gaussian reward modeling.

## Usage Example

```python
from transformers import pipeline

pipe = pipeline("text-generation", model="justthzz/preference-tuned-summarizer")
result = pipe("Summarize: The International Criminal Court granted full membership to Palestine, expanding their ability to challenge war crimes...")
print(result[0]["generated_text"])
```

## Applications

- News article summarization with improved human preference alignment
- Domain-specific summarizers (e.g., legal, medical) incorporating preference feedback
- Research and experimentation in preference learning and reward optimization techniques

---

**Author**  
Created by Thanuja Liyanage as part of an AI internship showcase project.
