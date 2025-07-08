# Preference-Tuned Summarizer using DPO

This project implements a summarization system fine-tuned using **Direct Preference Optimization (DPO)** on a small language model (DistilGPT2). Built using Hugging Face's `trl` library, it demonstrates how preference-based learning can guide a model to generate summaries closer to human preferences.

## 📁 Project Structure

```
preference-tuned-summarizer/
├── configs/
│   └── dpo_config.json              # Configuration file containing hyperparameters and settings for DPO fine-tuning
├── data/
│   └── dpo_format.json              # Dataset formatted with preference pairs (`prompt`, `chosen`, and `rejected`)
├── notebooks/
│   ├── data_preparation.ipynb       # Dataset preprocessing and preference formatting
│   └── train_dpo.ipynb              # Fine-tuning using DPO
├── models/
│   └── distilgpt2-dpo-checkpoint/   # Final fine-tuned model artifacts
├── outputs/
│   └── evaluation_results.json      # Evaluation scores and sample generations
├── scripts/
│   ├── train_dpo.py                 # DPO training pipeline (script version)
│   └── run_evaluation.py            # Evaluation on multiple examples
```

## Model Overview

- **Base Model**: `distilgpt2` (from Hugging Face)
- **Fine-Tuning Method**: Direct Preference Optimization (DPO)
- **Objective**: Improve summary generation using pairwise preferences between chosen and rejected summaries.

## Dataset

Used a preference-formatted summarization dataset in Hugging Face format. Each sample contains:
- `prompt`: Input text to summarize
- `chosen`: Preferred summary
- `rejected`: Less-preferred summary

## Training Details

- Framework: `transformers`, `trl` (DPOTrainer), `accelerate`
- Batch Size: 4
- Max Length: 512
- Optimizer: AdamW
- Mixed precision enabled for efficiency

## Evaluation Results

The following metrics were used to compare the baseline (untrained) and the DPO-fine-tuned model:

| Metric   | Base Summary (avg) | DPO Summary (avg) |
|----------|--------------------|--------------------|
| ROUGE-1  | 0.0442             | 0.2841             |
| ROUGE-L  | 0.0366             | 0.2247             |
| BLEU     | 0.0000             | 0.0286             |

These results show clear improvements in text relevance and structure when using preference-based fine-tuning.

## How to Use the Model

```python
from transformers import pipeline

pipe = pipeline("text-generation", model="justthzz/preference-tuned-summarizer")
result = pipe("Summarize: The International Criminal Court granted full membership to Palestine, expanding their ability to challenge war crimes...")
print(result[0]["generated_text"])
```

## Applications

- News summarization with human-like quality
- Custom summarizers for specific domains (legal, medical)
- Research on preference learning and RLHF

## Author

Created by [Thanuja Liyanage](https://github.com/justthzz) as a practical showcase project.
