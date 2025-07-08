# Preference-Tuned Summarizer using DPO

This project implements a summarization system fine-tuned using **Direct Preference Optimization (DPO)** on a small language model (DistilGPT2). Built using Hugging Face's `trl` library, it demonstrates how preference-based learning can guide a model to generate summaries closer to human preferences.

---

## Project Overview

- **Model**: DistilGPT2
- **Tuning Method**: Direct Preference Optimization (DPO)
- **Dataset**: Hugging Face preference-formatted summarization dataset
- **Evaluation**: ROUGE-1, ROUGE-L, BLEU
- **Libraries**: HuggingFace Transformers, TRL, Evaluate, Datasets

---

## Training Configuration

```json
{
  "output_dir": "./distilgpt2-dpo-checkpoint",
  "per_device_train_batch_size": 4,
  "learning_rate": 5e-5,
  "num_train_epochs": 3,
  "logging_dir": "./logs",
  "save_strategy": "epoch",
  "save_total_limit": 1,
  "bf16": false,
  "fp16": true,
  "remove_unused_columns": false,
  "report_to": "none",
  "padding_value": 50256
}
```

---

## Evaluation Results

| Metric      | **Base Summary (avg)** | **DPO Summary (avg)** |
| ----------- | ---------------------- | --------------------- |
| **ROUGE-1** | 0.0442                 | **0.2841**            |
| **ROUGE-L** | 0.0366                 | **0.2247**            |
| **BLEU**    | 0.0000                 | **0.0286**            |

> The DPO-tuned model shows improvement in ROUGE metrics indicating better alignment with reference summaries.

---

## Inference

Use `pipeline("text-generation", model=...)` to generate summaries after training. See `scripts/run_evaluation.py`.

---

## Folder Structure

```
preference-tuned-summarizer/
│
├── configs/
│   └── dpo_config.json
│
├── data/
│   └── dataset_prepared.json
│
├── models/
│   └── distilgpt2-dpo-checkpoint/
│
├── outputs/
│   └── evaluation_results.json
│
├── scripts/
│   ├── train_dpo.py
│   └── run_evaluation.py
│
└── README.md
```

---

## Credits

- Hugging Face `trl` (https://github.com/huggingface/trl)
- Hugging Face Datasets, Transformers, and Evaluate libraries

---

## Author

Thanuja Liyanage | [GitHub](https://github.com/justthzz) | [LinkedIn](https://linkedin.com/in/thanujaliyanage)
