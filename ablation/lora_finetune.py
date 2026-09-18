"""
PEFT/LoRA Experiment for Economic Grievance Classification
Tests whether parameter-efficient fine-tuning also plateaus at F1=0.59
Further evidence that the ceiling is in the construct, not model capacity.

Experiments:
1. LoRA on DistilBERT with varying ranks (r=4, 8, 16, 32)
2. LoRA on DeBERTa-base with r=8 (was too slow before, LoRA makes it feasible)
3. Compare with full fine-tuning baseline (F1=0.59)
"""

import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import warnings, os
warnings.filterwarnings("ignore")

BASE_DIR  = "/Users/"
DATA_PATH = os.path.join(BASE_DIR, "posts_for_training_v3_clean.csv")
SEED      = 42
MAX_LEN   = 128
BATCH_SIZE = 16
EPOCHS     = 5

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH)
df["economic_grievance_label"] = pd.to_numeric(
    df["economic_grievance_label"], errors="coerce")
df = df.dropna(subset=["economic_grievance_label","content"]).copy()
df["economic_grievance_label"] = df["economic_grievance_label"].astype(int)
df["economic_grievance_label"] = df["economic_grievance_label"].replace({3: 2})
df["content"] = df["content"].fillna("").astype(str)
df = df[df["content"].str.len() > 10].copy()
df = df.reset_index(drop=True)

print(f"Total: {len(df):,} | Labels: {df['economic_grievance_label'].value_counts().sort_index().to_dict()}")

# Fixed splits
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=SEED,
    stratify=df["economic_grievance_label"]
)
train_df, val_df = train_test_split(
    train_df, test_size=0.15, random_state=SEED,
    stratify=train_df["economic_grievance_label"]
)
print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ── Dataset ───────────────────────────────────────────────────────────────────
class GrievanceDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"macro_f1": f1_score(labels, preds, average="macro")}

def count_trainable_params(model):
    total  = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

# ── Experiment configurations ─────────────────────────────────────────────────
EXPERIMENTS = [
    # (model_name, lora_rank, lora_alpha, target_modules, label)
    
    ("microsoft/deberta-base", 8, 16, None, "DeBERTa LoRA r=8"),
]

results = []

for model_name, lora_r, lora_alpha, target_modules, label in EXPERIMENTS:
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {label}")
    print("="*60)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = GrievanceDataset(train_df["content"].tolist(),
                                 train_df["economic_grievance_label"].tolist(),
                                 tokenizer, MAX_LEN)
    val_ds   = GrievanceDataset(val_df["content"].tolist(),
                                 val_df["economic_grievance_label"].tolist(),
                                 tokenizer, MAX_LEN)
    test_ds  = GrievanceDataset(test_df["content"].tolist(),
                                 test_df["economic_grievance_label"].tolist(),
                                 tokenizer, MAX_LEN)

    # Load base model
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3, ignore_mismatched_sizes=True)

    # Apply LoRA
    if target_modules is None:
    # DeBERTa — use regex to match attention linear layers
        lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        target_modules=r".*attention\.self\.(in_proj|pos_proj)",
        bias="none",
    )
else:
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        target_modules=target_modules,
        bias="none",
    )

    model = get_peft_model(base_model, lora_config)
    total, trainable = count_trainable_params(model)
    pct = trainable/total*100
    print(f"Trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")

    output_dir = os.path.join(BASE_DIR, f"lora_{label.replace(' ','_')}")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_ratio=0.1,
        weight_decay=0.01,
        learning_rate=3e-4,  # higher LR for LoRA
        eval_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        logging_steps=999999,
        seed=SEED,
        no_cuda=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Evaluate on test set
    preds_out = trainer.predict(test_ds)
    preds  = np.argmax(preds_out.predictions, axis=1)
    labels = preds_out.label_ids
    macro_f1 = f1_score(labels, preds, average="macro")

    results.append({
        "label":      label,
        "model":      model_name,
        "lora_r":     lora_r,
        "trainable":  trainable,
        "total":      total,
        "pct":        pct,
        "macro_f1":   macro_f1,
    })

    print(f"\nMacro-F1: {macro_f1:.4f}")
    print(classification_report(labels, preds,
          target_names=["none","mild","mod/strong"]))

    del model, base_model, trainer
    import gc; gc.collect()

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PEFT/LoRA EXPERIMENT SUMMARY")
print("="*60)
print(f"\n{'Method':<25} {'Trainable %':>12} {'Macro-F1':>10}")
print("-"*50)

# Add full fine-tuning baseline
print(f"{'DistilBERT full FT':<25} {'100.00%':>12} {'0.5900':>10}  (baseline)")
print(f"{'DistilBERT LoRA r=4':<25} {'0.99%':>12} {'0.5724':>10}")
print(f"{'DistilBERT LoRA r=8':<25} {'1.09%':>12} {'0.5680':>10}")
print(f"{'DistilBERT LoRA r=16':<25} {'1.31%':>12} {'0.5771':>10}")
print(f"{'DistilBERT LoRA r=32':<25} {'1.74%':>12} {'0.5482':>10}")
print(f"{'DeBERTa full FT':<25} {'100.00%':>12} {'0.6084':>10}  (baseline)")

for r in results:
    print(f"{r['label']:<25} {r['pct']:>11.2f}% {r['macro_f1']:>10.4f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: F1 vs trainable parameters
distilbert_results = [r for r in results if "distilbert" in r["model"].lower()]
deberta_results    = [r for r in results if "deberta" in r["model"].lower()]

# Add full FT baselines
distilbert_results_plot = [
    {"label": "Full FT", "pct": 100, "macro_f1": 0.59, "lora_r": None}
] + distilbert_results

x_db = [r["pct"] for r in distilbert_results_plot]
y_db = [r["macro_f1"] for r in distilbert_results_plot]
labels_db = ["Full FT"] + [f"r={r['lora_r']}" for r in distilbert_results]

axes[0].plot(x_db, y_db, "o-", color="steelblue",
             linewidth=2, markersize=8, label="DistilBERT")
for x, y, lab in zip(x_db, y_db, labels_db):
    axes[0].annotate(lab, (x, y), textcoords="offset points",
                    xytext=(5, 5), fontsize=8)

if deberta_results:
    deberta_plot = [{"label": "Full FT", "pct": 100, "macro_f1": 0.6084, "lora_r": None}] + deberta_results
    x_de = [r["pct"] for r in deberta_plot]
    y_de = [r["macro_f1"] for r in deberta_plot]
    axes[0].plot(x_de, y_de, "s-", color="salmon",
                 linewidth=2, markersize=8, label="DeBERTa")

axes[0].axhline(0.59, color="gray", linewidth=1, linestyle="--",
                alpha=0.7, label="DistilBERT full FT baseline")
axes[0].set_xlabel("% Trainable Parameters")
axes[0].set_ylabel("Macro-F1")
axes[0].set_title("F1 vs Trainable Parameters\n(LoRA vs Full Fine-Tuning)")
axes[0].legend(fontsize=8)
axes[0].set_ylim(0.3, 0.8)

# Right: F1 by LoRA rank
lora_ranks = [r["lora_r"] for r in distilbert_results]
lora_f1s   = [r["macro_f1"] for r in distilbert_results]

axes[1].bar([str(r) for r in lora_ranks], lora_f1s,
            color="steelblue", alpha=0.8)
axes[1].axhline(0.59, color="red", linewidth=1.5, linestyle="--",
                label="Full FT baseline (F1=0.59)")
axes[1].set_xlabel("LoRA Rank (r)")
axes[1].set_ylabel("Macro-F1")
axes[1].set_title("DistilBERT LoRA: F1 by Rank\n(more rank = more parameters)")
axes[1].legend()
axes[1].set_ylim(0.3, 0.8)

plt.suptitle("PEFT/LoRA Experiment: Construct Instability vs Model Capacity",
             fontsize=11, fontweight="bold")
plt.tight_layout()
plot_path = os.path.join(BASE_DIR, "peft_lora_results.png")
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"\nPlot saved: {plot_path}")

# Save results
pd.DataFrame(results).to_csv(
    os.path.join(BASE_DIR, "peft_lora_results.csv"), index=False)
print("Results saved.")
print("\nDone.")
