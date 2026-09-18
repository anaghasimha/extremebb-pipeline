"""
Fine-tune DeBERTa-v3-base on economic grievance classification
Same setup as DistilBERT v4 for direct comparison
Goal: test whether a more powerful model also plateaus at ~F1=0.59
which would support the construct instability argument
"""

import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix
)
from torch.utils.data import Dataset
import warnings, os
warnings.filterwarnings("ignore")

BASE_DIR   = "/Users/"
DATA_PATH  = os.path.join(BASE_DIR, "posts_for_training_v3_clean.csv")
MODEL_NAME = "microsoft/deberta-base"
OUTPUT_DIR = os.path.join(BASE_DIR, "deberta_eco_grievance_v1")
MAX_LEN    = 128
BATCH_SIZE = 16
EPOCHS     = 5
SEED       = 42

# ── Device ────────────────────────────────────────────────────────────────────
# Force CPU — DeBERTa-v3 has MPS compatibility issues
device = torch.device("cpu")
print("Using CPU (DeBERTa-v3 has MPS compatibility issues)")

# ── Load data ─────────────────────────────────────────────────────────────────
print(f"\nLoading: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df):,} rows")

text_col  = "content"
label_col = "economic_grievance_label"

df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
df = df.dropna(subset=[label_col, text_col]).copy()
df[label_col] = df[label_col].astype(int)
df[label_col] = df[label_col].replace({3: 2})  # collapse 3 → 2, same as DistilBERT
df[text_col]  = df[text_col].fillna("").astype(str)
df = df[df[text_col].str.len() > 10].copy()

print(f"\nLabel distribution (after collapse):")
print(df[label_col].value_counts().sort_index())
print(f"Total samples: {len(df):,}")

# ── Train/val/test split ──────────────────────────────────────────────────────
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=SEED,
    stratify=df[label_col]
)
train_df, val_df = train_test_split(
    train_df, test_size=0.15, random_state=SEED,
    stratify=train_df[label_col]
)
print(f"\nTrain: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

# ── Tokenizer ─────────────────────────────────────────────────────────────────
print(f"\nLoading tokenizer: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

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

train_ds = GrievanceDataset(
    train_df[text_col].tolist(),
    train_df[label_col].tolist(),
    tokenizer, MAX_LEN
)
val_ds = GrievanceDataset(
    val_df[text_col].tolist(),
    val_df[label_col].tolist(),
    tokenizer, MAX_LEN
)
test_ds = GrievanceDataset(
    test_df[text_col].tolist(),
    test_df[label_col].tolist(),
    tokenizer, MAX_LEN
)

# ── Model ─────────────────────────────────────────────────────────────────────
print(f"\nLoading model: {MODEL_NAME}...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    ignore_mismatched_sizes=True
)

# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "macro_f1": f1_score(labels, preds, average="macro"),
        "micro_f1": f1_score(labels, preds, average="micro"),
        "accuracy": float((preds == labels).mean()),
    }

# ── Training arguments ────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_ratio=0.1,
    weight_decay=0.01,
    learning_rate=2e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    logging_steps=50,
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
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# ── Train ─────────────────────────────────────────────────────────────────────
print("\nTraining DeBERTa-v3-base...")
print(f"Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | Max len: {MAX_LEN}")
print("Running on CPU — estimated 30-60 minutes\n")

trainer.train()

# ── Evaluate ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TEST SET EVALUATION")
print("="*60)

preds_out = trainer.predict(test_ds)
preds  = np.argmax(preds_out.predictions, axis=1)
labels = preds_out.label_ids

macro_f1 = f1_score(labels, preds, average="macro")
micro_f1 = f1_score(labels, preds, average="micro")
acc      = float((preds == labels).mean())

print(f"\nMacro-F1:  {macro_f1:.4f}")
print(f"Micro-F1:  {micro_f1:.4f}")
print(f"Accuracy:  {acc:.4f}")

print("\nClassification Report:")
print(classification_report(labels, preds,
      target_names=["none","mild","moderate/strong"]))

print("\nConfusion Matrix:")
print(confusion_matrix(labels, preds))

# ── Comparison ────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("COMPARISON: DistilBERT v4 vs DeBERTa-v3-base")
print("="*60)
print(f"DistilBERT v4 Macro-F1:    0.5900")
print(f"DeBERTa-v3-base Macro-F1:  {macro_f1:.4f}")
diff = macro_f1 - 0.59
print(f"Difference:                {diff:+.4f}")

if abs(diff) < 0.05:
    print("\nCONCLUSION: Both models plateau at similar F1.")
    print("Supports the construct instability argument.")
elif diff > 0.05:
    print("\nCONCLUSION: DeBERTa significantly outperforms DistilBERT.")
    print("Model capacity may be a contributing factor.")
else:
    print("\nCONCLUSION: DeBERTa underperforms — check training setup.")

# ── Save ──────────────────────────────────────────────────────────────────────
print(f"\nSaving model to {OUTPUT_DIR}...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Done.")
