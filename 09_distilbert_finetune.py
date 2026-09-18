"""
DistilBERT Economic Grievance Classifier — v4 (3-class: collapse label 3 → 2)
Labels: 0=none, 1=mild, 2=moderate/strong
Run on your local machine:
    python train_distilbert_v4_3class.py
"""

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    f1_score,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight
import warnings, os, json
warnings.filterwarnings("ignore")

# ── Config ───────────────────────────────────────────────────────────────────
DATA_PATH  = "posts_for_training_v3_clean.csv"
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "./distilbert_eco_grievance_v4"
MAX_LEN    = 256
BATCH_SIZE = 16
EPOCHS     = 5
LR         = 2e-5
WARMUP_FRAC = 0.1
SEED       = 42
NUM_LABELS = 3

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Load & remap labels ───────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, dtype=str)
df["label"] = pd.to_numeric(df["economic_grievance_label"], errors="coerce")
df = df.dropna(subset=["label", "content"]).copy()
df["label"] = df["label"].astype(int)
df["content"] = df["content"].fillna("").astype(str)

# Collapse: 3 → 2
df["label"] = df["label"].replace({3: 2})

print(f"Loaded {len(df)} rows")
print("Label distribution (after collapse):")
print(df["label"].value_counts().sort_index())
# Expected: 0=1357, 1=798, 2=342 (315+27)

# ── Stratified train/val split ────────────────────────────────────────────────
train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=SEED, stratify=df["label"]
)
print(f"\nTrain: {len(train_df)}  |  Val: {len(val_df)}")
print("Train label dist:", train_df["label"].value_counts().sort_index().to_dict())
print("Val   label dist:", val_df["label"].value_counts().sort_index().to_dict())

# ── Class weights ─────────────────────────────────────────────────────────────
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_LABELS),
    y=train_df["label"].values,
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
print(f"\nClass weights: {class_weights.cpu().numpy().round(3)}")

# ── Tokenizer & Dataset ───────────────────────────────────────────────────────
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

class GrievanceDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }

train_ds = GrievanceDataset(train_df["content"], train_df["label"], tokenizer, MAX_LEN)
val_ds   = GrievanceDataset(val_df["content"],   val_df["label"],   tokenizer, MAX_LEN)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

# ── Model ─────────────────────────────────────────────────────────────────────
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=NUM_LABELS
).to(device)

loss_fn   = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

total_steps  = len(train_loader) * EPOCHS
warmup_steps = int(total_steps * WARMUP_FRAC)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)

# ── Eval function ─────────────────────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels    = batch["labels"].to(device)
            outputs   = model(input_ids=input_ids, attention_mask=attn_mask)
            loss      = loss_fn(outputs.logits, labels)
            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, macro_f1, all_preds, all_labels

# ── Training loop ─────────────────────────────────────────────────────────────
best_f1  = 0.0
history  = []
patience = 0
MAX_PATIENCE = 2  # early stop if no improvement for 2 epochs

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0

    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        labels    = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attn_mask)
        loss    = loss_fn(outputs.logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()

        if (step + 1) % 50 == 0:
            print(f"  Epoch {epoch} | Step {step+1}/{len(train_loader)} "
                  f"| Loss {train_loss/(step+1):.4f}")

    avg_train_loss = train_loss / len(train_loader)
    val_loss, val_f1, val_preds, val_labels = evaluate(model, val_loader)

    print(f"\nEpoch {epoch} summary:")
    print(f"  Train loss : {avg_train_loss:.4f}")
    print(f"  Val   loss : {val_loss:.4f}")
    print(f"  Val macro-F1: {val_f1:.4f}")
    print(classification_report(
        val_labels, val_preds,
        target_names=["0-none", "1-mild", "2-mod/strong"],
        zero_division=0,
    ))
    print("Confusion matrix:")
    print(confusion_matrix(val_labels, val_preds))

    history.append({
        "epoch":        epoch,
        "train_loss":   avg_train_loss,
        "val_loss":     val_loss,
        "val_macro_f1": val_f1,
    })

    if val_f1 > best_f1:
        best_f1  = val_f1
        patience = 0
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        # Save label map alongside model
        with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w") as f:
            json.dump({"0": "none", "1": "mild", "2": "moderate_or_strong"}, f)
        print(f"  ✓ Best model saved (macro-F1={best_f1:.4f})")
    else:
        patience += 1
        print(f"  No improvement ({patience}/{MAX_PATIENCE})")
        if patience >= MAX_PATIENCE:
            print("  Early stopping triggered.")
            break

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"Training complete. Best val macro-F1: {best_f1:.4f}")
print("\nTraining history:")
for h in history:
    print(f"  Epoch {h['epoch']}: train_loss={h['train_loss']:.4f} "
          f"val_loss={h['val_loss']:.4f} macro_f1={h['val_macro_f1']:.4f}")

with open(os.path.join(OUTPUT_DIR, "training_history.json"), "w") as f:
    json.dump(history, f, indent=2)

print(f"\nModel saved to: {OUTPUT_DIR}/")
print("\nTo use this model for inference:")
print("  from transformers import pipeline")
print(f"  clf = pipeline('text-classification', model='{OUTPUT_DIR}')")
print("  clf('wages are being suppressed by bankers')")