"""
classifier/evaluate.py

Evaluates the trained DistilBERT economic grievance classifier.
Runs:
1. Per-class F1, precision, recall
2. Confusion matrix
3. Error analysis (false positives, false negatives, severity errors)
4. McNemar's test comparing full fine-tuning vs LoRA
5. Saves error posts and plots


import argparse
import pandas as pd
import numpy as np
import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix
)
from statsmodels.stats.contingency_tables import mcnemar
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re, warnings, os
warnings.filterwarnings("ignore")

# ── Argument parser ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Evaluate economic grievance classifier")
parser.add_argument("--model",      type=str, required=True,
                    help="Path to trained DistilBERT model directory")
parser.add_argument("--data",       type=str, required=True,
                    help="Path to labeled training CSV")
parser.add_argument("--output_dir", type=str, default="results/",
                    help="Directory to save outputs")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--max_len",    type=int, default=128)
parser.add_argument("--seed",       type=int, default=42)
parser.add_argument("--lora_model", type=str, default=None,
                    help="Optional: path to LoRA model for McNemar comparison")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(args.data)
df["economic_grievance_label"] = pd.to_numeric(
    df["economic_grievance_label"], errors="coerce")
df = df.dropna(subset=["economic_grievance_label","content"]).copy()
df["economic_grievance_label"] = df["economic_grievance_label"].astype(int).replace({3:2})
df["content"] = df["content"].fillna("").astype(str)
df = df[df["content"].str.len() > 10].reset_index(drop=True)

_, test_df = train_test_split(
    df, test_size=0.2, random_state=args.seed,
    stratify=df["economic_grievance_label"]
)
test_df = test_df.reset_index(drop=True)
print(f"Test set: {len(test_df):,} posts")

# ── Dataset ───────────────────────────────────────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts     = texts
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
        return {k: v.squeeze() for k, v in enc.items()}

# ── Inference function ────────────────────────────────────────────────────────
def get_predictions(model_dir, texts, batch_size, max_len):
    tokenizer = DistilBertTokenizerFast.from_pretrained(
        model_dir, local_files_only=True)
    model = DistilBertForSequenceClassification.from_pretrained(
        model_dir, local_files_only=True)
    model.to(device)
    model.eval()

    ds     = TextDataset(texts, tokenizer, max_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_preds, all_probs = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out   = model(**batch)
            probs = torch.softmax(out.logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())

    return np.array(all_preds), np.array(all_probs)

# ── Run inference ─────────────────────────────────────────────────────────────
print(f"\nRunning inference with model: {args.model}")
preds, probs = get_predictions(
    args.model,
    test_df["content"].tolist(),
    args.batch_size,
    args.max_len
)
true_labels = test_df["economic_grievance_label"].values

# ── 1. Overall metrics ────────────────────────────────────────────────────────
print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
print(classification_report(true_labels, preds,
      target_names=["none","mild","moderate/strong"]))
print(f"Macro-F1: {f1_score(true_labels, preds, average='macro'):.4f}")

# ── 2. Confusion matrix ───────────────────────────────────────────────────────
cm = confusion_matrix(true_labels, preds)
print("\nConfusion Matrix:")
print(cm)

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["none","mild","mod/strong"],
            yticklabels=["none","mild","mod/strong"])
ax.set_title("Confusion Matrix — Economic Grievance Classifier")
ax.set_ylabel("True Label")
ax.set_xlabel("Predicted Label")
plt.tight_layout()
cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"Confusion matrix saved: {cm_path}")

# ── 3. Error analysis ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("ERROR ANALYSIS")
print("="*60)

test_df["pred_label"]  = preds
test_df["confidence"]  = [max(p) for p in probs]
test_df["correct"]     = (preds == true_labels).astype(int)
test_df["error_type"]  = test_df.apply(
    lambda r: "correct" if r["correct"] else
              f"true_{r['economic_grievance_label']}_pred_{r['pred_label']}",
    axis=1
)

print(f"\nTotal errors: {(test_df['correct']==0).sum()} / {len(test_df)}")
print("\nError type breakdown:")
print(test_df[test_df["correct"]==0]["error_type"].value_counts())

print(f"\nMean confidence (correct):   {test_df[test_df['correct']==1]['confidence'].mean():.4f}")
print(f"Mean confidence (incorrect): {test_df[test_df['correct']==0]['confidence'].mean():.4f}")

high_conf_errors = test_df[(test_df["correct"]==0) & (test_df["confidence"]>0.8)]
print(f"\nHigh confidence errors (>0.8): {len(high_conf_errors)} ({len(high_conf_errors)/len(test_df)*100:.1f}% of test set)")

# Lexical analysis
def get_words(text):
    return re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())

stopwords = {"the","and","that","this","have","for","not","with","you",
             "are","was","but","his","her","they","from","had","has",
             "been","which","will","would","could","should","their"}

fp = test_df[(test_df["pred_label"]>0) & (test_df["economic_grievance_label"]==0)]
fn = test_df[(test_df["pred_label"]==0) & (test_df["economic_grievance_label"]>0)]

fp_words = Counter()
fn_words = Counter()
for text in fp["content"]: fp_words.update(get_words(str(text)))
for text in fn["content"]: fn_words.update(get_words(str(text)))

print(f"\nFalse positives: {len(fp)}")
print("Top words:", [(w,c) for w,c in fp_words.most_common(20) if w not in stopwords][:8])
print(f"\nFalse negatives: {len(fn)}")
print("Top words:", [(w,c) for w,c in fn_words.most_common(20) if w not in stopwords][:8])

# Save error posts
error_path = os.path.join(args.output_dir, "error_posts.csv")
test_df[test_df["correct"]==0].to_csv(error_path, index=False)
print(f"\nError posts saved: {error_path}")

# Error plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
test_df[test_df["correct"]==0]["error_type"].value_counts().plot(
    kind="bar", ax=axes[0], color="steelblue", alpha=0.8)
axes[0].set_title("Error Types")
axes[0].set_xlabel("Error type")
axes[0].set_ylabel("Count")
axes[0].tick_params(axis="x", rotation=45)

axes[1].hist(test_df[test_df["correct"]==1]["confidence"],
             bins=20, color="steelblue", alpha=0.7, label="Correct")
axes[1].hist(test_df[test_df["correct"]==0]["confidence"],
             bins=20, color="salmon", alpha=0.7, label="Incorrect")
axes[1].set_title("Model Confidence")
axes[1].set_xlabel("Max probability")
axes[1].set_ylabel("Count")
axes[1].legend()

plt.tight_layout()
error_plot_path = os.path.join(args.output_dir, "error_analysis.png")
plt.savefig(error_plot_path, dpi=150)
plt.close()
print(f"Error analysis plot saved: {error_plot_path}")

# ── 4. McNemar's test (optional) ─────────────────────────────────────────────
if args.lora_model:
    print("\n" + "="*60)
    print("McNEMAR'S TEST: Full FT vs LoRA")
    print("="*60)

    from peft import PeftModel
    from transformers import DistilBertForSequenceClassification as DistilBert

    tokenizer2 = DistilBertTokenizerFast.from_pretrained(
        args.model, local_files_only=True)
    base = DistilBert.from_pretrained("distilbert-base-uncased", num_labels=3)
    lora_model = PeftModel.from_pretrained(base, args.lora_model)
    lora_model.to(device)
    lora_model.eval()

    ds2     = TextDataset(test_df["content"].tolist(), tokenizer2, args.max_len)
    loader2 = DataLoader(ds2, batch_size=args.batch_size, shuffle=False)

    lora_preds = []
    with torch.no_grad():
        for batch in loader2:
            batch = {k: v.to(device) for k, v in batch.items()}
            out   = lora_model(**batch)
            lora_preds.extend(torch.argmax(out.logits, dim=1).cpu().numpy().tolist())
    lora_preds = np.array(lora_preds)

    full_correct = (preds      == true_labels).astype(int)
    lora_correct = (lora_preds == true_labels).astype(int)

    b = ((full_correct==1) & (lora_correct==0)).sum()
    c = ((full_correct==0) & (lora_correct==1)).sum()

    table  = np.array([[((full_correct==1)&(lora_correct==1)).sum(), b],
                       [c, ((full_correct==0)&(lora_correct==0)).sum()]])
    result = mcnemar(table, exact=False, correction=True)

    lora_f1 = f1_score(true_labels, lora_preds, average="macro")
    full_f1 = f1_score(true_labels, preds,      average="macro")

    print(f"Full FT Macro-F1: {full_f1:.4f}")
    print(f"LoRA    Macro-F1: {lora_f1:.4f}")
    print(f"Discordant pairs: {b+c}")
    print(f"McNemar p-value:  {result.pvalue:.4f}")

    if result.pvalue > 0.05:
        print("NOT significant — supports construct instability argument")
    else:
        print("SIGNIFICANT — models make systematically different errors")

print("\nDone.")
