"""
Error Analysis of DistilBERT Economic Grievance Classifier
Systematically analyzes misclassified posts to understand:
1. What types of posts does the model get wrong?
2. Are errors systematic or random?
3. What words/patterns correlate with errors?
"""

import pandas as pd
import numpy as np
import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
)
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix
)
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, os
warnings.filterwarnings("ignore")

BASE_DIR   = "/Users/anaghamv/Downloads/Hate Speech"
MODEL_DIR  = os.path.join(BASE_DIR, "distilbert_eco_grievance_v4")
DATA_PATH  = os.path.join(BASE_DIR, "posts_for_training_v3_clean.csv")
SEED       = 42
MAX_LEN    = 128
BATCH_SIZE = 64

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

print(f"Total samples: {len(df):,}")
print(f"Label distribution: {df['economic_grievance_label'].value_counts().sort_index().to_dict()}")

# Same split as training
_, test_df = train_test_split(
    df, test_size=0.2, random_state=SEED,
    stratify=df["economic_grievance_label"]
)
test_df = test_df.reset_index(drop=True)
print(f"Test set: {len(test_df):,}")

# ── Load model ────────────────────────────────────────────────────────────────
print("\nLoading DistilBERT model...")
device    = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
model     = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()
print(f"Model loaded on {device}")

# ── Run inference on test set ─────────────────────────────────────────────────
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
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
        }

print("\nRunning inference on test set...")
dataset = TextDataset(test_df["content"].tolist(), tokenizer, MAX_LEN)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

all_preds  = []
all_probs  = []

with torch.no_grad():
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        outputs   = model(input_ids=input_ids, attention_mask=attn_mask)
        probs     = torch.softmax(outputs.logits, dim=1).cpu().numpy()
        preds     = np.argmax(probs, axis=1)
        all_preds.extend(preds.tolist())
        all_probs.extend(probs.tolist())

test_df["pred_label"]    = all_preds
test_df["prob_none"]     = [p[0] for p in all_probs]
test_df["prob_mild"]     = [p[1] for p in all_probs]
test_df["prob_strong"]   = [p[2] for p in all_probs]
test_df["pred_score"]    = 1*test_df["prob_mild"] + 2*test_df["prob_strong"]
test_df["confidence"]    = [max(p) for p in all_probs]
test_df["correct"]       = (test_df["pred_label"] == test_df["economic_grievance_label"]).astype(int)
test_df["error_type"]    = test_df.apply(
    lambda r: "correct" if r["correct"] else
              f"true_{r['economic_grievance_label']}_pred_{r['pred_label']}",
    axis=1
)

# ── Overall metrics ───────────────────────────────────────────────────────────
print("\n" + "="*60)
print("OVERALL METRICS")
print("="*60)
labels = test_df["economic_grievance_label"].values
preds  = test_df["pred_label"].values

print(classification_report(labels, preds,
      target_names=["none","mild","moderate/strong"]))
print(f"Macro-F1: {f1_score(labels, preds, average='macro'):.4f}")

# ── Confusion matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(labels, preds)
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["none","mild","mod/strong"],
            yticklabels=["none","mild","mod/strong"])
ax.set_title("Confusion Matrix — DistilBERT Economic Grievance")
ax.set_ylabel("True Label")
ax.set_xlabel("Predicted Label")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "error_confusion_matrix.png"), dpi=150)
plt.close()
print("\nConfusion matrix saved.")

# ── Error type breakdown ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("ERROR TYPE BREAKDOWN")
print("="*60)
error_counts = test_df[test_df["correct"]==0]["error_type"].value_counts()
print(error_counts)

# ── Confidence analysis ───────────────────────────────────────────────────────
print("\n" + "="*60)
print("CONFIDENCE ANALYSIS")
print("="*60)
print(f"Mean confidence (correct):   {test_df[test_df['correct']==1]['confidence'].mean():.4f}")
print(f"Mean confidence (incorrect): {test_df[test_df['correct']==0]['confidence'].mean():.4f}")
print(f"\nHigh confidence errors (>0.8 confidence but wrong):")
high_conf_errors = test_df[(test_df["correct"]==0) & (test_df["confidence"]>0.8)]
print(f"  Count: {len(high_conf_errors)} ({len(high_conf_errors)/len(test_df)*100:.1f}% of test set)")
print(f"  Error types:")
print(high_conf_errors["error_type"].value_counts())

# ── Lexical analysis of errors ────────────────────────────────────────────────
print("\n" + "="*60)
print("LEXICAL ANALYSIS — What words appear in errors?")
print("="*60)

# Ambiguous words — appear in both correct and incorrect predictions
from collections import Counter
import re

def get_words(text):
    return re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())

# False positives: model says grievance but true label is none
fp = test_df[(test_df["pred_label"] > 0) & (test_df["economic_grievance_label"] == 0)]
# False negatives: model says none but true label is grievance
fn = test_df[(test_df["pred_label"] == 0) & (test_df["economic_grievance_label"] > 0)]
# True positives
tp = test_df[(test_df["pred_label"] > 0) & (test_df["economic_grievance_label"] > 0)]

fp_words = Counter()
fn_words = Counter()
tp_words = Counter()

for text in fp["content"]:
    fp_words.update(get_words(str(text)))
for text in fn["content"]:
    fn_words.update(get_words(str(text)))
for text in tp["content"]:
    tp_words.update(get_words(str(text)))

# Stopwords to exclude
stopwords = {"the","and","that","this","have","for","not","with","you",
             "are","was","but","his","her","they","from","had","has",
             "been","which","will","would","could","should","their",
             "what","when","who","how","all","one","can","more","also",
             "just","its","our","your","there","some","than","then",
             "into","about","out","they","she","him","her","was","were"}

print(f"\nFalse Positives (predicted grievance, actually none): {len(fp)}")
print("Top words in false positives:")
fp_top = [(w,c) for w,c in fp_words.most_common(20) if w not in stopwords][:10]
for w, c in fp_top:
    print(f"  '{w}': {c}")

print(f"\nFalse Negatives (missed grievance): {len(fn)}")
print("Top words in false negatives:")
fn_top = [(w,c) for w,c in fn_words.most_common(20) if w not in stopwords][:10]
for w, c in fn_top:
    print(f"  '{w}': {c}")

# ── Ambiguous word analysis ───────────────────────────────────────────────────
print("\n" + "="*60)
print("AMBIGUOUS WORD ANALYSIS")
print("='*60")
print("Words that appear in BOTH false positives and false negatives")
print("(these are the semantically unstable words):")

ambiguous = set(w for w,_ in fp_top) & set(w for w,_ in fn_top)
if ambiguous:
    print(f"  {ambiguous}")
else:
    # Find words with high FP AND FN rates
    all_error_words = set(w for w,_ in fp_top) | set(w for w,_ in fn_top)
    print("  Top words appearing in errors:")
    for w in list(all_error_words)[:10]:
        fp_count = fp_words.get(w, 0)
        fn_count = fn_words.get(w, 0)
        tp_count = tp_words.get(w, 0)
        if fp_count + fn_count > 0:
            print(f"  '{w}': FP={fp_count}, FN={fn_count}, TP={tp_count}")

# ── Sample error analysis ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("SAMPLE FALSE POSITIVES (model says grievance, truth is none)")
print("="*60)
for _, row in fp.head(5).iterrows():
    print(f"\nTrue: {row['economic_grievance_label']} | "
          f"Pred: {row['pred_label']} | "
          f"Confidence: {row['confidence']:.2f}")
    print(f"Text: {str(row['content'])[:200]}")

print("\n" + "="*60)
print("SAMPLE FALSE NEGATIVES (model missed grievance)")
print("="*60)
for _, row in fn.head(5).iterrows():
    print(f"\nTrue: {row['economic_grievance_label']} | "
          f"Pred: {row['pred_label']} | "
          f"Confidence: {row['confidence']:.2f}")
    print(f"Text: {str(row['content'])[:200]}")

# ── Forum-level error analysis ────────────────────────────────────────────────
if "forum_name" in test_df.columns:
    print("\n" + "="*60)
    print("ERROR RATE BY FORUM")
    print("="*60)
    forum_errors = test_df.groupby("forum_name").agg(
        accuracy=("correct","mean"),
        n=("correct","count"),
        macro_f1=("correct", lambda x: f1_score(
            test_df.loc[x.index,"economic_grievance_label"],
            test_df.loc[x.index,"pred_label"],
            average="macro"
        ))
    ).round(3)
    print(forum_errors.sort_values("accuracy"))

# ── Confidence histogram ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(test_df[test_df["correct"]==1]["confidence"],
             bins=20, color="steelblue", alpha=0.7, label="Correct")
axes[0].hist(test_df[test_df["correct"]==0]["confidence"],
             bins=20, color="salmon", alpha=0.7, label="Incorrect")
axes[0].set_title("Model Confidence — Correct vs Incorrect")
axes[0].set_xlabel("Max probability (confidence)")
axes[0].set_ylabel("Count")
axes[0].legend()

# Error type bar chart
error_counts.plot(kind="bar", ax=axes[1], color="steelblue", alpha=0.8)
axes[1].set_title("Error Types")
axes[1].set_xlabel("Error type (true_X_pred_Y)")
axes[1].set_ylabel("Count")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "error_analysis.png"), dpi=150)
plt.close()
print("\nError analysis plot saved.")

# ── Save error dataframe ──────────────────────────────────────────────────────
error_df = test_df[test_df["correct"]==0].copy()
error_df.to_csv(os.path.join(BASE_DIR, "error_analysis_posts.csv"), index=False)
print(f"Error posts saved: {len(error_df)} errors out of {len(test_df)} test posts")

print("\nDone.")