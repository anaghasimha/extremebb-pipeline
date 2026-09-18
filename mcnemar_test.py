"""
McNemar's Test: DistilBERT vs DeBERTa
Tests whether the F1 difference (0.59 vs 0.61) is statistically significant
or just noise — supporting/undermining the construct instability argument.
"""

import pandas as pd
import numpy as np
import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    DebertaTokenizer,
    DebertaForSequenceClassification,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from statsmodels.stats.contingency_tables import mcnemar
from torch.utils.data import Dataset, DataLoader
import warnings, os
warnings.filterwarnings("ignore")

BASE_DIR    = "/Users/anaghamv/Downloads/Hate Speech"
DISTIL_DIR  = os.path.join(BASE_DIR, "distilbert_eco_grievance_v4")
DEBERTA_DIR = os.path.join(BASE_DIR, "deberta_eco_grievance_v1")
SEED        = 42
MAX_LEN     = 128
BATCH_SIZE  = 64

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(os.path.join(BASE_DIR, "posts_for_training_v3_clean.csv"))
df["economic_grievance_label"] = pd.to_numeric(
    df["economic_grievance_label"], errors="coerce")
df = df.dropna(subset=["economic_grievance_label","content"]).copy()
df["economic_grievance_label"] = df["economic_grievance_label"].astype(int).replace({3:2})
df["content"] = df["content"].fillna("").astype(str)
df = df[df["content"].str.len() > 10].reset_index(drop=True)

_, test_df = train_test_split(
    df, test_size=0.2, random_state=SEED,
    stratify=df["economic_grievance_label"]
)
test_df     = test_df.reset_index(drop=True)
true_labels = test_df["economic_grievance_label"].values
print(f"Test set: {len(test_df):,} posts")

# ── Dataset ───────────────────────────────────────────────────────────────────
class TextDS(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tok   = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(
            self.texts[i],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        return {k: v.squeeze() for k, v in enc.items()}

# ── Inference function ────────────────────────────────────────────────────────
def get_preds(model_dir, model_class, tok_class):
    print(f"  Loading {os.path.basename(model_dir)}...")
    tok   = tok_class.from_pretrained(model_dir, local_files_only=True)
    model = model_class.from_pretrained(model_dir, local_files_only=True)
    model.eval()

    ds     = TextDS(test_df["content"].tolist(), tok)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    preds  = []

    with torch.no_grad():
        for batch in loader:
            out = model(**{k: v for k, v in batch.items()})
            preds.extend(torch.argmax(out.logits, dim=1).numpy().tolist())

    return np.array(preds)

# ── Get predictions ───────────────────────────────────────────────────────────
print("\nRunning inference...")
distil_preds  = get_preds(DISTIL_DIR,  DistilBertForSequenceClassification,
                           DistilBertTokenizerFast)
deberta_preds = get_preds(DEBERTA_DIR, DebertaForSequenceClassification,
                           DebertaTokenizer)

# ── F1 scores ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("F1 SCORES")
print("="*60)
distil_f1  = f1_score(true_labels, distil_preds,  average="macro")
deberta_f1 = f1_score(true_labels, deberta_preds, average="macro")
print(f"DistilBERT Macro-F1: {distil_f1:.4f}")
print(f"DeBERTa    Macro-F1: {deberta_f1:.4f}")
print(f"Difference:          {deberta_f1 - distil_f1:+.4f}")

print("\nDistilBERT per-class:")
print(classification_report(true_labels, distil_preds,
      target_names=["none","mild","mod/strong"]))

print("DeBERTa per-class:")
print(classification_report(true_labels, deberta_preds,
      target_names=["none","mild","mod/strong"]))

# ── McNemar's test ────────────────────────────────────────────────────────────
print("="*60)
print("McNEMAR'S TEST")
print("="*60)

distil_correct  = (distil_preds  == true_labels).astype(int)
deberta_correct = (deberta_preds == true_labels).astype(int)

# Contingency table
both_correct     = ((distil_correct==1) & (deberta_correct==1)).sum()
distil_only      = ((distil_correct==1) & (deberta_correct==0)).sum()
deberta_only     = ((distil_correct==0) & (deberta_correct==1)).sum()
both_wrong       = ((distil_correct==0) & (deberta_correct==0)).sum()

print(f"\nContingency table:")
print(f"  Both correct:              {both_correct}")
print(f"  DistilBERT correct only:   {distil_only}  (b)")
print(f"  DeBERTa correct only:      {deberta_only}  (c)")
print(f"  Both wrong:                {both_wrong}")
print(f"  Discordant pairs (b+c):    {distil_only + deberta_only}")

table  = np.array([[both_correct, distil_only],
                   [deberta_only, both_wrong]])
result = mcnemar(table, exact=False, correction=True)

print(f"\nMcNemar statistic: {result.statistic:.4f}")
print(f"p-value:           {result.pvalue:.4f}")

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)
if result.pvalue > 0.05:
    print(f"NOT significant (p={result.pvalue:.4f} > 0.05)")
    print("The F1 difference ({distil_f1:.4f} vs {deberta_f1:.4f}) is statistically indistinguishable.")
    print("Both models make essentially the same errors on the same posts.")
    print("SUPPORTS construct instability argument: the ceiling is in the")
    print("construct, not the model — a more powerful model does not")
    print("systematically correct DistilBERT's errors.")
else:
    print(f"SIGNIFICANT (p={result.pvalue:.4f} < 0.05)")
    print("DeBERTa makes systematically different (fewer) errors than DistilBERT.")
    print("Model capacity contributes to performance — partially weakens")
    print("the construct instability argument.")

print("\nDone.")