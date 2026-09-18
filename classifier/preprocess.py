"""
07_prepare_labeling_v2.py
Combines targeted eco-grievance samples into one CSV ready for LLM labeling.
"""

import pandas as pd
import os

BASE_DIR = '/Users/'
OUTPUT   = os.path.join(BASE_DIR, 'posts_to_label_v2.csv')

FILES = [
    'eco_sample_stormfront2025.csv',
    'eco_sample_stormfront_extra.csv',
    'eco_sample_whitenations2021.csv',
    'eco_sample_onionfarms2025.csv',
    'eco_sample_incelsnet2021.csv',
    'eco_sample_survb2024.csv',
]

# ── 1. Load and combine ──────────────────────────────────────────────────────
print("── Loading samples ─────────────────────────────────────────")
dfs = []
for fname in FILES:
    path = os.path.join(BASE_DIR, fname)
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found — skipping")
        continue
    df = pd.read_csv(path, low_memory=False)
    dfs.append(df)
    print(f"  {fname}: {len(df):,} posts")

combined = pd.concat(dfs, ignore_index=True)

# Drop duplicates in case any post_id appears twice
combined = combined.drop_duplicates(subset=['post_id', 'forum_name'])
print(f"\n  Total after dedup: {len(combined):,} posts")

# ── 2. Clean content ─────────────────────────────────────────────────────────
combined['content'] = combined['content'].astype(str).str.strip()
combined['content_preview'] = combined['content'].str[:500]

# ── 3. Add labeling columns ──────────────────────────────────────────────────
combined['economic_grievance_label'] = ''
combined['notes'] = ''

# ── 4. Shuffle ───────────────────────────────────────────────────────────────
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
combined['label_id'] = combined.index + 1

# ── 5. Save labeling version ─────────────────────────────────────────────────
cols = ['label_id', 'post_id', 'forum_name', 'period', 'created_on',
        'economic_grievance_score', 'content_preview',
        'economic_grievance_label', 'notes']
combined[cols].to_csv(OUTPUT, index=False)
print(f"\n  Saved to {OUTPUT}")

# Save training version with full content
training_path = os.path.join(BASE_DIR, 'posts_for_training_v2.csv')
cols_train = ['label_id', 'post_id', 'user_id', 'forum_name', 'period',
              'created_on', 'content', 'economic_grievance_score',
              'job_loss_flag', 'financial_distress_flag', 'wage_flag',
              'scapegoat_flag', 'system_blame_flag', 'macro_aware_flag',
              'economic_grievance_label']
combined[cols_train].to_csv(training_path, index=False)
print(f"  Training file saved to {training_path}")

# ── 6. Summary ───────────────────────────────────────────────────────────────
print("\n── Sample breakdown ────────────────────────────────────────")
print(combined.groupby(['forum_name', 'period']).size().to_string())

print("\n── Seed score distribution ─────────────────────────────────")
print(combined['economic_grievance_score'].value_counts().sort_index())
