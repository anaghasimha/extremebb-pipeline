"""
08_llm_label_ollama.py
Uses Ollama local LLM to label posts for economic grievance intensity.
Run `ollama serve` in a separate terminal before running this script.
"""

import pandas as pd
import json
import time
import os
import requests
import re

BASE_DIR   = '/home/anagha'
INPUT      = os.path.join(BASE_DIR, 'posts_for_training_new_forums.csv')
OUTPUT     = os.path.join(BASE_DIR, 'posts_labeled_ollama.csv')
CHECKPOINT = os.path.join(BASE_DIR, 'labeling_checkpoint_ollama.csv')

OLLAMA_URL = 'http://localhost:11434/api/generate'
MODEL      = 'llama3.2:1b'
SLEEP_SEC  = 0.5

# Label ONE post at a time — more reliable for small models
PROMPT_TEMPLATE = """Label this forum post for economic grievance. Reply with ONLY a single digit: 0, 1, 2, or 3.

0 = No economic content (cultural/racial topics, no money/jobs/economy)
1 = Economy mentioned in passing, no strong emotion
2 = Economic frustration or blame, no specific scapegoat
3 = Strong economic grievance with scapegoating or crisis framing

Examples:
"Poor Barbie! There goes the neighbourhood!" -> 0
"The unemployment numbers came out today" -> 1
"I can't find a decent job, this economy is terrible" -> 2
"Immigrants are deliberately taking our jobs and destroying our wages" -> 3
"USS Liberty" or "ZOG" with no economic content -> 0

Post: {post}

Reply with ONLY the digit (0, 1, 2, or 3):"""

def clean_post(text):
    text = str(text)
    text = re.sub(r'\*{2,}[A-Z]+\*{2,}', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:300]

def label_single(post_text):
    """Label one post at a time — more reliable than batch JSON."""
    prompt = PROMPT_TEMPLATE.format(post=clean_post(post_text))

    response = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 10  # only need one digit
        }
    }, timeout=60)

    if response.status_code != 200:
        return None, "api error"

    text = response.json().get('response', '').strip()

    # Extract first digit found
    match = re.search(r'[0-3]', text)
    if match:
        return int(match.group()), text.strip()
    return None, f"no digit found: {text[:50]}"

# ── 1. Load posts ─────────────────────────────────────────────────────────────
print("── Loading posts ────────────────────────────────────────────")
df = pd.read_csv(INPUT, low_memory=False)
print(f"  Total posts: {len(df):,}")

# ── 2. Check checkpoint ───────────────────────────────────────────────────────
if os.path.exists(CHECKPOINT):
    done = pd.read_csv(CHECKPOINT)
    done_ids = set(done['label_id'].tolist())
    df_todo = df[~df['label_id'].isin(done_ids)].copy()
    print(f"  Resuming: {len(done_ids)} done, {len(df_todo)} remaining")
    results = done.to_dict('records')
else:
    df_todo = df.copy()
    results = []
    print(f"  Starting fresh")

# ── 3. Test connection ────────────────────────────────────────────────────────
print("\n── Testing Ollama connection ────────────────────────────────")
try:
    test = requests.get('http://localhost:11434/api/tags', timeout=5)
    models = [m['name'] for m in test.json().get('models', [])]
    print(f"  Connected. Models: {models}")
except Exception as e:
    print(f"  Cannot connect: {e}")
    exit(1)

# ── 4. Label one by one ───────────────────────────────────────────────────────
print(f"\n── Labeling {len(df_todo):,} posts one by one ───────────────")
errors = 0

for i, (_, row) in enumerate(df_todo.iterrows()):
    try:
        label, rationale = label_single(row['content_preview'])

        if label is None:
            # Fallback to seed score
            score = row.get('economic_grievance_score', 1)
            label = 1 if score <= 2 else 2 if score <= 4 else 3
            rationale = f'fallback: {rationale}'
            needs_review = 1
            errors += 1
        else:
            needs_review = 0

        results.append({
            'label_id':   row['label_id'],
            'post_id':    row['post_id'],
            'forum_name': row['forum_name'],
            'period':     row['period'],
            'created_on': row['created_on'],
            'content_preview': row['content_preview'],
            'economic_grievance_label': label,
            'rationale':  rationale,
            'needs_review': needs_review
        })

        # Progress and checkpoint
        if (i + 1) % 500 == 0:
            pd.DataFrame(results).to_csv(CHECKPOINT, index=False)
            pct = (i + 1) / len(df_todo) * 100
            print(f"  {i+1:,}/{len(df_todo):,} ({pct:.1f}%) — checkpoint saved")
        elif (i + 1) % 100 == 0:
            pct = (i + 1) / len(df_todo) * 100
            print(f"  {i+1:,}/{len(df_todo):,} ({pct:.1f}%)...", end='\r')

        time.sleep(SLEEP_SEC)

    except Exception as e:
        print(f"  Error on row {i}: {e}")
        errors += 1

# ── 5. Save ───────────────────────────────────────────────────────────────────
print(f"\n── Saving ───────────────────────────────────────────────────")
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT, index=False)
print(f"  {len(results_df):,} posts saved to {OUTPUT}")
print(f"  Errors/fallbacks: {errors}")

print("\n── Label distribution ───────────────────────────────────────")
print(results_df['economic_grievance_label'].value_counts().sort_index())

print("\n── By forum ─────────────────────────────────────────────────")
print(results_df.groupby('forum_name')['economic_grievance_label'].mean().round(3))