# extremebb-pipeline

# Data

## ExtremeBB Corpus

The ExtremeBB corpus is maintained by the Cambridge Cybercrime Centre and is available
under a formal license agreement.

The corpus covers 14 forums and 36 million posts spanning 2001-2023.

**We cannot share the raw post data or training labels due to licensing restrictions.**

## Expected Data Files

Once you have access, place the following files in this directory:

```
data/
├── posts_to_label.csv          # Posts sampled for annotation
├── posts_for_training.csv      # Labeled training data (post_id, content, economic_grievance_label)
├── user_quarter_panel.csv      # User-quarter panel with grievance scores
├── nfp_surprise_series.csv     # NFP surprise series (Philadelphia Fed SPF)
└── cpi_surprise_series.csv     # CPI surprise series (Philadelphia Fed SPF)
```

## Macroeconomic Data

NFP and CPI surprise series are constructed from:
- **BLS release dates**: https://www.bls.gov/schedule/news_release/empsit.htm
- **Philadelphia Fed SPF forecasts**: https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/survey-of-professional-forecasters

## Reddit Data

Reddit deplatforming data (The_Donald, Braincels, TheRedPill, exredpill) is
available from Academic Torrents via the Pushshift archive:
https://academictorrents.com/details/56aa49f9653ba545f48df2e33679f014d2829c10

# ExtremeBB Economic Grievance Pipeline

This repository contains the code for the paper:

**"Domain-Adapted Text Classification and Causal Estimation under Distribution Shift in Online Communities"**  
Anagha MV Simha, Michaela Pagel — Washington University in St. Louis

---

## Overview

We study the relationship between macroeconomic shocks and activity on extremist online forums using the ExtremeBB corpus — a database of 36 million posts across 14 extremist bulletin board forums spanning 2001-2023.

The pipeline has two main components:

1. **Economic Grievance Classifier** — A fine-tuned DistilBERT model that assigns a continuous economic grievance score (0-2) to each post, trained on 2,497 human-validated labels with LLM-assisted annotation.

2. **Causal Identification** — A difference-in-differences design exploiting the scheduled release timing of macroeconomic announcements (NFP, CPI) as exogenous variation to estimate the causal effect of economic shocks on extremist forum activity.

---

## Repository Structure

```
extremebb-pipeline/
├── README.md
├── requirements.txt
├── data/
│   └── README.md          # Data access instructions
├── annotation/
│   └── annotate.py        # LLM-assisted annotation via Ollama
├── classifier/
│   ├── preprocess.py      # Prepare labeled data for training
│   ├── train.py           # Fine-tune DistilBERT classifier
│   └── evaluate.py        # Evaluate classifier, error analysis, McNemar test
├── ablations/
│   ├── lora_finetune.py   # PEFT/LoRA experiments across ranks
│   └── deberta_finetune.py # DeBERTa-base comparison
└── analysis/
    └── did_analysis.py    # Difference-in-differences event study
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Obtain data

See `data/README.md` for instructions on accessing the ExtremeBB corpus via the Cambridge Cybercrime Centre.

### 3. Annotate posts

```bash
# Start Ollama server first
ollama serve

# Run LLM-assisted annotation
python annotation/annotate.py \
    --input data/posts_to_label.csv \
    --output data/posts_labeled.csv \
    --model llama3.2:1b
```

### 4. Train classifier

```bash
python classifier/train.py \
    --data data/posts_for_training.csv \
    --output models/distilbert_eco_grievance \
    --epochs 5 \
    --batch_size 16
```

### 5. Evaluate classifier

```bash
python classifier/evaluate.py \
    --model models/distilbert_eco_grievance \
    --data data/posts_for_training.csv
```

### 6. Run ablations

```bash
# LoRA experiments
python ablations/lora_finetune.py \
    --data data/posts_for_training.csv \
    --ranks 4 8 16 32

# DeBERTa comparison
python ablations/deberta_finetune.py \
    --data data/posts_for_training.csv
```

### 7. Run causal analysis

```bash
python analysis/did_analysis.py \
    --panel data/user_quarter_panel.csv \
    --nfp data/nfp_surprise_series.csv \
    --cpi data/cpi_surprise_series.csv
```

---

## Model

The trained DistilBERT economic grievance classifier (v4) is available on Hugging Face:

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="AnaghaSimha/economic-grievance-distilbert",
    return_all_scores=True
)

result = classifier("Immigrants are taking our jobs and wages keep falling.")
# grievance_score = 0*P(LABEL_0) + 1*P(LABEL_1) + 2*P(LABEL_2)
```

---

## Key Results

| Specification | β_NFP | p-value | N |
|---|---|---|---|
| Treatment forums only | -0.0064 | <0.001 | 219,244 |
| Control forums only | +0.0025 | <0.001 | 77,341 |
| Full panel | -0.0003 | 0.395 | 296,585 |

**Classifier performance:**

| Method | Macro-F1 |
|---|---|
| TF-IDF + Logistic Regression | 0.41 |
| DistilBERT LoRA (r=4 to 32) | 0.55-0.58 |
| DistilBERT Full Fine-Tuning | 0.59 |
| DeBERTa-base Full Fine-Tuning | 0.61 |

---



## Authors

- **Anagha MV Simha** — Washington University in St. Louis ([anagha@wustl.edu](mailto:anagha@wustl.edu))
- **Michaela Pagel** — Washington University in St. Louis ([mpagel@wustl.edu](mailto:mpagel@wustl.edu))

*Research conducted under formal license agreement with the Cambridge Cybercrime Centre.*
