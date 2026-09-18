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
