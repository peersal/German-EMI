# Ground truth of the document-level validation surveys

| File | Content |
|---|---|
| `groundtruth_parliament_2024.csv` | 1,142 Bundestag chunks of the 2024 survey (all eras): text, party, era, mean human evidence / intuition rating (−2…+2), majority labels, dictionary scores |
| `groundtruth_parliament_2026.csv` | 300 Bundestag chunks (2015–2025) of the 2026 survey: text, party, date, chunk and whole-speech dictionary EMI, mean human ratings, number of raters |
| `ratings_parliament_2026.csv` | the single ratings of the 2026 Bundestag survey (anonymised rater id, evidence and intuition rating) |
| `groundtruth_twitter_2026.csv` | 300 tweets of the 2026 survey: tweet id, party, date, dictionary EMI, mean human ratings (no tweet text, in line with the X Developer Agreement) |
| `ratings_twitter_2026.csv` | the single ratings of the 2026 Twitter survey (anonymised rater id) |
| `ratings_parliament_2024.csv`, `rater_quality_parliament_2024.csv` | the single ratings of the 2024 survey (anonymised rater codes a2024_r001…, flag for the attention check; built by `export_ratings_2024.py`); item means over raters who passed the check equal `groundtruth_parliament_2024.csv` (its `n_raters` column counts rows of overlapping export files twice for some items) |
| `rater_quality_*_2026.csv` | per anonymised rater: items rated, gold (attention) items shown and passed by direction; raters who failed all their gold items are excluded in the analysis |
| `llm_parliament_2024.csv`, `llm_2026.csv` | ratings of Llama 3.1 8B and Qwen 2.5 7B on the same items (evidence, intuition, 0–4) |

Rater ids are sequential codes; Prolific ids, demographics and timing data are not included.
