# LLM ratings on the 2026 validation surveys

Run 22 October 2026 on one H100 GPU (about 5 minutes): Llama 3.1 8B Instruct and Qwen 2.5 7B Instruct rate the
300 Bundestag chunks and the 300 tweets of the new Prolific surveys with the rating prompt (the two survey items, 0–4)
and the use-vs-mention prompt. Inputs `inputs/survey_input.csv` and `inputs/survey_meta.csv` (built by
`build_survey_input.py`), outputs `output/survey_{rating,use_mention}_<model>.parquet`, analysis `analyze_survey.py`
(table `llm_vs_human.csv`, `numbers.json`).

Human label: mean rater EMI (evidence − intuition) > 0 evidence-based, < 0 intuition-based, ties dropped.
LLM composite: mean over the two models of evidence rating − intuition rating.

## Agreement with the human ratings (AUC against the human label, Spearman ρ with the mean human EMI)

| Measure | Bundestag chunks (n = 283) | Tweets (n = 282) |
|---|---|---|
| Dictionary EMI | AUC 0.72, ρ 0.42 | AUC 0.66, ρ 0.36 |
| Llama composite | AUC 0.81, ρ 0.57 | AUC 0.69, ρ 0.30 |
| Qwen composite | AUC 0.78, ρ 0.59 | AUC 0.72, ρ 0.41 |
| Two-model composite | AUC 0.82, ρ 0.63 | AUC 0.76, ρ 0.43 |
| Inter-model agreement (ρ, composites) | 0.69 | 0.41 |
| LLM composite vs dictionary EMI (ρ) | 0.43 | 0.27 |

Components (two models): Bundestag evidence AUC 0.78 / 0.76, intuition 0.79 / 0.81; tweets evidence 0.66 / 0.72,
intuition 0.71 / 0.72.

## By party (AUC of the two-model composite / dictionary EMI)

| Party | Bundestag | Tweets |
|---|---|---|
| AfD | 0.89 / 0.86 | 0.70 / 0.72 |
| CDU/CSU | 0.91 / 0.75 | 0.76 / 0.60 |
| SPD | 0.78 / 0.69 | 0.86 / 0.74 |
| FDP | 0.79 / 0.83 | 0.69 / 0.63 |
| Greens | 0.84 / 0.71 | 0.62 / 0.49 |
| Left | 0.72 / 0.56 | 0.80 / 0.70 |

## Reading

- The pattern of the 2024 survey repeats on the new, independent samples: the LLM composite agrees with human raters
  better than the dictionary EMI (Bundestag 0.82 vs 0.72, before: 0.77 vs 0.71), and the dictionary's validity is
  lowest for the Left in the Bundestag and for the Greens on Twitter.
- On tweets both measures are weaker (LLM 0.76, dictionary 0.66) and the two models agree less with each other
  (ρ 0.41 vs 0.69 on chunks): short texts leave less to rate. The gap between LLM and dictionary is similar on both
  platforms (about 0.10 AUC).
- The dictionary EMI and the LLM composite correlate only moderately with each other (ρ 0.43 chunks, 0.27 tweets)
  while both correlate with the human ratings, so they capture partly different aspects of the same construct.
