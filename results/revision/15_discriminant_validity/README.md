# Part A of the open analyses: discriminant validity of EMI against moral and emotional language

Status 3 Nov 2026.

## 1. Question (Reviewer 4, comments R4.1, R4.3, R4.4, R4.12)

Does EMI measure epistemic style, or is it moralisation, emotionality or antagonism under another
name? The response document so far conceded that no separate discriminant analysis had been run.

## 2. What was done

Script: `code/05_validation/document/0519_discriminant_validity.py` (runs in about 70 s; log in
stdout).

**Dictionaries** (exact word-form lists, counts per 100 tokens of the lower-cased text, tokens =
runs of `a–z ä ö ü ß`; copies in `data/dictionaries/`):

| Score | Source | Word forms |
|---|---|---|
| moral | multilingual Moral Political Dictionary, German file `mMPD_de.yml`, category MORALITY (union of the five foundations, virtue and vice poles, general moral terms); also `moral_virtue` (POSITIVE) and `moral_vice` (NEGATIVE) | 6,181 (2,270 / 3,634) |
| negemo | ed8 German emotion dictionary, union of ANGER, FEAR, DISGUST, SADNESS | 11,624 |
| posemo | ed8, union of JOY, ENTHUSIASM, PRIDE, HOPE | 9,005 |

Sources: mMPD from https://github.com/tweedmann/mMPD (Simonsen & Widmann, *British Journal of
Political Science* 2025, "When do political parties moralize?"; free for academic use, cite the
article); ed8 from https://github.com/tweedmann/3x8emotions (Widmann & Wich, *Political Analysis*
2023, "Creating and comparing dictionary, word embedding, and transformer-based models to measure
discrete emotions in German political text"). SentiWS was planned as a negativity baseline but the
Leipzig server did not respond; the four negative ed8 emotions cover negativity.

Bib entries to add to the SI bibliography:

```bibtex
@article{simonsen-widmann-2025,
  author  = {Simonsen, Kristina Bakk{\ae}r and Widmann, Tobias},
  title   = {When Do Political Parties Moralize? A Cross-National Study of the Use of Moral Language in Political Communication on Immigration},
  journal = {British Journal of Political Science},
  volume  = {55},
  year    = {2025},
  doi     = {10.1017/S0007123425000122}
}
@article{widmann-wich-2022,
  author  = {Widmann, Tobias and Wich, Maximilian},
  title   = {Creating and Comparing Dictionary, Word Embedding, and Transformer-Based Models to Measure Discrete Emotions in German Political Text},
  journal = {Political Analysis},
  volume  = {31},
  number  = {4},
  pages   = {560--575},
  year    = {2023},
  doi     = {10.1017/pan.2022.15}
}
```

**Data**

- Validation items with human ratings: 1,141 Bundestag chunks of the 2025 survey
  (`code/05_validation/document/data_2026/groundtruth_parliament_2024.csv`, one duplicate id dropped),
  290 chunks of the 2026 Bundestag survey (`groundtruth_parliament_2026.csv`), 300 tweets of the
  2026 survey (`groundtruth_twitter_2026.csv`; texts looked up by tweet id in
  `data/twitter_nhb.csv`). Human EMI = mean evidence rating − mean intuition rating; binary label =
  its sign, ties dropped (1,347 Bundestag chunks, 282 tweets).
- All 59,170 Bundestag speeches 2015–2025 (`data/parliament_nhb.csv`, raw `text` column).

**Steps**

1. Score every item and speech with the three dictionaries.
2. Spearman correlations between dictionary EMI, human EMI and the three scores; AUC of each score
   against the human label.
3. Logistic regression of the human label on EMI (per s.d.) with and without moral, negative
   emotion, positive emotion and log text length as covariates; AUC of EMI alone, with controls,
   of the controls alone, and of EMI residualised on the controls.
4. Constrained breakpoint model (Eq. 2 of the main text: linear trend by camp, level shift and
   slope change at the 2017, 2021 and 2025 breakpoints, lagged quarterly actor mean, random speaker
   intercepts; n = 56,066 speeches, same sample as the main model) for EMI without and with the
   three standardised scores as document-level covariates, and with each score (standardised) as
   the outcome.

## 3. Results

### Validation items

| | Bundestag chunks (2025 + 2026) | Tweets (2026) |
|---|---|---|
| items (correlations / non-tied label) | 1,431 / 1,347 | 300 / 282 |
| ρ(dictionary EMI, human EMI) | 0.46 | 0.36 |
| ρ(dictionary EMI, moral) | −0.32 | −0.18 |
| ρ(dictionary EMI, negative emotion) | −0.40 | −0.34 |
| ρ(dictionary EMI, positive emotion) | −0.22 | 0.05 |
| ρ(human EMI, moral) | −0.19 | 0.01 |
| ρ(human EMI, negative emotion) | −0.19 | −0.17 |
| ρ(human EMI, positive emotion) | −0.16 | 0.09 |
| AUC dictionary EMI | 0.73 | 0.66 |
| AUC moral / negative / positive emotion | 0.41 / 0.42 / 0.42 | 0.54 / 0.45 / 0.51 |
| logit: EMI coefficient per s.d., alone → with controls | 0.90 (0.07) → 0.85 (0.07) | 0.55 (0.18) → 0.61 (0.19) |
| AUC: EMI alone / EMI + controls / controls alone | 0.73 / 0.74 / 0.63 | 0.66 / 0.70 / 0.61 |
| AUC: EMI residualised on the controls | 0.70 | 0.66 |

AUC values below 0.5 mean that more moral or emotional language goes with the *intuition* label
(oriented AUC 0.58–0.59). The 2025 and 2026 Bundestag surveys separately give the same picture
(`auc_logit.csv`). The tweet AUC of 0.66 for EMI differs from the 0.68 in the SI because the label
here is the sign of the mean human EMI on the 282 non-tied tweets scored in this run.

### Speeches 2015–2025

- Speech-level Spearman correlation of EMI with moral −0.22, negative emotion −0.29, positive
  emotion −0.05 (n = 59,170).
- Mean hits per 100 tokens: moral 1.50, negative emotion 1.21, positive emotion 2.61.
- Before vs after the AfD's entry (24 Oct 2017): moral 1.50 → 1.50 (Cohen's d = 0.01), negative
  emotion 1.16 → 1.23 (d = 0.07), positive emotion 2.48 → 2.64 (d = 0.13), EMI d = −0.44.
- Yearly means in `yearly_means.csv`: moral language flat (1.45–1.66), negative emotion 1.14–1.39,
  positive emotion 2.46–2.79, while EMI falls from +0.5 to −0.8.

Constrained breakpoint model (level shifts, s.e.):

| Outcome | Covariates | 2017 | 2021 | 2025 | covariate coefficients (per s.d.) |
|---|---|---|---|---|---|
| EMI | none | −0.421 (0.027) | −0.207 (0.023) | −0.502 (0.050) | |
| EMI | moral + neg. + pos. emotion | −0.444 (0.026) | −0.246 (0.022) | −0.462 (0.048) | moral −0.115, negemo −0.210, posemo −0.099 |
| moral (z) | none | −0.013 (0.024), P = 0.60 | −0.090 (0.020) | +0.064 (0.045), P = 0.16 | |
| negative emotion (z) | none | −0.090 (0.023) | −0.086 (0.020) | +0.130 (0.044) | |
| positive emotion (z) | none | +0.039 (0.023), P = 0.09 | −0.079 (0.020) | +0.072 (0.044), P = 0.11 | |

For comparison the EMI 2017 shift is −0.33 s.d. of EMI.

### Reading

- EMI predicts what human raters call evidence- or intuition-based; moral and emotional language
  does not (AUC 0.41–0.42 vs 0.73), and the human EMI is only weakly related to the three
  constructs (|ρ| ≤ 0.19).
- Controlling for the three scores and text length leaves the EMI coefficient and the AUC
  unchanged; EMI residualised on the controls still reaches 0.70. The constructs are correlated with
  the dictionary EMI (intuition-based passages carry somewhat more moral and emotional vocabulary,
  as the reviewer expected), but they do not carry its discrimination.
- In the chamber, moral and emotional language did not break in 2017 (−0.01 and −0.09 s.d.), and
  adding them as covariates leaves the 2017 EMI shift at −0.44. The shift towards intuition-based
  language is not a shift towards moralisation or negativity under another name.

## 4. Files in this folder

| File | Content |
|---|---|
| `validation_scores.csv` | per item: dictionary EMI, human ratings, human EMI, label, dictionary scores, token count, survey |
| `correlations.csv` | Spearman matrix per set (pooled Bundestag, 2025, 2026, Twitter) |
| `auc_logit.csv` | AUCs and logistic models per set |
| `speech_scores.csv` (not included, 8 MB; recreated by the script) | moral / emotion scores per speech (59,170 rows) |
| `yearly_means.csv` | yearly means of EMI and the three scores |
| `breakpoint_models.csv` | full coefficient tables of the five mixed models |
| `numbers.json` | all numbers above |
| `fig_discriminant.png/.pdf` | A: correlation matrix on the 1,431 chunks; B: level shifts for the five models |

## 5. Caveats

- The model with moral language as outcome did not reach the optimiser's convergence tolerance
  (`converged = False` in `breakpoint_models.csv`); the coefficients are small and stable across the
  retried optimisers, but say so if reported.
- Dictionary scores are exact word-form counts; ed8 is a political-text dictionary, mMPD a
  translated moral-foundations dictionary. No LIWC (licensed) and no SentiWS (server unreachable).
- The correlation panel uses all 1,431 chunks (including ties); the AUC and logit use the 1,347
  non-tied ones.
