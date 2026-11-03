# Part B of the open analyses: count-dictionary baseline for EMI (Reviewer 1, R1.13)

Status 3 Nov 2026.

## 1. Question

Reviewer 1: "How do these embedding-based scores compare to a simple dictionary baseline on the
human-validated sample? I expect the simple dictionary would produce much sparser estimates, but it
could be good to see why the embedding EMI score is a better approach." The response so far said no
baseline had been run.

## 2. What was done

Script: `code/05_validation/document/0520_count_baseline.py` (a few seconds).

- Count-based EMI on the same 86 validated keywords as the DDR dictionaries (Extended Data Table 1;
  `code/05_validation/keywords/validated_keywords_german.csv`: 48 evidence, 38 intuition terms):
  (evidence hits − intuition hits) per 100 tokens of the lower-cased text. Multi-word keywords
  (for example "Fake News", "gesunder Menschenverstand") are matched as phrases.
- Two matching rules: **exact** word form, and **prefix** (a token counts if it begins with the
  keyword), which approximates inflected forms and compounds without a lemmatiser.
- Items: the 1,141 chunks of the 2025 Bundestag survey, the 290 chunks of the 2026 Bundestag survey
  and the 300 tweets of the 2026 survey, with their human ratings (human EMI = mean evidence rating
  − mean intuition rating; label = its sign, ties dropped), the DDR EMI, and the LLM composite from
  `code/05_validation/document/data_2026/llm_parliament_2024.csv` and `llm_2026.csv`.
- Reported per measure: AUC against the human label, Spearman ρ with the human EMI, Spearman ρ with
  the DDR EMI, and for the count score the share of items with a score of exactly zero and the share
  with no keyword hit at all.

## 3. Results

| Measure | Bundestag chunks pooled (n = 1,348) | | | Tweets (n = 282) | | |
|---|---|---|---|---|---|---|
| | AUC | ρ human | zero share | AUC | ρ human | zero share |
| count EMI, exact | 0.57 | 0.15 | 68 % | 0.50 | 0.04 | 90 % |
| count EMI, prefix | 0.59 | 0.18 | 56 % | 0.53 | 0.07 | 84 % |
| DDR EMI | 0.73 | 0.46 | | 0.66 | 0.36 | |
| LLM composite | 0.78 | 0.54 | | 0.76 | 0.43 | |

- The count score is sparse: two thirds of the Bundestag chunks (45–90 words) and nine of ten tweets
  contain none of the 86 keywords in exact form; with prefix matching still 53 % and 82 % have no
  hit. A chunk with a hit typically has one (mean 0.3 evidence and 0.2 intuition hits per chunk,
  exact).
- Where it is non-zero it points the right way (ρ with the DDR EMI 0.38 exact, 0.43 prefix on the
  chunks), but it barely separates the human labels (AUC 0.57–0.59 on the chunks, chance on
  tweets).
- The two Bundestag surveys give the same picture separately (`summary.csv`). The tweet AUC of the
  LLM composite here (0.76) is computed on the sign of the mean human EMI; the SI reports 0.73 with
  its own label rule.
- Figure `fig_count_baseline.png`: ROC curves of the four measures on the pooled chunks (A) and the
  tweets (B).

Why the embedding score does better: the DDR score uses the similarity of every word in a passage to
the dictionary vectors, so a passage about "Statistik", "belegt" or "Studien" scores high on evidence
even though none of these words is a keyword; the count score only registers the 86 literal forms
and is zero for most short texts.

## 4. Files

| File | Content |
|---|---|
| `count_scores.csv` | per item: tokens, evidence and intuition hits (exact, prefix), count EMI (both rules), DDR EMI, LLM composite, human EMI, label, survey |
| `summary.csv` | AUC, ρ with human EMI, ρ with DDR EMI, zero share, no-hit share per measure and set |
| `numbers.json` | the summary plus mean hits per item |
| `fig_count_baseline.png/.pdf` | ROC curves |
