# EMI-Germany

**Rhetorical convergence toward intuition-based discourse following the rise of the far right in Germany**

Companion repository for Saleth, Aroyehun, Carrella, Abels, Lewandowsky & Garcia. Code, validation surveys, and figures for the paper.

## Project overview

The spread of misinformation is widely perceived as a threat to democratic deliberation, yet how political elites' rhetorical commitments to truth shift in tandem with the rise of populist actors remains poorly understood. This project analyses **4.5 million tweets** and **56,213 parliamentary speeches** from German political elites between **2015 and 2025** and measures the relative prevalence of evidence-based and intuition-based rhetoric using a validated **Distributed Dictionary Representation (DDR)**.

For every text we compute an **Evidence-Minus-Intuition (EMI)** score: the z-scored cosine similarity to a German evidence dictionary minus the z-scored cosine similarity to a German intuition dictionary, where both similarities are taken in a domain-specific word2vec embedding space pre-trained on the historical Bundestag corpus (1867–2025) and fine-tuned per platform. Texts with EMI < 0 lean toward intuition-based language; EMI > 0 toward evidence-based language.

Across both arenas, intuition-based language has become more prominent, with right-leaning actors consistently exhibiting the lowest EMI scores. The AfD's 2017 parliamentary entry coincides with a discrete downward shift in EMI for the broader Bundestag chamber, while a more gradual decline is observed on Twitter. These findings document a co-movement between far-right visibility and a discourse-wide shift toward intuition-based rhetoric in a multiparty European democracy.

The Twitter analysis is among the first to use platform data accessed through the EU Digital Services Act (DSA, Regulation (EU) 2022/2065) Article 40 vetted-researcher framework.

## Repository layout

```
German-EMI/
├── code/         # All analysis code (Jupyter notebooks + one R script)
├── docs/         # Manuscript PDF and survey materials
├── plots/        # Final figures (main paper + supplementary)
└── requirements.txt
```

### `code/`
The full pipeline, organised top-down by stage. File prefixes encode `0[stage][substage][sequence]` — e.g. `0521_*` is stage 5 (validation), substage 2 (emi), file 1.

### `docs/`
- `writing/NHB.pdf` — the manuscript itself.
- `survey/` — consent forms, instructions, and the funding application that supported the validation surveys.
- `survey/example/` — a worked example of the Prolific survey flow (configs, templates, data files).

### `plots/`
- `main/` — figures used in the main paper (`parliament/`, `twitter/` subfolders for arena-specific plots, plus combined overlay and ROC).
- `supplementary/` — extended-data and supplementary-note figures (annotator demographics, bootstrap CIs, correlation matrices, density distributions, word-shift graphs, etc.).

---

## Code in detail

```
code/
├── 01_data/              # 1. Corpus construction
│   ├── parliament/
│   │   ├── 0111_get_speeches_new.ipynb
│   │   ├── 0112_get_speeches_old.ipynb
│   │   └── 0113_preprocess_speeches.ipynb
│   └── twitter/
│       └── 0121_twitter_data.ipynb
│
├── 02_models/            # 2. Embedding training
│   ├── 0201_w2v_base.ipynb
│   ├── 0202_w2v_twitter.ipynb
│   ├── 0203_delete_first_line.ipynb
│   └── 0204_w2v_to_sbert.ipynb
│
├── 03_scoring/           # 3. EMI scoring + topic modelling
│   ├── 0301_sbert_emi_speeches.ipynb
│   ├── 0302_sbert_emi_twitter.ipynb
│   └── 0303_topic_modelling.ipynb
│
├── 04_analysis/          # 4. Descriptive + regression analysis
│   ├── 0401_descriptives.ipynb
│   ├── 0402_group_plots.ipynb
│   └── 0403_analysis.ipynb
│
└── 05_validation/        # 5. Validation studies
    ├── document/
    │   ├── 0511_validation_sample.ipynb
    │   ├── 0512_process_survey_data.ipynb
    │   └── 0513_emi_validation.ipynb
    ├── emi/
    │   ├── 0521_corpus_analysis_speeches.ipynb
    │   ├── 0522_corpus_analysis_twitter.ipynb
    │   ├── 0523_bootstrapp_word.ipynb
    │   ├── 0524_bootstrapp_document.ipynb
    │   ├── 0525_bootstrapp_analysis.ipynb
    │   ├── 0526_negative_clipping.ipynb
    │   ├── 0527_dictionary_correlation.ipynb
    │   └── 0528_model_correlation.ipynb
    └── keywords/
        └── 0531_prodeminfo_kw_validation.R
```

### 01_data — corpus construction

**`parliament/`** — builds the contemporary Bundestag corpus (56,213 speeches, 18th–21st electoral terms, 2015–2025) and merges the historical corpus (1867–2021, ~329k speeches) used solely for embedding pre-training.

| Script | Description |
|---|---|
| `0111_get_speeches_new.ipynb` | Downloads and parses XML protocols from the official Bundestag document server for the 18th–21st electoral terms; extracts session metadata, speaker information, and full speech text. |
| `0112_get_speeches_old.ipynb` | Pulls the historical Bundestag corpus (Open Discourse + GPC 1867–1942) and standardises its schema. |
| `0113_preprocess_speeches.ipynb` | Cleans, deduplicates, and assigns unique IDs; merges metadata with party/leaning labels (CHES coding). |

**`twitter/`** — builds the contemporary Twitter corpus (4.5M tweets, 2,067 accounts, 2015–2025) by combining the Lasser et al. (2008–2021) German subsample with new tweets retrieved under DSA Article 40.

| Script | Description |
|---|---|
| `0121_twitter_data.ipynb` | Retrieves tweets via the X Academic API (DSA vetted-researcher access) for verified handles of 20th-Bundestag MPs (GESIS handle list), merges with the Lasser corpus, de-duplicates by tweet ID. |

### 02_models — embedding training

Trains the gensim word2vec backbone used by the DDR pipeline. The base model is pre-trained on the combined historical Bundestag corpus, then fine-tuned separately on the Twitter and contemporary Bundestag corpora to obtain platform-specific embeddings.

| Script | Description |
|---|---|
| `0201_w2v_base.ipynb` | Trains the base word2vec model on the combined historical Bundestag corpus (1867–2025). |
| `0202_w2v_twitter.ipynb` | Fine-tunes the base model on the Twitter corpus to produce platform-specific embeddings. |
| `0203_delete_first_line.ipynb` | Utility for stripping the gensim header line from saved KeyedVectors text dumps so they can be loaded by downstream tooling. |
| `0204_w2v_to_sbert.ipynb` | Wraps the trained word2vec vectors in a sentence-transformers-compatible format for use in the scoring pipeline. |

### 03_scoring — EMI scoring + topic modelling

Computes per-document EMI scores via DDR and fits BERTopic models for the topic-controlled robustness checks.

| Script | Description |
|---|---|
| `0301_sbert_emi_speeches.ipynb` | Computes EMI for every Bundestag speech: cosine similarity between the averaged speech embedding and the averaged evidence/intuition dictionary embeddings, z-scored across the corpus, and EMI = z(D_evidence) − z(D_intuition). |
| `0302_sbert_emi_twitter.ipynb` | Same DDR scoring pipeline applied to the 4.5M-tweet corpus. |
| `0303_topic_modelling.ipynb` | Fits a `BERTopic` model with `paraphrase-multilingual-mpnet-base-v2` embeddings, UMAP, and HDBSCAN; reduces to K = 30 topics per corpus and assigns each document a hard topic label for the topic-controlled regressions. |

### 04_analysis — descriptive + regression analysis

The empirical core of the paper: time-series, descriptives, and the three mixed-effects regression specifications (baseline, breakpoint, full breakpoint with leaning interactions) estimated with `statsmodels` MixedLM.

| Script | Description |
|---|---|
| `0401_descriptives.ipynb` | Corpus-level descriptives: speeches/tweets per year, party/leaning composition, EMI distributions, and basic temporal trends. |
| `0402_group_plots.ipynb` | Group-level plotting helpers: party- and leaning-stratified time series, density overlays, and the per-arena components (intuition similarity, evidence similarity, EMI). |
| `0403_analysis.ipynb` | Main regression pipeline: quarterly aggregation, baseline linear MixedLM with leaning × time interaction and lagged EMI, breakpoint model with piecewise-linear and level-shift terms at each election (2017, 2021, 2025), and the full breakpoint model with leaning-specific slope/level deviations. Model fit via likelihood-ratio, AIC/BIC, ICC, and Durbin–Watson. |

### 05_validation — validation studies

Three complementary validation tracks for the EMI measure.

**`document/`** — pre-registered Prolific survey (N = 284) where German speakers annotated 1,300 parliamentary passages on evidence, intuition, and EMI Likert scales. Used to estimate ICCs and document-level AUC.

| Script | Description |
|---|---|
| `0511_validation_sample.ipynb` | Draws the stratified sample of 1,300 passages (10–50 tokens each, equally across four EMI bins, balanced across time and party). |
| `0512_process_survey_data.ipynb` | Cleans Prolific exports: applies the attention check, removes failed participants (32 excluded → 252 valid), reshapes ratings to long format. |
| `0513_emi_validation.ipynb` | Computes single-rater and average ICCs, ROC/AUC against majority-vote ground truth (AUC = 0.59 evidence, 0.68 intuition, 0.72 EMI), and produces the validation figures. |

**`emi/`** — internal validation of the EMI pipeline: corpus diagnostics, bootstrap robustness, negative-score handling, and cross-dictionary / cross-model correlations.

| Script | Description |
|---|---|
| `0521_corpus_analysis_speeches.ipynb` | Token, length, and EMI-distribution diagnostics for the Bundestag corpus; word-keyness grids and word-shift graphs. |
| `0522_corpus_analysis_twitter.ipynb` | Same diagnostics for the Twitter corpus. |
| `0523_bootstrapp_word.ipynb` | Word-level bootstrap: 1,000 resamples of the dictionary composition, recomputes EMI, and tracks per-word contribution stability. |
| `0524_bootstrapp_document.ipynb` | Document-level bootstrap: confirms document-level EMI scores are not driven by a small subset of influential words. |
| `0525_bootstrapp_analysis.ipynb` | Aggregates the bootstrap runs into the dashed-CI plots used in the supplementary materials. |
| `0526_negative_clipping.ipynb` | Sensitivity analysis for clipping negative cosine similarities (alternative DDR variants). |
| `0527_dictionary_correlation.ipynb` | Correlations across alternative dictionary specifications, broken out by document length and arena. |
| `0528_model_correlation.ipynb` | Robustness check using publicly available generic German fastText embeddings: r = 0.93 between gensim- and fastText-based document-level EMI scores. |

**`keywords/`** — statistical validation of the keyword-level survey (separate dictionary survey, N = 47).

| Script | Description |
|---|---|
| `0531_prodeminfo_kw_validation.R` | R script that runs paired t-tests of intuition vs. evidence ratings per candidate keyword with Holm-corrected p-values; retains 86 of 113 candidate keywords. |

---

## Setup

Analyses were run with **Python 3.9.1**. Install dependencies into a fresh virtualenv:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The R script (`0531_prodeminfo_kw_validation.R`) requires R ≥ 4.0 with `tidyverse`, `rstatix`, and `effsize`.

## Data availability

The validated German evidence and intuition dictionaries, the contemporary Bundestag speech dataset, the tweet IDs of the Twitter corpus (in line with the X Developer Agreement), the survey response data, and all aggregated time-series used in the figures and regressions are deposited on the **Open Science Framework** at https://osf.io/8tk9q/. Raw tweet content is restricted by the X Developer Agreement and the DSA Article 40 framework.

## Citation

If you use this code, please cite the paper (DOI to be assigned upon acceptance) and:
- Lasser et al. (2023) — original English EMI dictionaries.
- Aroyehun et al. — DDR pipeline.
- Open Discourse and Abrami et al. — historical Bundestag corpus.

## License

MIT.
