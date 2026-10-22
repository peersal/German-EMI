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
├── demo/         # Small self-contained demo of the statistical analysis
├── docs/         # Manuscript PDF and survey materials
├── plots/        # Final figures (main paper + supplementary + revision)
├── results/      # Small result tables of the revision scripts (see results/README.md)
├── LICENSE
└── requirements.txt
```

The scripts of the 2026 revision expect the OSF data in `data/` (not in the repository) and write to `results/`; see [Revision analyses (2026)](#revision-analyses-2026).

### `code/`
The full pipeline, organised top-down by stage. File prefixes encode `0[stage][substage][sequence]` — e.g. `0521_*` is stage 5 (validation), substage 2 (emi), file 1.

### `demo/`
A small, self-contained demo of the downstream statistical analysis: it fits the paper's two core mixed-effects specifications — the constrained breakpoint model on 2,932 Bundestag speeches and the baseline model on 6,000 tweets — using precomputed EMI scores, and writes the regression tables and quarterly EMI trend figures to `demo_output/`. Runs in a few seconds on any machine with Python ≥ 3.9; no GPU or full corpora needed. See [`demo/README.md`](demo/README.md) for system requirements, installation, and expected output.

### `docs/`
- `writing/NHB.pdf` — the manuscript itself.
- `survey/` — consent forms, instructions, and the funding application that supported the validation surveys.
- `survey/example/` — a worked example of the Prolific survey flow (configs, templates, data files).
- `survey/revision/` — the potato projects of the two new validation surveys of the revision (Bundestag passages, tweets) and the surveyflow of the original 2024 survey.

### `plots/`
- `main/` — figures used in the main paper (`parliament/`, `twitter/` subfolders for arena-specific plots, plus combined overlay and ROC).
- `supplementary/` — extended-data and supplementary-note figures (annotator demographics, bootstrap CIs, correlation matrices, density distributions, word-shift graphs, etc.).
- `revision/` — figures of the 2026 revision (cross-platform figure with the AfD-entry split, contagion figures, composition, LLM annotation); `plots/revision/README.md` maps each file to its script.

---

## Code in detail

```
code/
├── 01_data/              # 1. Corpus construction
│   ├── parliament/
│   │   ├── 0111_get_speeches_new.ipynb
│   │   ├── 0112_get_speeches_old.ipynb
│   │   ├── 0113_preprocess_speeches.ipynb
│   │   ├── 0114_extract_bundestag_2000.py      (revision)
│   │   └── 0115_build_input_2000_2025.py       (revision)
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
│   ├── 0303_topic_modelling.ipynb
│   └── 0304_score_emi.py                       (revision: scoring module reused by the revision scripts)
│
├── 04_analysis/          # 4. Descriptive + regression analysis
│   ├── 0401_descriptives.ipynb
│   ├── 0402_group_plots.ipynb
│   ├── 0403_analysis.ipynb                     (updated in the revision)
│   ├── 0404_revision_analysis.ipynb            (revision)
│   ├── 0405_composition_stayers.py             (revision)
│   ├── 0406_run_figure_cells.py                (revision)
│   ├── 0407_xplat_figure_cell.py, 0408_xplat_run_cell.py, 0409_xplat_split2000_cell.py
│   ├── contagion/   0411–0426                  (revision: within-sitting contagion)
│   └── mechanisms/  0431–0436                  (revision: mechanism tests)
│
└── 05_validation/        # 5. Validation studies
    ├── document/
    │   ├── 0511_validation_sample.ipynb
    │   ├── 0512_process_survey_data.ipynb
    │   ├── 0513_emi_validation.ipynb
    │   └── 0514–0516 *.py                      (revision: validation by era and party)
    ├── topics/      0541, 0542                  (revision: topic controls on identical samples)
    ├── llm/         0551–0559 + prompts/        (revision: LLM annotation)
    ├── survey/      0561, 0562                  (revision: new validation surveys)
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

Trains the gensim word2vec backbone used by the DDR pipeline. The base model is pre-trained on the combined historical Bundestag corpus, then fine-tuned separately on the Twitter and contemporary Bundestag corpora to obtain platform-specific embeddings. Model training and embedding computation were run on a dedicated compute node with an NVIDIA H100 GPU, with a total runtime under 8 hours.

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
| `0403_analysis.ipynb` | Main regression pipeline: quarterly aggregation, baseline linear MixedLM with leaning × time interaction and lagged EMI, breakpoint model with piecewise-linear and level-shift terms at each election (2017, 2021, 2025), and the full breakpoint model with leaning-specific slope/level deviations. Model fit via likelihood-ratio, AIC/BIC, ICC, and Durbin–Watson. Updated in the revision: cross-platform figure and per-dataset figures with the before/after split at the AfD's entry (24 Oct 2017), density panels, the 2000–2025 comparison and the within-sitting contagion cell. |

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

## Revision analyses (2026)

Analyses added in the revision of the paper. All scripts run from the repository root with the OSF data in `data/` (`data/parliament_nhb.csv`, `data/twitter_nhb.csv`, `data/validation/`, the parliament word2vec model in `data/parliament_model/`) and write to `results/`; intermediate files they create go to `data/revision/` and `data/llm/`. Every script resolves paths from its own location, so `python code/04_analysis/0405_composition_stayers.py` works from anywhere. The notebooks are stored without outputs and change to the repository root in their first cell.

**Bundestag corpus 2000–2025 and composition of the shift**

| Script | Description | Output |
|---|---|---|
| `01_data/parliament/0114_extract_bundestag_2000.py` | Extracts all plenary speeches from January 2000 from the Open Discourse record (`data/full_parliament/speeches_table_Germany.json`). | `data/revision/bundestag_speeches_2000_2025_raw.csv` |
| `01_data/parliament/0115_build_input_2000_2025.py` | Joins the paper's 2015–2025 speeches with the 2000–2014 extension and applies the paper's preprocessing. | `data/revision/bundestag_2000_2025_input.csv` |
| `03_scoring/0304_score_emi.py` | Re-scores texts with the paper's parliament embedding model (document vectors, length-bin adjustment, z-scores); CLI (`--input/--text/--id/--output`) and module used by the other scripts. Preprocessing follows `0113`; the stop-word lists are stored in `03_scoring/german_stopwords.txt` (speeches: NLTK German list plus 15 words) and `03_scoring/twitter_stopwords.txt` (tweets, 496 words). | `data/revision/bundestag_2000_2025_scored.csv` |
| `04_analysis/0404_revision_analysis.ipynb` | Section-by-section revision notebook: cohort analysis, 2000–2025 series with shifts at all seven elections, placebo breakpoints and single-break search, validation by era and party, topic controls, ideological distance and constituency exposure, mechanism tests, contagion summary. One `numbers.json` per section. | `results/revision/<section>/` |
| `04_analysis/0405_composition_stayers.py` | Shift-share decomposition into stayers / entrants / leavers, paired within-politician change, politician fixed-effects and balanced-panel breakpoint models. `STAYERS_SPLIT=2017-10-24` splits at the AfD's entry (default 2020-01-01). | `results/parliament/stayers_afd/`, `results/parliament/stayers/` |
| `04_analysis/0406_run_figure_cells.py` | Runs the figure cells of `0403_analysis.ipynb` outside Jupyter: `fig` (cross-platform figure, uses `0407`/`0408`), `rq1`, `v2`, `split`, `split2000` (`0409`, before/after the AfD's entry on the 2000–2025 corpus), `contagion` (`contagion/0411`). | `results/emi_twitter_parliament.pdf`, `results/<dataset>/plots/`, `results/parliament/tables/temporal_comparison_*` |

**Within-sitting contagion (`04_analysis/contagion/`)**

| Script | Description |
|---|---|
| `0411_contagion_sequence.py` | Core analysis: speech sequences per sitting, contagion effect (next minus previous speech by another party), k-windows, permutation test, within-speaker models, dose / year / naming splits, party reply network, MdB mention network, example sitting. Tables in `results/parliament/tables/contagion_*.csv`. |
| `0412_contagion_matching.py`, `0413_contagion_spec_curve.py` | Matched and adjusted estimates of the reaction to an AfD speech (OLS, speaker FE, coarsened exact matching, IPW, nearest neighbour) and a 62-specification curve. |
| `0414_reply_direction.py` | Within-speaker, position-adjusted reply levels for every ordered party pair (`results/parliament/tables/reply_direction_matrix*.csv`). |
| `0415`–`0419` | Earlier versions of the local-window analysis, placebo parties, position and climate controls, naming the AfD, share of the 2017 shift attributable to confrontation (`results/revision/12_afd_sequence/`). |
| `0421_figure_contagion_main.py` | Main-text contagion figure (reply network, k-window effect by party, example sitting). |
| `0422`–`0426` | Further figures: robustness ladder, network with values, within-speaker matrix / hub, 2000–2025 sequence overview, window effects with permutation distribution. |

**Mechanism tests (`04_analysis/mechanisms/`)**

| Script | Description |
|---|---|
| `0431_mechanism_tests.py` | 2021 transition, adoption hazard, long-run drift, similarity to AfD language, debate contact, party wings, 2013 placebo, lead–lag (`results/revision/10_mechanism_tests/`). |
| `0432_ideology_level_shock.py`, `0433_exposure_finegrained.py`, `0434_build_constituency_exposure.py` | Ideological distance to the AfD (Breunig & Guinaudeau 2025 positions, `data/revision/breunig_guinaudeau/`) and constituency AfD vote share (Bundeswahlleiter `btw17_kerg.csv`, `MDB_STAMMDATEN.XML`) as predictors of the post-2017 shift (`results/revision/09_mechanism_2017/`). |
| `0435_scrape_kandidierendencheck.py`, `0436_kandidierendencheck_tests.py` | abgeordnetenwatch Kandidierenden-Check thesis answers as an alternative position measure. |

**Validation (`05_validation/`)**

| Script | Description |
|---|---|
| `document/0514_validation_correlations.py`, `0515_validation_linke_diagnostics.py`, `0516_lookup_validation_parties.py` | Document-level validation (AUC, correlations) by era and party; diagnostics for the Left (`results/revision/07_validation_auc/`). |
| `topics/0541_topic_parliament_same_sample.py`, `0542_topic_twitter_same_sample.py` | Topic controls on identical samples, noise cluster kept as its own category (`results/revision/08_topic_controls/`). |
| `llm/0551`–`0559`, `llm/prompts/` | LLM annotation with Llama 3.1 8B and Qwen 2.5 7B (vLLM): validation against the human ratings, use vs. mention of evidence, other-directed sentences and re-scored EMI, time series on the LLM ratings; see `llm/README.md` (`results/llm/`). |
| `survey/0561_build_survey_samples.py`, `0562_make_survey_files.py` | Draw the 300 Bundestag passages and 300 tweets of the new validation surveys and write the potato projects in `docs/survey/revision/`. |
| `document/0517_validation_2026.py` + `document/data_2026/` | Document-level validation of the dictionary EMI against the human ratings of the 2024 survey (1,142 chunks, all eras), the 2026 Bundestag survey (300 chunks, 2015–2025) and the 2026 Twitter survey (300 tweets): AUC and Spearman ρ pooled, by era / period and by party, inter-rater reliability, LLM composite on the same items, annotator demographics. `data_2026/` holds the anonymised ground truth (item means, single ratings with sequential rater codes, no Prolific ids, no tweet texts); outputs in `results/validation_2026/`. |
| `llm/0571_build_survey_input.py`, `0572_analyze_survey.py` | LLM ratings of the 600 survey items of 2026 and their agreement with the human ratings (`results/llm/survey2026/`). |

**Words and elections (`04_analysis/words/`, `04_analysis/elections/`)**

| Script | Description |
|---|---|
| `words/0451_word_influence.py` | Word-level contributions to the DDR EMI per party (or camp) and period: word score $s_w$ = cosine to the evidence minus cosine to the intuition dictionary, token shares, word-shift contributions (`results/word_influence/`; `WI_GROUP=leaning WI_WINDOWS=biennial` for the timeline). |
| `words/0452_figure_word_influence.py`, `0453_figure_word_timeline.py` | Figures: most influential evidence- and intuition-type words per party and period, and the timeline per camp with callouts. |
| `elections/0461_breakpoint7_by_party.py` | Seven-election breakpoint model and pre/post comparison per party, Bundestag 2000–2025 (`results/revision/02_pre2015/breakpoint7_by_party.csv`). |
| `elections/0462_figure_afd_entry_supp.py` | Supplementary figures on the 2017 election: rolling series 2000–2025, level shifts at the seven elections, single-break search against placebo breaks, recovery in event time. |

The demo (`demo/run_demo.py`, Part 3) contains a reduced version of the composition and before/after analyses.

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

MIT — see [LICENSE](LICENSE).
