# German-EMI — Demo

This folder contains a small, self-contained demo of the statistical
analysis pipeline used in *"Rhetorical convergence toward intuition-based
discourse following the rise of the far right in Germany"*. It fits the
paper's two core regression specifications on small real subsets of the
two corpora and produces model summaries plus figures of the quarterly
mean EMI (Evidence Minus Intuition) score per ideological leaning:

1. **Bundestag** — the constrained breakpoint linear mixed-effects
   regression (Methods, Equation 2).
2. **Twitter** — the baseline linear mixed-effects model (Methods,
   Equation 1; the per-tweet engagement covariate is omitted because
   engagement counts are not included in the shared file).

### Why the demo uses precomputed EMI scores

The upstream EMI scoring pipeline — training the domain-specific
word2vec embedding model on the historical Bundestag corpus (1867–2025),
fine-tuning it on the contemporary Bundestag and Twitter corpora, and
computing the DDR (Distributed Dictionary Representation) scores for all
~6.3 million documents — was carried out on a dedicated GPU compute node
(NVIDIA H100) and requires the full corpora and trained embedding
models. It is therefore **not part of this demo**. The demo instead
ships precomputed document-level EMI scores and demonstrates the
downstream statistical analysis, which is what produces the results
reported in the paper. The scoring pipeline itself is documented in the
paper's Methods and Supplementary Note 2, and its code is included in
the main repository.

## Contents

| File | Description |
|---|---|
| `demo_data.csv` | 2,932 Bundestag speeches (2015–2025) by 40 randomly sampled politicians, stratified by ideological leaning. Columns: `id`, `date`, `politicianId`, `party`, `leaning`, `emi_w2vparliament` (precomputed EMI score). Speech texts are omitted; the underlying plenary protocols are publicly available from the Bundestag document server. |
| `demo_data_twitter.csv` | 6,000 tweets (2015–2025) by 40 randomly sampled politician accounts, stratified by leaning. Columns: `id` (tweet ID), `actor` (pseudonymised account code), `date`, `party`, `leaning`, `emi_w2vparliament`. In line with the X Developer Agreement, **only tweet IDs are shared** — no tweet text, no user names, no raw author IDs, no engagement counts. Tweets can be rehydrated from their IDs via the X API. |
| `run_demo.py` | Self-contained demo script: builds the lagged quarterly autoregressive term and (for the Bundestag) the election-breakpoint regressors, fits both mixed models, writes outputs. |
| `requirements.txt` | Python dependencies. |

## 1. System requirements

- **Operating system:** any OS with Python ≥ 3.9 (tested on Windows 11
  Pro (10.0.26200) and Ubuntu 22.04).
- **Python:** ≥ 3.9 (tested on Python 3.13.0).
- **Dependencies** (installed via `requirements.txt`; versions the demo
  was tested with in parentheses):
  - numpy (2.3.5)
  - pandas (2.3.3)
  - scipy (1.16.3)
  - statsmodels (0.14.5)
  - matplotlib (3.10.7)
- **Hardware:** no non-standard hardware is required for the demo; any
  desktop or laptop with ≥ 2 GB RAM is sufficient. (Only the upstream
  embedding-model training and corpus-wide EMI scoring — not included
  here — used a GPU; see above.)

## 2. Installation guide

```bash
git clone https://github.com/peersal/German-EMI.git
cd German-EMI/demo
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

**Typical install time** on a normal desktop computer: 1–2 minutes
(dominated by downloading the scientific Python wheels).

## 3. Demo

Run:

```bash
python run_demo.py
```

**Expected output.** The script prints, for each corpus, the number of
documents loaded, the fitted mixed-model coefficient table and the
intraclass correlation of the null model, and writes four files to
`demo_output/`:

- `model_summary_parliament.txt`, `model_summary_twitter.txt` — full
  `statsmodels` MixedLM regression tables.
- `emi_trends_parliament.png`, `emi_trends_twitter.png` — quarterly mean
  EMI per ideological leaning (Left / Centre / Right) with dashed
  vertical lines at the three election breakpoints (2017-07-01,
  2021-07-01, 2025-01-01).

On these small subsamples the models reproduce the qualitative patterns
of the paper. Bundestag: negative level shifts at the 2017, 2021 and
2025 breakpoints (e.g. `level_elec2017` ≈ −0.28, *P* ≈ 0.01), a positive
and significant lagged-EMI term (≈ 0.12, *P* < 0.001), null-model ICC
≈ 0.34. Twitter: substantially lower baseline EMI for right-leaning
accounts (`leaning[T.right]` ≈ −0.33, *P* < 0.001) and a small negative
overall time trend (≈ −0.003 per quarter, *P* < 0.001). Exact values
will differ from the paper, which uses all 59,170 speeches and 4.5
million tweets; coefficient signs and the general pattern should match.

**Expected run time** on a normal desktop computer: < 5 seconds
(1.0 s for both parts on the test machine).

## 4. Instructions for use

To run the analysis on the full data, obtain the complete datasets from
the OSF repository referenced in the paper (`https://osf.io/x3zpc/`) and
use the main analysis notebook (`analysis.ipynb`) in the repository
root. The notebook expects CSV files with, at minimum, the columns used
here (date, actor identifier, `party`, `leaning`, and an EMI score
column) and reproduces all models, tables and figures of the paper: the
baseline linear model (Twitter), the constrained and leaning-interacted
breakpoint models (Bundestag), rolling-window robustness fits, and the
topic-controlled specifications.

`run_demo.py` also works unchanged on any CSVs with the same column
layouts as the two demo files — point the `pd.read_csv` calls at your
files.

## (Optional) Reproduction instructions

1. Download the corpora and precomputed scores from the OSF repository.
   (Tweet texts must be rehydrated from the shared tweet IDs via the
   X API, in line with the X Developer Agreement.)
2. Recompute EMI scores from raw text (optional; GPU recommended):
   train/fine-tune the word2vec models and apply the DDR scoring
   pipeline as described in the paper's Methods and Supplementary
   Note 2.
3. Run `analysis.ipynb` top to bottom; per-dataset outputs (model
   pickles, LaTeX tables, figures) are written to `results/<dataset>/`.
   Fitting the full Bundestag breakpoint models takes a few minutes;
   the Twitter models (6.2 M tweets) take on the order of 30–60 minutes
   on a normal desktop computer.
