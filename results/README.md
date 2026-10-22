# Results

Small output tables and figures of the revision scripts, in the paths the scripts write to:

| Folder | Produced by |
|---|---|
| `revision/01_cohort_stayers/` … `revision/12_afd_sequence/` | `code/04_analysis/0404_revision_analysis.ipynb` and the scripts it calls (one folder per section, each with a `numbers.json`) |
| `revision/figures/` | figure scripts in `code/04_analysis/contagion/` |
| `llm/` | `code/05_validation/llm/` |
| `parliament/stayers_afd/`, `parliament/stayers/` | `code/04_analysis/0405_composition_stayers.py` (split at the AfD's entry / at 2020) |
| `parliament/tables/contagion_*`, `parliament/tables/temporal_comparison_*` | `code/04_analysis/0406_run_figure_cells.py contagion` / `split` / `split2000` |
| `cross_platform_split_afd.csv` | `code/04_analysis/0406_run_figure_cells.py fig` |
| `validation_2026/` | `code/05_validation/document/0517_validation_2026.py` (AUC tables, reliability, demographics, figures) |
| `word_influence/` | `code/04_analysis/words/0451_word_influence.py` (word-level contributions per party / camp and period) |
| `llm/survey2026/` | `code/05_validation/llm/0572_analyze_survey.py` |

Files larger than 2 MB (speech-level tables, caches, model pickles, logs) are not included; the scripts recreate them
from the OSF data.
