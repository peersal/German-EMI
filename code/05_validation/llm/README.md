# LLM annotation (validation, use vs. mention, other-directed speech)

Two open-weight instruction models (`meta-llama/Llama-3.1-8B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`, served with
vLLM 0.11.0, temperature 0, one GPU) rate passages on the two items of the human validation survey and code
sentences as self- or other-directed. The ratings are used to (1) validate the embedding-based EMI against LLM
ratings and the LLM ratings against the human ground truth, (2) separate *using* evidence or intuition from
*talking about* it, and (3) re-score the Bundestag corpus without other-directed sentences.

The GPU job files of the cluster we used are not published; any OpenAI-compatible vLLM endpoint reproduces the
ratings with the prompts in `prompts/` (system prompt + JSON concept file per task, one request per text,
`{"concept": score}` answers written to one parquet file per model x prompt in `data/llm/output/`).

| Script | What it does | Reads | Writes |
|---|---|---|---|
| `0551_build_inputs.py` | ground truth (1,142 survey chunks with human means and EMI) and a 10k-speech chunk sample | `data/validation/`, `results/revision/07_validation_auc/` | `data/llm/inputs/` |
| `0552_build_corpus_input.py` | one chunk per speech for the whole Bundestag corpus | `data/parliament_nhb.csv` | `data/llm/inputs/corpus_*` |
| `0553_build_twitter_input.py` | tweet sample for the Twitter time series | `data/twitter_nhb.csv` | `data/llm/inputs/twitter_*` |
| `0554_build_sentence_inputs.py` | sentence-level input for the other-directed coding | `data/parliament_nhb.csv` | `data/llm/inputs/sentences_*` |
| `0555_analyze.py` | A: LLM vs human; B: EMI vs LLM; C: use vs mention by party; D: by period | `data/llm/output/` | `results/llm/*.csv` |
| `0556_timeseries.py` | quarterly series and breakpoint models on the LLM ratings (`DS=corpus` or `DS=twitter`) | `data/llm/output/` | `results/llm/{parliament,twitter}/` |
| `0557_krippendorff_alpha.py` | agreement of the two models on the sentence categories | `data/llm/output/` | `results/llm/krippendorff_alpha.csv` |
| `0558_sentence_categories.py` | share of other-directed sentences over time, re-scored EMI without them | `data/llm/output/`, `data/parliament_model/` | `results/llm/categories/` |
| `0559_figures_llm.py` | figures for the supplement | `results/llm/` | `results/llm/figures/` |

Prompts: `prompts/system_rating.txt` + `concepts_rating.json` (the two survey items, 0-4), `system_use_mention.txt`
+ `concepts_use_mention.json` (evidence/intuition use vs. mention, procedural), `system_category.txt` +
`concepts_category.json` (self-directed, other-directed, procedural, other). The human survey used a 5-point
agreement scale; the analysis compares ranks, so no mapping is needed.
