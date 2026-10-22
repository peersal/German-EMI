"""LLM ratings vs human ratings vs dictionary EMI on the 2026 validation surveys (300 Bundestag chunks, 300 tweets).
Reads output/survey_rating_<model>.parquet (and survey_use_mention_* if present) from the cluster job and
inputs/survey_meta.csv. Writes results/survey2026/{llm_vs_human.csv,numbers.json} and prints the comparison.

  python code/05_validation/llm/0572_analyze_survey.py
"""
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))   # German-EMI repository root: data in data/, outputs in results/
import os, glob, json, numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__)); OUTD = os.path.join(REPO_ROOT, 'results', 'llm', 'survey2026'); os.makedirs(OUTD, exist_ok=True)
M = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'llm', 'inputs', 'survey_meta.csv'), dtype={'id': str})
M['human_emi'] = M.ev_mean - M.iu_mean
TAGS = {'Llama': 'meta-llama_Llama-3.1-8B-Instruct', 'Qwen': 'Qwen_Qwen2.5-7B-Instruct'}


def load(prompt):
    out = {}
    for name, tag in TAGS.items():
        f = os.path.join(REPO_ROOT, 'data', 'llm', 'output', f'survey_{prompt}_{tag}.parquet')
        if os.path.exists(f):
            d = pd.read_parquet(f); d.index = d.index.astype(str); d.index.name = 'id'; out[name] = d
    return out


def auc(score, lab):
    m = pd.concat([score, lab], axis=1).dropna(); m = m[m.iloc[:, 1] != 0]
    return (roc_auc_score((m.iloc[:, 1] > 0).astype(int), m.iloc[:, 0]), len(m)) if m.iloc[:, 1].nunique() == 2 else (np.nan, len(m))


R = load('rating'); rows = []
for corpus, g in M.set_index('id').groupby('corpus'):
    g = g.dropna(subset=['human_emi'])
    comp = {}
    for name, d in R.items():
        d = d.reindex(g.index); comp[name] = d['evidenz_basiert'] - d['intuition_basiert']
        for construct, col, human in [('evidence', d['evidenz_basiert'], g.ev_mean), ('intuition', d['intuition_basiert'], g.iu_mean), ('EMI', comp[name], g.human_emi)]:
            rho = stats.spearmanr(col, human, nan_policy='omit'); a, n = auc(col, np.sign(human))
            rows.append(dict(corpus=corpus, rater=name, construct=construct, spearman=rho.correlation, auc=a, n=n))
    if len(comp) == 2:
        mean_c = (comp['Llama'] + comp['Qwen']) / 2
        for construct, col, human in [('EMI', mean_c, g.human_emi)]:
            rho = stats.spearmanr(col, human, nan_policy='omit'); a, n = auc(col, np.sign(human))
            rows.append(dict(corpus=corpus, rater='mean of both', construct=construct, spearman=rho.correlation, auc=a, n=n))
        rows.append(dict(corpus=corpus, rater='inter-model', construct='EMI', spearman=stats.spearmanr(comp['Llama'], comp['Qwen'], nan_policy='omit').correlation, auc=np.nan, n=int(comp['Llama'].notna().sum())))
        rows.append(dict(corpus=corpus, rater='LLM mean vs dictionary', construct='EMI', spearman=stats.spearmanr(mean_c, g.emi, nan_policy='omit').correlation, auc=np.nan, n=len(g)))
    rho = stats.spearmanr(g.emi, g.human_emi, nan_policy='omit'); a, n = auc(g.emi, np.sign(g.human_emi))
    rows.append(dict(corpus=corpus, rater='dictionary EMI', construct='EMI', spearman=rho.correlation, auc=a, n=n))
    # by party for the composite and the dictionary
    if len(comp) == 2:
        for party, gp in g.groupby('party'):
            a_l, _ = auc(mean_c.reindex(gp.index), np.sign(gp.human_emi)); a_d, _ = auc(gp.emi, np.sign(gp.human_emi))
            rows.append(dict(corpus=corpus, rater=f'by party: {party}', construct='EMI', spearman=np.nan, auc=a_l, n=len(gp), auc_dictionary=a_d))
T = pd.DataFrame(rows); T.to_csv(os.path.join(OUTD, 'llm_vs_human.csv'), index=False)
pd.set_option('display.width', 200); print(T.round(3).to_string(index=False))
json.dump({f'{r.corpus}|{r.rater}|{r.construct}': dict(spearman=None if pd.isna(r.spearman) else round(float(r.spearman), 3), auc=None if pd.isna(r.auc) else round(float(r.auc), 3), n=int(r.n)) for r in T.itertuples()},
          open(os.path.join(OUTD, 'numbers.json'), 'w'), indent=1)
print('written', OUTD)
