"""Krippendorff's alpha (nominal) between the two LLMs on the four-category coding of the 10,000-sentence sample
(output/sentsample_category_<model>.parquet), with percent agreement, confusion matrix and label shares; no human coders.
Run from code/:  .venv/Scripts/python.exe code/05_validation/llm/0557_krippendorff_alpha.py
"""
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))   # German-EMI repository root: data in data/, outputs in results/
import os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = {'meta-llama_Llama-3.1-8B-Instruct': 'Llama', 'Qwen_Qwen2.5-7B-Instruct': 'Qwen'}
def alpha_nominal(a, b):
    """Krippendorff's alpha for two coders, nominal data, pairable units only."""
    m = pd.notna(a) & pd.notna(b); a, b = np.asarray(a[m], dtype=int), np.asarray(b[m], dtype=int); n = len(a)
    if n == 0: return np.nan
    vals = np.unique(np.concatenate([a, b])); idx = {v: i for i, v in enumerate(vals)}; k = len(vals)
    O = np.zeros((k, k))
    for x, y in zip(a, b): O[idx[x], idx[y]] += 1; O[idx[y], idx[x]] += 1
    nc = O.sum(axis=1); N = nc.sum()
    Do = sum(O[i, j] for i in range(k) for j in range(k) if i != j)
    De = sum(nc[i] * nc[j] for i in range(k) for j in range(k) if i != j) / (N - 1)
    return 1 - Do / De if De > 0 else np.nan
sheet = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'llm', 'inputs', 'sentsample_meta.csv'), dtype={'id': str})
for tag, short in MODELS.items():
    f = os.path.join(REPO_ROOT, 'data', 'llm', 'output', f'sentsample_category_{tag}.parquet')
    if os.path.exists(f):
        d = pd.read_parquet(f); d = d.reset_index() if 'id' not in d.columns else d; idc = [c for c in d.columns if c not in ('category', 'parse_error') and not c.endswith('_missing')][0]
        sheet = sheet.merge(d[[idc, 'category']].rename(columns={idc: 'id', 'category': short}).assign(id=lambda x: x.id.astype(str)), on='id', how='left')
raters = [c for c in ['Llama', 'Qwen'] if c in sheet and sheet[c].notna().any()]
print(f'{len(sheet)} sentences; raters with data: {raters}')
rows = []
for i, r1 in enumerate(raters):
    for r2 in raters[i + 1:]:
        m = sheet[r1].notna() & sheet[r2].notna(); rows.append(dict(pair=f'{r1} vs {r2}', n=int(m.sum()), alpha=alpha_nominal(sheet[r1], sheet[r2]), agreement=(sheet.loc[m, r1] == sheet.loc[m, r2]).mean()))
R = pd.DataFrame(rows); print(R.round(3).to_string(index=False)); R.to_csv(os.path.join(REPO_ROOT, 'results', 'llm', 'krippendorff_alpha.csv'), index=False)
if len(raters) == 2:
    print('\nconfusion Llama (rows) vs Qwen (cols):\n' + pd.crosstab(sheet.Llama, sheet.Qwen).to_string())
    print('\nalpha by leaning:', {l: round(alpha_nominal(g.Llama, g.Qwen), 3) for l, g in sheet.groupby('leaning')})
    print('alpha by year:', {int(y): round(alpha_nominal(g.Llama, g.Qwen), 3) for y, g in sheet.groupby('year')})
for s in [c for c in ['Llama', 'Qwen'] if c in raters]:
    for c in [c for c in ['coder_1', 'coder_2'] if c in raters]: print(f'\nconfusion {c} (rows) vs {s} (cols):\n' + pd.crosstab(sheet[c], sheet[s]).to_string())
print('\nlabel shares:', {r: sheet[r].value_counts(normalize=True).sort_index().round(3).to_dict() for r in raters})
