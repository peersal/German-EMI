"""Anonymised single ratings of the 2024 document-level survey for the public repository.
Same attention-check rule and rating map as revision.ipynb section 7; Prolific ids replaced by sequential codes
(a2024_r001, ...); no texts, timestamps or demographics. Writes results/13_validation_2026/data/ratings_parliament_2024.csv
and rater_quality_parliament_2024.csv, and checks the item means against groundtruth_parliament_2024.csv.

  python code/05_validation/document/0518_export_ratings_2024.py
"""
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))   # German-EMI repository root: data in data/, outputs in results/
import os, glob, pandas as pd, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = REPO_ROOT
VD = os.path.join(REPO_ROOT, 'data', 'validation', 'unzipped'); DATA = os.path.join(REPO_ROOT, 'results', 'revision', '13_validation_2026', 'data')
EV, IU = 'Rating:::Evidenz basiert', 'Rating:::Intuition basiert'
ann = pd.concat([pd.read_csv(f, dtype=str) for f in sorted(glob.glob(os.path.join(VD, 'annotated_instances*.csv')))], ignore_index=True)
n_rows_raw = len(ann)
ann = ann.drop_duplicates(['user', 'instance_id', EV, IU])                 # the export files overlap
ok = {'Stimme voll und ganz zu', 'Stimme zu', 'Neutral'}
failed = set(ann[((ann.instance_id == '1_testing') & ~ann[IU].isin(ok)) | ((ann.instance_id == '0_testing') & ~ann[EV].isin(ok))].user)
users = sorted(ann.user.dropna().unique()); rid = {u: f'a2024_r{i + 1:03d}' for i, u in enumerate(users)}
rmap = {'Stimme überhaupt nicht zu': -2, 'Stimme nicht zu': -1, 'Neutral': 0, 'Stimme zu': 1, 'Stimme voll und ganz zu': 2}
items = ann[~ann.instance_id.isin(['0_testing', '1_testing', 'EMI-consent.html'])].copy()
items['evidence_rating'] = items[EV].map(rmap); items['intuition_rating'] = items[IU].map(rmap)
items = items.dropna(subset=['evidence_rating', 'intuition_rating'], how='all')
items['rater'] = items.user.map(rid); items['attention_check_passed'] = (~items.user.isin(failed)).astype(int)
out = items[['instance_id', 'rater', 'evidence_rating', 'intuition_rating', 'attention_check_passed']].rename(columns={'instance_id': 'id'}).sort_values(['id', 'rater'])
out.to_csv(os.path.join(DATA, 'ratings_parliament_2024.csv'), index=False)
q = pd.DataFrame({'rater': [rid[u] for u in users], 'attention_check_passed': [int(u not in failed) for u in users]})
q = q.merge(out.groupby('rater').size().rename('n_items').reset_index(), on='rater', how='left').fillna({'n_items': 0})
q['n_items'] = q.n_items.astype(int); q.to_csv(os.path.join(DATA, 'rater_quality_parliament_2024.csv'), index=False)
# consistency with the published item means (raters who passed the check)
gt = pd.read_csv(os.path.join(DATA, 'groundtruth_parliament_2024.csv'), dtype={'id': str}).set_index('id')
m = out[out.attention_check_passed == 1].groupby('id').agg(ev=('evidence_rating', 'mean'), iu=('intuition_rating', 'mean'), n=('rater', 'size')).join(gt[['ev_mean', 'iu_mean', 'n_raters']], how='inner')
print(f'raw rows {n_rows_raw}, after removing overlapping duplicates {len(ann)}; raters {len(users)}, failed {len(failed)}; item ratings {len(out)} ({int(out.attention_check_passed.sum())} from valid raters), items {out.id.nunique()}')
print(f'items matched to ground truth {len(m)}; max |diff| evidence {np.abs(m.ev - m.ev_mean).max():.3f}, intuition {np.abs(m.iu - m.iu_mean).max():.3f}; n_raters equal on {(m.n == m.n_raters).mean():.1%}')
