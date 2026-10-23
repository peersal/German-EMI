"""Does the Bundestag's switch to XML protocols in the 19th term (the 2017 break) produce the level shift through parsing?
Check 1: continuity of text statistics in our own parse across the boundary (printed by the inline analysis, see notes).
Check 2: the same speeches parsed independently by the German Parliamentary Corpus pipeline (GerParCor, 'Bundestag'
source rows of data/full_parliament/speeches_table_Germany.json, one pipeline across the boundary), re-scored with
the paper's model: quarterly series, six-quarter pre/post difference at the 2017 election and the speech-level
agreement with our parse. Input: data/revision/gerparcor_2014_2025_scored.csv (score_emi.py on the GerParCor text).
Output: results/revision/14_parse_check/{quarterly_two_parses.csv, numbers.json}
"""
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))   # German-EMI repository root: data in data/, outputs in results/
import os, json, numpy as np, pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = REPO_ROOT; OUT = os.path.join(REPO_ROOT, 'results', 'revision', '14_parse_check'); os.makedirs(OUT, exist_ok=True)
g = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'revision', 'gerparcor_2014_2025_scored.csv'), dtype={'ID': str, 'Term': str})
g['date'] = pd.to_datetime(g.Date, utc=True); g = g[g.date >= '2015-01-01'].copy(); g['q'] = g.date.dt.to_period('Q')
p = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'parliament_nhb.csv'), usecols=['id', 'date', 'electoralTerm', 'party', 'emi_w2vparliament', 'length'], dtype={'id': str})
p['date'] = pd.to_datetime(p.date, utc=True); p['q'] = p.date.dt.to_period('Q')
NUM = {'n_gerparcor': int(len(g)), 'n_ours': int(len(p))}
# quarterly series of both parses (GerParCor scores are z-standardised over 2014-2025 by score_emi; ours over 2015-2025)
qg = g.groupby('q').emi.agg(['mean', 'size']).rename(columns={'mean': 'emi_gerparcor', 'size': 'n_gerparcor'})
qp = p.groupby('q').emi_w2vparliament.agg(['mean', 'size']).rename(columns={'mean': 'emi_ours', 'size': 'n_ours'})
Q = qg.join(qp, how='inner'); Q.to_csv(os.path.join(OUT, 'quarterly_two_parses.csv'))
NUM['quarterly_corr'] = float(Q.emi_gerparcor.corr(Q.emi_ours))
# six full quarters before the election quarter vs six after the constitution quarter (as in the pre/post table)
eq = pd.Period('2017Q3', freq='Q')
def prepost(d, col):
    pre = d[(d.q < eq) & (d.q >= eq - 6)][col]; post = d[(d.q > eq + 1) & (d.q <= eq + 7)][col]
    t, pv = stats.ttest_ind(pre, post, equal_var=False); sd = np.sqrt((pre.var() + post.var()) / 2)
    return dict(n_pre=int(len(pre)), n_post=int(len(post)), mean_pre=float(pre.mean()), mean_post=float(post.mean()), diff=float(post.mean() - pre.mean()), d=float((post.mean() - pre.mean()) / sd), t=float(t), p=float(pv))
NUM['prepost_2017_gerparcor'] = prepost(g, 'emi'); NUM['prepost_2017_ours'] = prepost(p, 'emi_w2vparliament')
# same speeches in both parses (GerParCor Source_ID equals our speech id)
m = g.merge(p[['id', 'emi_w2vparliament', 'length']], left_on='Source_ID', right_on='id', how='inner')
NUM['matched_speeches'] = int(len(m)); NUM['speech_level_corr'] = float(m.emi.corr(m.emi_w2vparliament)); NUM['speech_level_spearman'] = float(m.emi.corr(m.emi_w2vparliament, method='spearman'))
NUM['length_corr'] = float(m.length_x.corr(m.length_y)) if 'length_x' in m.columns else None
NUM['prepost_2017_matched_gerparcor'] = prepost(m, 'emi'); NUM['prepost_2017_matched_ours'] = prepost(m, 'emi_w2vparliament')
# term-level: 18th vs 19th
for name, d, col in [('gerparcor', g, 'emi'), ('ours', p, 'emi_w2vparliament')]:
    term = d.Term if 'Term' in d.columns else d.electoralTerm.astype(int).astype(str)
    NUM[f'term_means_{name}'] = {t: float(v) for t, v in d.groupby(term)[col].mean().items()}
json.dump(NUM, open(os.path.join(OUT, 'numbers.json'), 'w'), indent=1)
pd.set_option('display.width', 200); print(Q.loc['2016Q1':'2019Q4'].round(3).to_string()); print(json.dumps(NUM, indent=1))
