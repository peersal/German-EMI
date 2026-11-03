"""Count-dictionary baseline for EMI (revision, Reviewer 1).

Count-based EMI = (evidence keyword hits - intuition keyword hits) per 100 tokens, using the 86 validated keywords of
Extended Data Table 1 (code/05_validation/keywords/validated_keywords_german.csv). Two matching rules: exact (lower-cased
word form; multi-word keywords as phrases) and prefix (a token counts if it begins with the keyword, which approximates
inflected forms and compounds). Compared with the DDR EMI and the LLM composite on the same human-validated items: AUC
against the human label, Spearman rho with the human EMI, share of items with a score of exactly zero. Tweets are
included only where data/twitter_nhb.csv with the tweet text is available.
Output: results/revision/16_count_baseline/{count_scores.csv, summary.csv, numbers.json, fig_count_baseline.png/pdf}
"""
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))   # German-EMI repository root: data in data/, outputs in results/
import os, re, json, numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(REPO_ROOT, 'results', 'revision', '16_count_baseline'); os.makedirs(OUT, exist_ok=True)
VD = os.path.join(HERE, 'data_2026')
kw = pd.read_csv(os.path.join(REPO_ROOT, 'code', '05_validation', 'keywords', 'validated_keywords_german.csv'), encoding='utf-8-sig')
EV = [w.strip().lower() for w in kw.evidence.dropna()]; IU = [w.strip().lower() for w in kw.intuition.dropna()]
print('keywords: evidence', len(EV), 'intuition', len(IU), 'total', len(EV) + len(IU))
TOK = re.compile(r'[a-zäöüß]+')


def counter(words):
    single = set(w for w in words if ' ' not in w); phrases = [re.compile(r'\b' + re.escape(w) + r'\b') for w in words if ' ' in w]
    prefixes = tuple(sorted(single, key=len))
    def f(text):
        t = str(text).lower(); toks = TOK.findall(t); n = len(toks)
        exact = sum(1 for x in toks if x in single) + sum(len(p.findall(t)) for p in phrases)
        pref = sum(1 for x in toks if x.startswith(prefixes)) + sum(len(p.findall(t)) for p in phrases)
        return n, exact, pref
    return f


cev, ciu = counter(EV), counter(IU)
g24 = pd.read_csv(os.path.join(VD, 'groundtruth_parliament_2024.csv'), dtype={'id': str}).drop_duplicates('id')
g26 = pd.read_csv(os.path.join(VD, 'groundtruth_parliament_2026.csv'), dtype={'id': str})
sets = [('Bundestag 2025', g24), ('Bundestag 2026', g26)]
gt = pd.read_csv(os.path.join(VD, 'groundtruth_twitter_2026.csv'), dtype={'tweet_id': str})
TW = os.path.join(REPO_ROOT, 'data', 'twitter_nhb.csv')
if os.path.exists(TW):   # tweet texts are not part of the public data (X Developer Agreement)
    ids = set(gt.tweet_id); tw = pd.concat([c[c.id.isin(ids)] for c in pd.read_csv(TW, usecols=['id', 'text'], dtype={'id': str}, chunksize=500000)]).drop_duplicates('id')
    gt = gt.merge(tw.rename(columns={'id': 'tweet_id'}), on='tweet_id', how='left'); sets.append(('Twitter 2026', gt))
else:
    print('data/twitter_nhb.csv not available: tweets skipped')
l24 = pd.read_csv(os.path.join(VD, 'llm_parliament_2024.csv'), dtype={'id': str}); l26 = pd.read_csv(os.path.join(VD, 'llm_2026.csv'), dtype={'id': str})
LLM = pd.concat([l24, l26])[['id', 'llm_emi']]
rows = []
for name, g in sets:
    g = g.dropna(subset=['ev_mean', 'iu_mean', 'emi', 'text']).copy()
    for r in g.itertuples():
        n, e_ex, e_pr = cev(r.text); _, i_ex, i_pr = ciu(r.text)
        rows.append(dict(set=name, id=r.id, n_tokens=n, ev_exact=e_ex, iu_exact=i_ex, ev_prefix=e_pr, iu_prefix=i_pr,
                         count_emi_exact=100.0 * (e_ex - i_ex) / n if n else np.nan, count_emi_prefix=100.0 * (e_pr - i_pr) / n if n else np.nan,
                         ddr_emi=r.emi, human_emi=r.ev_mean - r.iu_mean))
V = pd.DataFrame(rows).merge(LLM, on='id', how='left'); V['label'] = np.where(V.human_emi > 0, 1, np.where(V.human_emi < 0, 0, np.nan))
V.to_csv(os.path.join(OUT, 'count_scores.csv'), index=False)
MEAS = {'count_emi_exact': 'count EMI (exact)', 'count_emi_prefix': 'count EMI (prefix)', 'ddr_emi': 'DDR EMI', 'llm_emi': 'LLM composite'}
groups = {'Bundestag pooled': V[V.set.str.startswith('Bundestag')], 'Bundestag 2025': V[V.set == 'Bundestag 2025'], 'Bundestag 2026': V[V.set == 'Bundestag 2026']}
if (V.set == 'Twitter 2026').any(): groups['Twitter 2026'] = V[V.set == 'Twitter 2026']
summ = []
for gname, g in groups.items():
    for m, lab in MEAS.items():
        gl = g.dropna(subset=['label', m]); y = gl.label.astype(int)
        summ.append(dict(set=gname, measure=lab, n=len(gl), auc=roc_auc_score(y, gl[m]), rho_human=stats.spearmanr(g[m], g.human_emi, nan_policy='omit').correlation,
                         rho_ddr=stats.spearmanr(g[m], g.ddr_emi, nan_policy='omit').correlation if m != 'ddr_emi' else 1.0,
                         share_zero=float((g[m] == 0).mean()) if m.startswith('count') else np.nan,
                         share_no_hit=float(((g.ev_exact + g.iu_exact) == 0).mean()) if m == 'count_emi_exact' else (float(((g.ev_prefix + g.iu_prefix) == 0).mean()) if m == 'count_emi_prefix' else np.nan)))
S = pd.DataFrame(summ); S.to_csv(os.path.join(OUT, 'summary.csv'), index=False)
pd.set_option('display.width', 200); print(S.round(3).to_string(index=False))
hits = {s: dict(mean_ev_exact=float(g.ev_exact.mean()), mean_iu_exact=float(g.iu_exact.mean()), mean_ev_prefix=float(g.ev_prefix.mean()), mean_iu_prefix=float(g.iu_prefix.mean()), mean_tokens=float(g.n_tokens.mean())) for s, g in groups.items()}
json.dump({'keywords': {'evidence': len(EV), 'intuition': len(IU)}, 'summary': S.to_dict('records'), 'hits': hits}, open(os.path.join(OUT, 'numbers.json'), 'w'), indent=1, default=float)
# figure: ROC curves
plt.rcParams.update({'font.family': 'Arial', 'font.size': 11}); panels = [g for g in ['Bundestag pooled', 'Twitter 2026'] if g in groups]
fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.6), squeeze=False); axes = axes[0]
cols = {'count_emi_exact': '#999999', 'count_emi_prefix': '#4D4D4D', 'ddr_emi': '#0072B2', 'llm_emi': '#D55E00'}
for ax, gname in zip(axes, panels):
    g = groups[gname]
    for m, lab in MEAS.items():
        gl = g.dropna(subset=['label', m]); fpr, tpr, _ = roc_curve(gl.label.astype(int), gl[m]); a = roc_auc_score(gl.label.astype(int), gl[m])
        ax.plot(fpr, tpr, color=cols[m], lw=2 if m in ('ddr_emi', 'llm_emi') else 1.5, ls='-' if m != 'count_emi_exact' else '--', label=f'{lab} (AUC {a:.2f})')
    ax.plot([0, 1], [0, 1], color='#BBBBBB', lw=0.8); ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate')
    ax.set_title(f'{gname} (n = {len(g.dropna(subset=["label"])):,})', loc='left'); ax.legend(frameon=False, fontsize=9, loc='lower right')
for ax, L in zip(axes, 'AB'): ax.text(-0.12, 1.05, L, transform=ax.transAxes, fontsize=14, fontweight='bold')
fig.tight_layout()
for ext in ('png', 'pdf'): fig.savefig(os.path.join(OUT, f'fig_count_baseline.{ext}'), dpi=200, bbox_inches='tight')
print('done')
