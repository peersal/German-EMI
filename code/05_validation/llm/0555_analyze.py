"""Analyse the LLM annotations (after `bash k8s.sh fetch`).

Answers three questions:
 A. LLM vs human ground truth (1,142 survey chunks): Spearman with the human mean, AUC against the human labels,
    agreement between the two models. Also EMI vs human on the same chunks for comparison.
 B. EMI vs LLM on the 10k sample and on the ground truth: evidence_z vs LLM evidence, intuition_z vs LLM intuition,
    EMI vs LLM (evidence - intuition); by party and by year.
 C. Use vs mention: does the embedding score track evidence that is USED or evidence that is merely TALKED ABOUT?
    Partial correlations, the share of mention-heavy chunks by party, EMI-vs-LLM agreement with and without them,
    and the Linke question (does the model's evidence score follow mention rather than use for the Left?).
Outputs: results/llm/*.csv, numbers.json, figures.
"""
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))   # German-EMI repository root: data in data/, outputs in results/
import os, glob, json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score
import statsmodels.formula.api as smf
HERE = os.path.dirname(os.path.abspath(__file__)); OUTD = os.path.join(REPO_ROOT, 'results', 'llm'); os.makedirs(OUTD, exist_ok=True)
MODELS = {'meta-llama_Llama-3.1-8B-Instruct': 'Llama', 'Qwen_Qwen2.5-7B-Instruct': 'Qwen'}
NUM, rows = {}, []

def load(prompt):
    out = None
    for tag, short in MODELS.items():
        f = os.path.join(REPO_ROOT, 'data', 'llm', 'output', f'{prompt}_{tag}.parquet')
        if not os.path.exists(f): print('missing', f); continue
        d = pd.read_parquet(f)
        if 'id' not in d.columns: d = d.reset_index().rename(columns={d.index.name or 'index': 'id'})
        d = d[['id'] + [c for c in d.columns if c != 'id' and not c.endswith('_missing') and c != 'parse_error']].copy(); d['id'] = d.id.astype(str)
        d.columns = ['id'] + [f'{c}_{short}' for c in d.columns if c != 'id']
        d = d.drop_duplicates('id'); out = d if out is None else out.merge(d, on='id', how='outer')
    return out
R = load('rating'); U = load('use_mention'); L = R.merge(U, on='id', how='outer') if U is not None else R
for c in [c for c in L.columns if c != 'id']: L[c] = pd.to_numeric(L[c], errors='coerce')
for short in MODELS.values():
    if f'evidenz_basiert_{short}' in L: L[f'llm_emi_{short}'] = L[f'evidenz_basiert_{short}'] - L[f'intuition_basiert_{short}']
# model-average
for base in ['evidenz_basiert', 'intuition_basiert', 'llm_emi', 'evidence_use', 'evidence_mention', 'intuition_use', 'intuition_mention', 'procedural']:
    cs = [f'{base}_{s}' for s in MODELS.values() if f'{base}_{s}' in L]
    if cs: L[f'{base}_mean'] = L[cs].mean(axis=1)
print(f'annotations: {len(L):,} ids; parse failures per column:\n' + L.isna().mean().round(3).to_string())

gt = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'llm', 'inputs', 'groundtruth_meta.csv'), dtype={'id': str}).merge(L, on='id')
sm = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'llm', 'inputs', 'sample10k_meta.csv'), dtype={'id': str}, parse_dates=['date'])
emi = os.path.join(REPO_ROOT, 'data', 'llm', 'inputs', 'sample10k_emi.csv')
if os.path.exists(emi):
    e = pd.read_csv(emi, dtype={'id': str}); e = e.rename(columns={c: f'chunk_{c}' for c in e.columns if c != 'id'}); sm = sm.merge(e, on='id', how='left')
sm = sm.merge(L, on='id'); print(f'ground truth rows: {len(gt)}, sample rows: {len(sm)}')
def sp(x, y):
    m = pd.notna(x) & pd.notna(y); r, p = spearmanr(x[m], y[m]); return r, p, int(m.sum())
def auc(y, s):
    m = pd.notna(y) & pd.notna(s); return roc_auc_score(y[m], s[m]) if y[m].nunique() == 2 else np.nan

# ---------------- A. LLM vs human ------------------------------------------------------------------------------
print('\nA) LLM vs human ground truth')
A = []
for short in list(MODELS.values()) + ['mean']:
    for hum, lab, llm in [('ev_mean', 'ev_label', 'evidenz_basiert'), ('iu_mean', 'iu_label', 'intuition_basiert'), ('ev_mean', 'emi_label', 'llm_emi')]:
        c = f'{llm}_{short}'
        if c not in gt: continue
        if llm == 'llm_emi': r, p, n = sp(gt.ev_mean - gt.iu_mean, gt[c]); a = auc(gt.emi_label, gt[c]); hname = 'ev_mean - iu_mean'
        else: r, p, n = sp(gt[hum], gt[c]); a = auc(gt[lab], gt[c]); hname = hum
        A.append(dict(rater=short, construct=llm, human=hname, spearman=r, p=p, auc_vs_label=a, n=n)); print(f'   {short:5s} {llm:18s} rho={r:+.3f} AUC={a:.3f} n={n}')
for hum, lab, emi_c in [('ev_mean', 'ev_label', 'evidence_z'), ('iu_mean', 'iu_label', 'intuition_z'), ('ev_mean - iu_mean', 'emi_label', 'emi')]:
    x = gt.ev_mean - gt.iu_mean if '-' in hum else gt[hum]; r, p, n = sp(x, gt[emi_c]); a = auc(gt[lab], gt[emi_c])
    A.append(dict(rater='EMI (embedding)', construct=emi_c, human=hum, spearman=r, p=p, auc_vs_label=a, n=n)); print(f'   EMI   {emi_c:18s} rho={r:+.3f} AUC={a:.3f}')
if all(f'evidenz_basiert_{s}' in gt for s in MODELS.values()):
    s1, s2 = MODELS.values()
    for b in ['evidenz_basiert', 'intuition_basiert', 'evidence_use', 'evidence_mention', 'intuition_use', 'intuition_mention', 'procedural']:
        if f'{b}_{s1}' in L: r, p, n = sp(L[f'{b}_{s1}'], L[f'{b}_{s2}']); ex = (L[f'{b}_{s1}'] == L[f'{b}_{s2}']).mean(); A.append(dict(rater='inter-model', construct=b, human='', spearman=r, p=p, auc_vs_label=ex, n=n)); print(f'   inter-model {b:18s} rho={r:+.3f} exact agreement={ex:.2f}')
# by era / party on the ground truth (LLM mean vs human, EMI vs human)
grp = []
for g, d in list(gt.groupby('era')) + list(gt.groupby('party')):
    if len(d) < 30: continue
    grp.append(dict(group=g, n=len(d), auc_emi=auc(d.emi_label, d.emi), auc_llm=auc(d.emi_label, d.llm_emi_mean), rho_emi=sp(d.ev_mean - d.iu_mean, d.emi)[0], rho_llm=sp(d.ev_mean - d.iu_mean, d.llm_emi_mean)[0]))
G = pd.DataFrame(grp); print('\n   by group (AUC vs human EMI label): \n' + G.round(3).to_string(index=False)); G.to_csv(os.path.join(OUTD, 'A_by_group.csv'), index=False)
pd.DataFrame(A).to_csv(os.path.join(OUTD, 'A_llm_vs_human.csv'), index=False); NUM['A'] = A

# ---------------- B. EMI vs LLM --------------------------------------------------------------------------------
print('\nB) EMI (embedding) vs LLM')
B = []
for name, d, ev, iu, em in [('ground truth', gt, 'evidence_z', 'intuition_z', 'emi'), ('sample, speech-level EMI', sm, 'evidence_z_w2vparliament', 'intuition_z_w2vparliament', 'emi_w2vparliament')] + \
                           ([('sample, chunk-level EMI', sm, 'chunk_evidence_z', 'chunk_intuition_z', 'chunk_emi')] if 'chunk_emi' in sm else []):
    for short in list(MODELS.values()) + ['mean']:
        for emb, llm in [(ev, 'evidenz_basiert'), (iu, 'intuition_basiert'), (em, 'llm_emi')]:
            c = f'{llm}_{short}'
            if c in d and emb in d: r, p, n = sp(d[emb], d[c]); B.append(dict(set=name, rater=short, embedding=emb, llm=llm, spearman=r, p=p, n=n)); print(f'   {name:26s} {short:5s} {emb:28s} ~ {llm:18s} rho={r:+.3f} n={n}')
pd.DataFrame(B).to_csv(os.path.join(OUTD, 'B_emi_vs_llm.csv'), index=False); NUM['B'] = B
emc = 'chunk_emi' if 'chunk_emi' in sm else 'emi_w2vparliament'
byp = sm.groupby('party').apply(lambda d: pd.Series(dict(n=len(d), rho_emi_llm=sp(d[emc], d.llm_emi_mean)[0], emi=d[emc].mean(), llm_emi=d.llm_emi_mean.mean(), llm_ev=d.evidenz_basiert_mean.mean(), llm_iu=d.intuition_basiert_mean.mean())))
byy = sm.groupby(sm.date.dt.year).apply(lambda d: pd.Series(dict(n=len(d), rho_emi_llm=sp(d[emc], d.llm_emi_mean)[0], emi=d[emc].mean(), llm_emi=d.llm_emi_mean.mean(), llm_ev=d.evidenz_basiert_mean.mean(), llm_iu=d.intuition_basiert_mean.mean())))
print('\n   by party:\n' + byp.round(3).to_string()); print('\n   by year:\n' + byy.round(3).to_string())
byp.to_csv(os.path.join(OUTD, 'B_by_party.csv')); byy.to_csv(os.path.join(OUTD, 'B_by_year.csv'))

# ---------------- C. use vs mention ----------------------------------------------------------------------------
if 'evidence_use_mean' in sm:
    print('\nC) use vs mention')
    C = {}
    ev_emb = 'chunk_evidence_z' if 'chunk_evidence_z' in sm else 'evidence_z_w2vparliament'; iu_emb = 'chunk_intuition_z' if 'chunk_intuition_z' in sm else 'intuition_z_w2vparliament'
    for name, d, ev, iu in [('sample', sm, ev_emb, iu_emb), ('ground truth', gt, 'evidence_z', 'intuition_z')]:
        d = d.dropna(subset=['evidence_use_mean', 'evidence_mention_mean', 'intuition_use_mean', 'intuition_mention_mean', 'procedural_mean', ev, iu]).copy()
        for emb, lab in [(ev, 'embedding evidence'), (iu, 'embedding intuition')]:
            d['_y'] = d[emb]
            m = smf.ols('_y ~ evidence_use_mean + evidence_mention_mean + intuition_use_mean + intuition_mention_mean + procedural_mean', d).fit(cov_type='HC1')
            print(f'   {name:13s} {lab:20s} ~ ' + ' | '.join(f'{k.replace("_mean","")} {v:+.3f} (p={m.pvalues[k]:.3f})' for k, v in m.params.items() if k != 'Intercept') + f'  R2={m.rsquared:.3f}')
            C[f'{name}|{lab}'] = {k: dict(coef=v, p=m.pvalues[k]) for k, v in m.params.items()}
        # human raters: do they follow use or mention?
        if name == 'ground truth':
            for hum, lab in [('ev_mean', 'human evidence'), ('iu_mean', 'human intuition')]:
                d['_y'] = d[hum]; m = smf.ols('_y ~ evidence_use_mean + evidence_mention_mean + intuition_use_mean + intuition_mention_mean + procedural_mean', d).fit(cov_type='HC1')
                print(f'   {name:13s} {lab:20s} ~ ' + ' | '.join(f'{k.replace("_mean","")} {v:+.3f} (p={m.pvalues[k]:.3f})' for k, v in m.params.items() if k != 'Intercept') + f'  R2={m.rsquared:.3f}')
                C[f'{name}|{lab}'] = {k: dict(coef=v, p=m.pvalues[k]) for k, v in m.params.items()}
    # mention-heavy chunks: mention >= 3 and use <= 1
    sm['mention_heavy'] = (sm.evidence_mention_mean >= 2.5) & (sm.evidence_use_mean <= 1.5); sm['use_heavy'] = (sm.evidence_use_mean >= 2.5)
    byp2 = sm.groupby('party').agg(n=('id', 'size'), share_mention_heavy=('mention_heavy', 'mean'), share_use_heavy=('use_heavy', 'mean'), procedural=('procedural_mean', 'mean'),
                                   emb_evidence_mention_heavy=(ev_emb, lambda s: s[sm.loc[s.index, 'mention_heavy']].mean()), emb_evidence_use_heavy=(ev_emb, lambda s: s[sm.loc[s.index, 'use_heavy']].mean()))
    print('\n   by party: share of chunks that talk about evidence without using it, and the embedding evidence score of those chunks\n' + byp2.round(3).to_string()); byp2.to_csv(os.path.join(OUTD, 'C_by_party.csv'))
    r_all = sp(sm[emc], sm.llm_emi_mean); r_nomention = sp(sm.loc[~sm.mention_heavy, emc], sm.loc[~sm.mention_heavy, 'llm_emi_mean'])
    print(f'   EMI ~ LLM EMI: all chunks rho={r_all[0]:+.3f} (n={r_all[2]}); excluding mention-heavy rho={r_nomention[0]:+.3f} (n={r_nomention[2]})')
    C['emi_vs_llm_all'] = r_all[0]; C['emi_vs_llm_no_mention_heavy'] = r_nomention[0]; C['share_mention_heavy'] = sm.mention_heavy.mean(); NUM['C'] = C
    # ground truth, Left only
    lk = gt[gt.party.isin(['Linke/PDS', 'The Left Party', 'DIE LINKE', 'Linke', 'PDS'])]
    if len(lk) >= 20:
        print(f'   Left (n={len(lk)}): embedding evidence ~ use rho={sp(lk.evidence_z, lk.evidence_use_mean)[0]:+.2f}, ~ mention rho={sp(lk.evidence_z, lk.evidence_mention_mean)[0]:+.2f}; human evidence ~ use rho={sp(lk.ev_mean, lk.evidence_use_mean)[0]:+.2f}, ~ mention rho={sp(lk.ev_mean, lk.evidence_mention_mean)[0]:+.2f}')

# ---------------- D. the 2017 shift in the LLM measures ---------------------------------------------------------
print('\nD) LLM measures by period (10k sample): did use or mention move in 2017?')
sm['period'] = np.select([sm.date < '2017-07-01', (sm.date >= '2017-10-24') & (sm.date < '2019-07-01'), sm.date >= '2019-07-01'], ['2015-2017H1', '2017Q4-2019H1', '2019H2-2025'], 'election window')
D = sm[sm.period != 'election window'].groupby('period')[[emc, 'llm_emi_mean', 'evidenz_basiert_mean', 'intuition_basiert_mean', 'evidence_use_mean', 'evidence_mention_mean', 'intuition_use_mean', 'intuition_mention_mean', 'procedural_mean']].mean().round(3)
D.loc['step 2017 (post - pre)'] = D.loc['2017Q4-2019H1'] - D.loc['2015-2017H1']; print(D.to_string()); D.to_csv(os.path.join(OUTD, 'D_llm_by_period.csv'))
non = sm[(sm.party != 'AFD') & (sm.period != 'election window')]
Dn = non.groupby('period')[['llm_emi_mean', 'evidenz_basiert_mean', 'intuition_basiert_mean', 'evidence_use_mean', 'evidence_mention_mean', 'intuition_use_mean', 'intuition_mention_mean']].mean().round(3)
Dn.loc['step 2017 (post - pre)'] = Dn.loc['2017Q4-2019H1'] - Dn.loc['2015-2017H1']; print('\n   non-AfD speeches only:\n' + Dn.to_string()); Dn.to_csv(os.path.join(OUTD, 'D_llm_by_period_nonafd.csv'))
from scipy.stats import ttest_ind
for c in ['llm_emi_mean', 'evidenz_basiert_mean', 'intuition_basiert_mean', 'evidence_use_mean', 'intuition_use_mean']:
    a = non[non.period == '2015-2017H1'][c]; b = non[non.period == '2017Q4-2019H1'][c]; t = ttest_ind(a, b); print(f'   non-AfD {c:24s} pre {a.mean():.3f} (n={len(a)}) post {b.mean():.3f} (n={len(b)}) diff {b.mean()-a.mean():+.3f} p={t.pvalue:.4f}')
NUM['D'] = D.to_dict(); NUM['D_nonafd'] = Dn.to_dict()
json.dump(NUM, open(os.path.join(OUTD, 'numbers.json'), 'w'), indent=1, default=float)
# figure: EMI vs LLM EMI on the sample (binned), and by party
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
b = sm.groupby(pd.cut(sm[emc], np.quantile(sm[emc].dropna(), np.linspace(0, 1, 21)), include_lowest=True), observed=True).agg(x=(emc, 'mean'), y=('llm_emi_mean', 'mean'), se=('llm_emi_mean', 'sem'))
ax[0].errorbar(b.x, b.y, yerr=1.96 * b.se, fmt='o-', color='#0072B2', ms=4); ax[0].set_xlabel('embedding EMI (chunk, ventiles)'); ax[0].set_ylabel('LLM evidence minus intuition (0-4 scale, 2-model mean)'); ax[0].set_title('A  EMI vs LLM rating, 10k sample', loc='left'); ax[0].grid(alpha=.3)
ax[1].bar(range(len(byp)), byp.rho_emi_llm, color='#4D4D4D'); ax[1].set_xticks(range(len(byp))); ax[1].set_xticklabels(byp.index, rotation=20, ha='right'); ax[1].set_ylabel('Spearman rho, EMI vs LLM EMI'); ax[1].set_title('B  Agreement by party', loc='left'); ax[1].grid(alpha=.3, axis='y')
plt.tight_layout(); plt.savefig(os.path.join(OUTD, 'emi_vs_llm.png'), dpi=150, bbox_inches='tight'); print('\nfigure: results/emi_vs_llm.png')
