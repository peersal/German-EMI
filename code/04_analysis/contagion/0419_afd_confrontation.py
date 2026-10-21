"""Confrontation with the AfD as a component of the EMI shift.
 1. Mentioning parties: EMI of non-member speeches that name a party (AfD, CDU/CSU, SPD, FDP, Greens, Left), speaker FE
 2. Share of non-AfD speeches mentioning the AfD by quarter 2015-2025; quarterly EMI with and without those speeches;
    the paper's constrained breakpoint model with a 'mentions AfD' control and on the non-mentioning subsample
 3. Dose: reply EMI drop as a function of the preceding AfD speech's EMI
Output: results/revision/12_afd_sequence/confrontation_*.csv/png
"""
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))   # German-EMI repository root: data in data/, outputs in results/
import os, re, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt, statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = REPO_ROOT; OUT = os.path.join(REPO_ROOT, 'results', 'revision', '12_afd_sequence')
EMI = 'emi_w2vparliament'
sc = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'parliament_nhb.csv'), usecols=['id', 'date', 'party', 'leaning', 'politicianId', 'text_clean', EMI], dtype={'id': str}).dropna(subset=['politicianId', EMI])
sc['politicianId'] = sc.politicianId.astype(int).astype(str); sc['date'] = pd.to_datetime(sc.date, utc=True); sc['q'] = sc.date.dt.tz_localize(None).dt.to_period('Q')
t = sc.text_clean.fillna('').str.lower()
PAT = {'AfD': r'\bafd\b|alternative für deutschland', 'CDU/CSU': r'\bcdu\b|\bcsu\b|\bunion\b|cducsu', 'SPD': r'\bspd\b|sozialdemokrat', 'FDP': r'\bfdp\b|\bliberale', 'Greens': r'\bgrünen?\b|bündnis 90', 'Left': r'\blinke\b|\blinkspartei\b|\bpds\b'}
OWN = {'AfD': 'AFD', 'CDU/CSU': 'CDU/CSU', 'SPD': 'SPD', 'FDP': 'FDP', 'Greens': 'Alliance 90/The Greens', 'Left': 'The Left Party'}
for k, pat in PAT.items(): sc[f'm_{k}'] = t.str.contains(pat, regex=True).astype(int)
def cl_fit(y, X, groups): return sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(groups)[0]})

# --- 1. mentioning a party (non-member speeches), speaker FE, 2017Q4-2025 ----------------------------------
post = sc[sc.date >= '2017-10-24'].copy()
print('1) EMI effect of naming a party, speeches by non-members of that party, speaker fixed effects, since Oct 2017:')
rows = []
for k in PAT:
    g = post[post.party != OWN[k]].copy(); y = g[EMI] - g.groupby('politicianId')[EMI].transform('mean'); x = g[f'm_{k}'] - g.groupby('politicianId')[f'm_{k}'].transform('mean')
    r = cl_fit(y, pd.DataFrame({'m': x}), g.politicianId); rows.append(dict(party_named=k, share_mentioning=g[f'm_{k}'].mean(), effect=r.params['m'], se=r.bse['m'], p=r.pvalues['m'], n=int(r.nobs)))
    print(f'   names {k:8s}: {g[f"m_{k}"].mean():5.1%} of speeches | EMI {r.params["m"]:+.3f} (se {r.bse["m"]:.3f})')
# all at once, non-AfD speeches
g = post[post.party != 'AFD'].copy(); cols = [f'm_{k}' for k in PAT]
y = g[EMI] - g.groupby('politicianId')[EMI].transform('mean'); X = g[cols] - g.groupby('politicianId')[cols].transform('mean')
r = cl_fit(y, X, g.politicianId); print('   jointly (non-AfD speeches): ' + ' | '.join(f'{c[2:]} {r.params[c]:+.3f}' for c in cols))
pd.DataFrame(rows).to_csv(os.path.join(OUT, 'confrontation_mentions_by_party.csv'), index=False)

# --- 2. share of AfD-mentioning speeches over time and the level shift with/without them ---------------------
non = sc[sc.party != 'AFD'].copy()
qq = non.groupby('q').agg(share_mention=('m_AfD', 'mean'), emi_all=(EMI, 'mean'), n=(EMI, 'size'))
qq['emi_no_mention'] = non[non.m_AfD == 0].groupby('q')[EMI].mean(); qq['emi_mention'] = non[non.m_AfD == 1].groupby('q')[EMI].mean()
qq.to_csv(os.path.join(OUT, 'confrontation_quarterly.csv'))
print('\n2) non-AfD speeches mentioning the AfD, by year:'); yr = non.groupby(non.date.dt.year).agg(share=('m_AfD', 'mean'), emi_all=(EMI, 'mean'), emi_no_mention=(EMI, lambda s: s[non.loc[s.index, 'm_AfD'] == 0].mean()))
print(yr.round(3).T.to_string())
pre_all = non[(non.q >= pd.Period('2016Q1')) & (non.q <= pd.Period('2017Q2'))]; post_all = non[(non.q >= pd.Period('2018Q1')) & (non.q <= pd.Period('2019Q2'))]
print(f'   2017 step, all non-AfD speeches:            {post_all[EMI].mean() - pre_all[EMI].mean():+.3f}')
print(f'   2017 step, speeches NOT mentioning the AfD: {post_all[post_all.m_AfD==0][EMI].mean() - pre_all[pre_all.m_AfD==0][EMI].mean():+.3f}')
print(f'   2017 step, speeches mentioning the AfD:     {post_all[post_all.m_AfD==1][EMI].mean() - pre_all[pre_all.m_AfD==1][EMI].mean():+.3f}  (n pre {int(pre_all.m_AfD.sum())}, post {int(post_all.m_AfD.sum())})')
# breakpoint model (constrained, Eq. 2) with and without the mention control, and on the non-mentioning subsample
BP = {'elec2017': 20173, 'elec2021': 20213, 'elec2025': 20251}
def prep(d):
    m = d[[EMI, 'q', 'leaning', 'politicianId', 'm_AfD']].copy(); m['time_quarterly'] = m.q.dt.year * 10 + m.q.dt.quarter
    qm = m.groupby(['politicianId', 'time_quarterly'])[EMI].mean().rename('_q').reset_index(); qm['emi_lag1'] = qm.groupby('politicianId')['_q'].shift(1)
    m = m.merge(qm[['politicianId', 'time_quarterly', 'emi_lag1']], on=['politicianId', 'time_quarterly']).dropna(); tm = m.time_quarterly.mean(); m['time_centered'] = m.time_quarterly - tm
    for k, bq in BP.items(): m[f'slope_{k}'] = np.maximum(0, m.time_centered - (bq - tm)); m[f'level_{k}'] = (m.time_quarterly >= bq).astype(float)
    m['leaning'] = pd.Categorical(m.leaning, categories=['middle', 'left', 'right']); return m.reset_index(drop=True)
base = f'{EMI} ~ time_centered + leaning + time_centered:leaning + ' + ' + '.join(f'slope_{k} + level_{k}' for k in BP) + ' + emi_lag1'
m_all = prep(sc)
print('\n   constrained breakpoint model (all speeches, random speaker intercepts):')
for tag, f, data in [('baseline', base, m_all), ('+ mentions-AfD control', base + ' + m_AfD', m_all), ('speeches not mentioning the AfD', base, prep(sc[sc.m_AfD == 0]))]:
    mod = MixedLM.from_formula(f, groups=data.politicianId, data=data)
    try: r = mod.fit(reml=False)
    except Exception: r = mod.fit(reml=False, method='powell')
    extra = f" | mentions AfD = {r.params['m_AfD']:+.3f}" if 'm_AfD' in r.params else ''
    print(f'   {tag:34s} level 2017 = {r.params["level_elec2017"]:+.3f} (se {r.bse["level_elec2017"]:.3f}) | level 2021 = {r.params["level_elec2021"]:+.3f} | level 2025 = {r.params["level_elec2025"]:+.3f}{extra} | N={int(r.nobs):,}')

# --- 3. dose: reply drop vs the AfD speech's EMI -------------------------------------------------------------
L = pd.read_csv(os.path.join(OUT, 'local_window_all_speeches.csv'), dtype={'id': str})
A = L[L.party == 'AfD'].merge(sc[['id', EMI]], on='id').dropna(subset=['diff1']); A['afd_emi_q'] = pd.qcut(A[EMI], 4, labels=['most intuitive', 'q2', 'q3', 'most evidence-based'])
print('\n3) reply drop (next non-AfD speech minus previous non-AfD speech) by the AfD speech\'s own EMI quartile:')
print(A.groupby('afd_emi_q', observed=True).agg(n=('diff1', 'size'), reply_drop=('diff1', 'mean'), share_reply_mentions_afd=('after1_mentions_afd', 'mean')).round(3).to_string())
r = cl_fit(A.diff1, sm.add_constant(A[[EMI]]), A.sitting); print(f'   slope of reply diff on AfD speech EMI: {r.params[EMI]:+.3f} per EMI unit (se {r.bse[EMI]:.3f}, p={r.pvalues[EMI]:.4f})')

# figure
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
x = qq.index.to_timestamp(); ax[0].plot(x, qq.emi_all, 'k-', lw=2, label='all non-AfD speeches'); ax[0].plot(x, qq.emi_no_mention, color='#0072B2', lw=2, ls='--', label='speeches not mentioning the AfD'); ax[0].plot(x, qq.emi_mention, color='#D55E00', lw=1.5, ls=':', label='speeches mentioning the AfD')
ax[0].axhline(0, color='grey', lw=.8); ax[0].axvline(pd.Timestamp('2017-10-24'), color='grey', ls=':'); ax[0].set_ylabel('quarterly mean EMI'); ax[0].legend(frameon=False, fontsize=9); ax[0].grid(alpha=.3); ax[0].set_title('A  Non-AfD speeches with and without mention of the AfD')
ax2 = ax[0].twinx(); ax2.fill_between(x, 0, qq.share_mention, color='grey', alpha=.15); ax2.set_ylabel('share mentioning the AfD', color='grey'); ax2.set_ylim(0, 0.6)
s = A.groupby('afd_emi_q', observed=True).diff1.agg(['mean', 'sem']); ax[1].bar(range(4), s['mean'], yerr=1.96 * s['sem'], color=['#D55E00', '#E69F00', '#56B4E9', '#0072B2'], capsize=3)
ax[1].set_xticks(range(4)); ax[1].set_xticklabels(s.index, rotation=15); ax[1].axhline(0, color='grey', lw=.8); ax[1].set_ylabel('reply drop: next minus previous non-AfD speech'); ax[1].set_title('B  Reply drop by the AfD speech\'s own EMI'); ax[1].grid(alpha=.3, axis='y')
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'confrontation.png'), dpi=150, bbox_inches='tight'); plt.savefig(os.path.join(OUT, 'confrontation.pdf'), bbox_inches='tight'); print('\nfigure: confrontation.png')
