"""Checks on the local before/after effect around AfD speeches.
 (a) EMI profile of other parties' speeches by distance from an AfD / FDP / SPD speech
 (b) position-matched: local diff around every speech regressed on AfD indicator + position-in-item dummies + item FE
 (c) placebo with intuition-heavy speeches of other parties (bottom quartile within party)
 (d) is the drop concentrated in following speeches that mention the AfD?
"""
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))   # German-EMI repository root: data in data/, outputs in results/
import os, re, numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = REPO_ROOT; OUT = os.path.join(REPO_ROOT, 'results', 'revision', '12_afd_sequence')
EMI = 'emi_w2vparliament'
raw = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'revision', 'bundestag_speeches_2000_2025_raw.csv'), usecols=['Source_ID', 'Date', 'Sitting', 'Agenda', 'Party', 'Role'], dtype=str)
raw = raw[(raw.Date >= '2017-10-24') & (raw.Role == 'Regular')]
sc = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'parliament_nhb.csv'), usecols=['id', EMI, 'party', 'politicianId', 'text_clean'], dtype={'id': str})
d = raw.merge(sc, left_on='Source_ID', right_on='id'); d['order'] = pd.to_numeric(d.Source_ID.str.extract(r'ID(\d+)')[0], errors='coerce')
d['item'] = d.Sitting + '|' + d.Agenda.fillna('(no agenda)'); d = d.sort_values(['Sitting', 'order']).reset_index(drop=True)
d['pos_item'] = d.groupby('item').cumcount().clip(upper=8); d['mentions_afd'] = d.text_clean.fillna('').str.lower().str.contains(r'\bafd\b|alternative für deutschland').astype(int)
d['Party'] = d.Party.fillna('none'); d['low_q'] = d.groupby('Party')[EMI].transform(lambda s: (s <= s.quantile(0.25)).astype(float)).fillna(0).astype(int)

# (a) profile
print('(a) mean EMI of other parties at distance o (in non-X speeches) from an X speech:')
prof = pd.read_csv(os.path.join(OUT, 'local_window_profile.csv'), index_col=0)
print(prof.loc[[-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]].round(3).T.to_string())

# local diff around every speech (k = 1 and 2): next non-same-party speech minus previous non-same-party speech
rows = []
for sit, g in d.groupby('Sitting', sort=False):
    party = g.Party.values; emi = g[EMI].values; ment = g.mentions_afd.values; pos = g.pos_item.values; item = g.item.values; lowq = g.low_q.values; ids = g.Source_ID.values
    n = len(g)
    for i in range(n):
        X = party[i]; before = [j for j in range(i - 1, -1, -1) if party[j] != X][:2]; after = [j for j in range(i + 1, n) if party[j] != X][:2]
        if len(before) >= 1 and len(after) >= 1:
            rows.append(dict(id=ids[i], sitting=sit, item=item[i], party=X, pos=pos[i], low_q=lowq[i], diff1=emi[after[0]] - emi[before[0]],
                             diff2=(emi[after[:2]].mean() - emi[before[:2]].mean()) if len(before) == 2 and len(after) == 2 else np.nan,
                             after1_mentions_afd=ment[after[0]], before1=emi[before[0]], after1=emi[after[0]]))
L = pd.DataFrame(rows); L['is_afd'] = (L.party == 'AfD').astype(int)
L.to_csv(os.path.join(OUT, 'local_window_all_speeches.csv'), index=False)

def cl_ols(y, X, groups):
    r = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(groups)[0]}); return r
# (b) position-matched, item FE (demean within item)
print('\n(b) local diff (k=1) ~ AfD indicator, with position-in-item dummies and item fixed effects:')
Lb = L.dropna(subset=['diff1']).copy()
Xb = pd.get_dummies(Lb.pos.astype(int), prefix='pos', drop_first=True).astype(float); Xb['is_afd'] = Lb.is_afd
yb = Lb.diff1 - Lb.groupby('item').diff1.transform('mean'); Xb = Xb - Xb.groupby(Lb.item).transform('mean')
r = cl_ols(yb, Xb, Lb.sitting); print(f'   AfD effect on diff1 = {r.params["is_afd"]:+.3f} (se {r.bse["is_afd"]:.3f}, p={r.pvalues["is_afd"]:.4f}), n={int(r.nobs):,}')
Lb2 = L.dropna(subset=['diff2']).copy(); Xb2 = pd.get_dummies(Lb2.pos.astype(int), prefix='pos', drop_first=True).astype(float); Xb2['is_afd'] = Lb2.is_afd
yb2 = Lb2.diff2 - Lb2.groupby('item').diff2.transform('mean'); Xb2 = Xb2 - Xb2.groupby(Lb2.item).transform('mean')
r2 = cl_ols(yb2, Xb2, Lb2.sitting); print(f'   AfD effect on diff2 = {r2.params["is_afd"]:+.3f} (se {r2.bse["is_afd"]:.3f}, p={r2.pvalues["is_afd"]:.4f}), n={int(r2.nobs):,}')
print('   mean diff1 by position in item, AfD vs others:'); print(L.groupby(['pos', 'is_afd']).diff1.mean().unstack().round(3).rename(columns={0: 'other parties', 1: 'AfD'}).T.to_string())
print('   where do AfD speeches sit in the item?', (L[L.is_afd == 1].pos.value_counts(normalize=True).sort_index().round(2)).to_dict())

# (c) intuition-heavy speeches of each party: do the following speeches drop too?
print('\n(c) local diff (k=1) after speeches in the bottom EMI quartile of their own party:')
for X, g in L[L.low_q == 1].groupby('party'):
    gg = g.dropna(subset=['diff1']); m = gg.diff1.mean(); se = gg.groupby('sitting').diff1.sum().pipe(lambda s: np.sqrt(((s - m * gg.groupby('sitting').size()) ** 2).sum()) / len(gg))
    print(f'   {X:24s} diff1 = {m:+.3f} (se {se:.3f}, p={2*(1-stats.norm.cdf(abs(m/se))):.3f}) n={len(gg):,}')
# same with position + item FE: low_q x is_afd interaction
Lc = L.dropna(subset=['diff1']).copy(); Xc = pd.get_dummies(Lc.pos.astype(int), prefix='pos', drop_first=True).astype(float)
Xc['is_afd'] = Lc.is_afd; Xc['low_q'] = Lc.low_q; Xc['afd_x_low'] = Lc.is_afd * Lc.low_q
yc = Lc.diff1 - Lc.groupby('item').diff1.transform('mean'); Xc = Xc - Xc.groupby(Lc.item).transform('mean')
rc = cl_ols(yc, Xc, Lc.sitting); print(f'   item FE model: is_afd {rc.params["is_afd"]:+.3f} (p={rc.pvalues["is_afd"]:.3f}) | low-EMI speech (any party) {rc.params["low_q"]:+.3f} (p={rc.pvalues["low_q"]:.3f}) | AfD x low-EMI {rc.params["afd_x_low"]:+.3f} (p={rc.pvalues["afd_x_low"]:.3f})')

# (d) does the following speech mention the AfD?
print('\n(d) the speech right after an AfD speech: mentions the AfD?')
A = L[L.is_afd == 1].dropna(subset=['diff1'])
print(f'   share of following speeches mentioning the AfD: {A.after1_mentions_afd.mean():.1%} (after other parties\' speeches: {L[L.is_afd==0].after1_mentions_afd.mean():.1%})')
for m_, g in A.groupby('after1_mentions_afd'):
    mm = g.diff1.mean(); se = np.sqrt(((g.groupby('sitting').diff1.sum() - mm * g.groupby('sitting').size()) ** 2).sum()) / len(g)
    print(f'   following speech {"mentions" if m_ else "does not mention"} the AfD: diff1 = {mm:+.3f} (se {se:.3f}) n={len(g):,} | mean EMI of that speech {g.after1.mean():+.3f}')
# EMI of all non-AfD speeches that mention the AfD vs not (speaker FE)
nn = d[d.Party != 'AfD'].copy(); nn['politicianId'] = nn.politicianId.astype(str)
y = nn[EMI] - nn.groupby('politicianId')[EMI].transform('mean'); x = nn.mentions_afd - nn.groupby('politicianId').mentions_afd.transform('mean')
rr = cl_ols(y, pd.DataFrame({'mentions_afd': x}), nn.Sitting); print(f'   any non-AfD speech, speaker FE: mentioning the AfD -> EMI {rr.params["mentions_afd"]:+.3f} (se {rr.bse["mentions_afd"]:.3f}, p={rr.pvalues["mentions_afd"]:.4f}); share of non-AfD speeches mentioning the AfD {nn.mentions_afd.mean():.1%}')
