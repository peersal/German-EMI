"""Discourse EMI before and after an AfD speech within the same sitting (local windows).

For each speech by party X in a sitting, the k non-X speeches right before it and the k non-X speeches right after it
(same sitting, by speech order). Statistic: mean EMI(after) - mean EMI(before), averaged over all X speeches, with
standard errors clustered by sitting. Computed for X = AfD and for every other party as placebo; the AfD-specific
effect is AfD minus the placebo average. Also the profile of other speakers' EMI by distance (in speeches) from an
AfD speech, -10..+10, relative to the same profile around FDP speeches. Variants: k = 1, 2, 3, 5; first AfD speech of
the sitting only; AfD speeches that are themselves strongly intuition-based (bottom quartile of AfD EMI).
Output: results/revision/12_afd_sequence/local_window_*.csv/png
"""
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))   # German-EMI repository root: data in data/, outputs in results/
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = REPO_ROOT; OUT = os.path.join(REPO_ROOT, 'results', 'revision', '12_afd_sequence'); os.makedirs(OUT, exist_ok=True)
EMI = 'emi_w2vparliament'
raw = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'revision', 'bundestag_speeches_2000_2025_raw.csv'), usecols=['Source_ID', 'Date', 'Sitting', 'Party', 'Role'], dtype=str)
raw = raw[(raw.Date >= '2017-10-24') & (raw.Role == 'Regular')]
sc = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'parliament_nhb.csv'), usecols=['id', EMI, 'party'], dtype={'id': str})
d = raw.merge(sc, left_on='Source_ID', right_on='id'); d['order'] = pd.to_numeric(d.Source_ID.str.extract(r'ID(\d+)')[0], errors='coerce')
d = d.sort_values(['Sitting', 'order']).reset_index(drop=True)
PARTIES = ['AfD', 'FDP', 'SPD', 'CDU/CSU', 'BÜNDNIS 90/DIE GRÜNEN', 'DIE LINKE']
afd_q25 = d.loc[d.Party == 'AfD', EMI].quantile(0.25)

def windows(X, k, first_only=False, strong_only=False):
    """Return per-X-speech rows: mean EMI of the k non-X speeches before and after, same sitting."""
    rows = []
    for sit, g in d.groupby('Sitting', sort=False):
        party = g.Party.values; emi = g[EMI].values; isx = party == X
        idx = np.where(isx)[0]
        if first_only and len(idx): idx = idx[:1]
        for i in idx:
            if strong_only and emi[i] > afd_q25: continue
            before = [j for j in range(i - 1, -1, -1) if not isx[j]][:k]; after = [j for j in range(i + 1, len(g)) if not isx[j]][:k]
            if len(before) == k and len(after) == k:
                rows.append(dict(sitting=sit, before=emi[before].mean(), after=emi[after].mean()))
    w = pd.DataFrame(rows); w['diff'] = w.after - w.before
    return w
def cl_mean(w):
    g = w.groupby('sitting')['diff'].agg(['sum', 'size']); n = len(w); m = w['diff'].mean()
    # cluster-robust SE of the mean (sum of within-sitting deviations)
    u = (g['sum'] - m * g['size']); se = np.sqrt((u ** 2).sum()) / n
    return m, se, n, w.sitting.nunique()
res = []
for k in [1, 2, 3, 5]:
    for X in PARTIES:
        m, se, n, ns = cl_mean(windows(X, k)); res.append(dict(k=k, party=X, variant='all speeches', diff=m, se=se, n_speeches=n, n_sittings=ns))
for X in PARTIES:
    m, se, n, ns = cl_mean(windows(X, 3, first_only=True)); res.append(dict(k=3, party=X, variant='first speech of sitting', diff=m, se=se, n_speeches=n, n_sittings=ns))
m, se, n, ns = cl_mean(windows('AfD', 3, strong_only=True)); res.append(dict(k=3, party='AfD', variant='strongly intuition-based AfD speeches (bottom quartile)', diff=m, se=se, n_speeches=n, n_sittings=ns))
R = pd.DataFrame(res); R['p'] = 2 * (1 - __import__('scipy').stats.norm.cdf(np.abs(R['diff'] / R.se))); R.to_csv(os.path.join(OUT, 'local_window_results.csv'), index=False)
pd.set_option('display.width', 200)
print('mean EMI of other parties: k speeches AFTER minus k speeches BEFORE a speech by party X (same sitting), SE clustered by sitting')
print(R[R.variant == 'all speeches'].pivot(index='k', columns='party', values='diff').round(3).to_string())
print('\np-values:'); print(R[R.variant == 'all speeches'].pivot(index='k', columns='party', values='p').round(3).to_string())
plc = R[(R.variant == 'all speeches') & (R.party != 'AfD')].groupby('k')['diff'].mean(); afd = R[(R.variant == 'all speeches') & (R.party == 'AfD')].set_index('k')['diff']
print('\nAfD minus placebo-party average:', (afd - plc).round(3).to_dict())
print('\nvariants (k = 3):'); print(R[R.variant != 'all speeches'][['party', 'variant', 'diff', 'se', 'p', 'n_speeches']].round(3).to_string(index=False))

# distance profile: other speakers' EMI at offsets -10..+10 from an X speech (offsets count non-X speeches)
def profile(X, K=10):
    acc = {o: [] for o in range(-K, K + 1) if o != 0}
    for sit, g in d.groupby('Sitting', sort=False):
        party = g.Party.values; emi = g[EMI].values; isx = party == X
        nonx = np.where(~isx)[0]
        for i in np.where(isx)[0]:
            pos = np.searchsorted(nonx, i)
            for o in acc:
                j = pos + o if o > 0 else pos + o          # o>0: o-th non-X speech after; o<0: |o|-th before
                jj = j - 1 if o > 0 else j
                if 0 <= jj < len(nonx): acc[o].append(emi[nonx[jj]])
    return pd.Series({o: np.mean(v) for o, v in acc.items()})
pa, pf, ps = profile('AfD'), profile('FDP'), profile('SPD')
prof = pd.DataFrame({'AfD': pa, 'FDP': pf, 'SPD': ps}); prof.index.name = 'offset'; prof.to_csv(os.path.join(OUT, 'local_window_profile.csv'))
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
for col, c in [('AfD', '#0072B2'), ('FDP', '#D4B300'), ('SPD', '#E3000F')]: ax[0].plot(prof.index, prof[col], 'o-', ms=4, color=c, label=f'around a {col} speech')
ax[0].axvline(0, color='grey', ls=':'); ax[0].set_xlabel('distance in speeches from the speech (negative = before)'); ax[0].set_ylabel('mean EMI of other parties\' speeches'); ax[0].legend(frameon=False); ax[0].grid(alpha=.3); ax[0].set_title('A  Other speakers\' EMI around a speech by …')
sub = R[R.variant == 'all speeches']
for X, c in [('AfD', '#0072B2'), ('FDP', '#D4B300'), ('SPD', '#E3000F'), ('CDU/CSU', '#000000'), ('BÜNDNIS 90/DIE GRÜNEN', '#46962B'), ('DIE LINKE', '#BE3075')]:
    s = sub[sub.party == X]; ax[1].errorbar(s.k + {'AfD': -0.25, 'FDP': -0.15, 'SPD': -0.05, 'CDU/CSU': 0.05, 'BÜNDNIS 90/DIE GRÜNEN': 0.15, 'DIE LINKE': 0.25}[X], s['diff'], yerr=1.96 * s.se, fmt='o', color=c, capsize=2, label=X[:12])
ax[1].axhline(0, color='grey', lw=.8); ax[1].set_xlabel('window size k (speeches before and after)'); ax[1].set_ylabel('after minus before, mean EMI of other parties'); ax[1].legend(frameon=False, fontsize=8, ncol=2); ax[1].grid(alpha=.3); ax[1].set_title('B  Local before/after difference, by party of the speech')
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'local_window.png'), dpi=150, bbox_inches='tight'); plt.savefig(os.path.join(OUT, 'local_window.pdf'), bbox_inches='tight'); print('\nfigure: local_window.png')
