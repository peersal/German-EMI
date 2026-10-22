"""Seven-election breakpoint model and pre/post comparison per party, Bundestag 2000-2025.

Same specification as revision.ipynb section 2.3 (speech-level EMI standardised over 2000-2025, linear trend,
level shift + slope change at each federal election, the speaker's mean EMI in the preceding quarter, random
speaker intercepts, ML), estimated separately for each party. Elections at which a party had no speeches on one
side (FDP 2013 and 2017: not in the 18th Bundestag; AfD before 2017) are dropped from that party's model.
Also the six-quarters-before / six-quarters-after comparison per party and election.

Outputs (results/02_pre2015/): breakpoint7_by_party.csv, prepost_by_party.csv, fig_supp_afd_by_party.{pdf,png}
"""
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))   # German-EMI repository root: data in data/, outputs in results/
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from statsmodels.regression.mixed_linear_model import MixedLM
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(REPO_ROOT, 'results', 'revision', '02_pre2015')
PLOT_LABEL_SIZE, PLOT_LEGEND_SIZE, PLOT_PANEL_SIZE, PLOT_TICK_SIZE = 18, 13, 18, 13
plt.rcParams.update({'font.family': 'Arial', 'axes.spines.top': False, 'axes.spines.right': False})
PARTY_COL = {'CDU/CSU': '#000000', 'SPD': '#E3000F', 'FDP': '#E0B400', 'Alliance 90/The Greens': '#46962B', 'The Left Party': '#BE3075', 'AFD': '#009EE0'}
SHORT = {'CDU/CSU': 'CDU/CSU', 'SPD': 'SPD', 'FDP': 'FDP', 'Alliance 90/The Greens': 'Greens', 'The Left Party': 'Left', 'AFD': 'AfD'}
PARTIES = ['CDU/CSU', 'SPD', 'FDP', 'Alliance 90/The Greens', 'The Left Party', 'AFD']
ELECTIONS = {'elec2002': '2002-09-22', 'elec2005': '2005-09-18', 'elec2009': '2009-09-27', 'elec2013': '2013-09-22', 'elec2017': '2017-09-24', 'elec2021': '2021-09-26', 'elec2025': '2025-02-23'}
BP7 = {k: pd.Timestamp(f'{v[:4]}-07-01' if k != 'elec2025' else '2025-01-01', tz='UTC') for k, v in ELECTIONS.items()}
HL = '#D55E00'

full = pd.read_csv(os.path.join(D, 'bundestag_2000_2025_emi_harmonised.csv'), usecols=['date', 'politicianId', 'party', 'emi_full', 'time_quarterly'], dtype={'politicianId': str})
full['date'] = pd.to_datetime(full.date, utc=True); full = full.dropna(subset=['emi_full', 'party'])
print('speeches', len(full))


def prep7(d, bps, y='emi_full'):
    m = d[[y, 'time_quarterly', 'politicianId']].rename(columns={y: 'y'}).copy()
    qm = m.groupby(['politicianId', 'time_quarterly'])['y'].mean().rename('_q').reset_index()
    qm['y_lag1'] = qm.groupby('politicianId')['_q'].shift(1)
    m = m.merge(qm[['politicianId', 'time_quarterly', 'y_lag1']], on=['politicianId', 'time_quarterly']).dropna()
    tm = m.time_quarterly.mean(); m['time_centered'] = m.time_quarterly - tm
    for k, ts in bps.items():
        bq = int(f'{ts.year}{ts.quarter}')
        m[f'slope_{k}'] = np.maximum(0, m.time_centered - (bq - tm)); m[f'level_{k}'] = (m.time_quarterly >= bq).astype(float)
    return m.reset_index(drop=True)


rows, pp = [], []
for p in PARTIES:
    d = full[full.party == p]
    # elections usable for this party: speeches in both the two years before and after the election
    bps = {k: ts for k, ts in BP7.items() if (d.date < ts).sum() > 500 and (d.date >= ts).sum() > 100 and
           ((d.date >= ts - pd.Timedelta(days=730)) & (d.date < ts)).sum() > 200 and ((d.date >= ts) & (d.date < ts + pd.Timedelta(days=730))).sum() > 100}
    if not bps: continue
    m = prep7(d, bps)
    f = 'y ~ time_centered + ' + ' + '.join([f'slope_{k} + level_{k}' for k in bps]) + ' + y_lag1'
    r = MixedLM.from_formula(f, groups=m.politicianId, data=m).fit(reml=False)
    ci = r.conf_int()
    for k in bps:
        rows.append(dict(party=p, election=k.replace('elec', ''), level=r.params[f'level_{k}'], se=r.bse[f'level_{k}'], p=r.pvalues[f'level_{k}'], lo=ci.loc[f'level_{k}', 0], hi=ci.loc[f'level_{k}', 1],
                         slope=r.params[f'slope_{k}'], slope_p=r.pvalues[f'slope_{k}'], n=int(r.nobs), speakers=int(m.politicianId.nunique())))
    print(f"{p:24s} n={int(r.nobs):,} speakers={m.politicianId.nunique():,}  " + '  '.join(f"{k[4:]}: {r.params[f'level_{k}']:+.3f}" for k in bps))
    # pre/post (six full quarters either side, skipping the two transition quarters)
    for k, ds in ELECTIONS.items():
        e = pd.Timestamp(ds, tz='UTC'); eq = e.to_period('Q'); q = d.date.dt.to_period('Q')
        pre = d[(q < eq) & (q >= eq - 6)]; post = d[(q > eq + 1) & (q <= eq + 7)]
        if len(pre) >= 100 and len(post) >= 100:
            pp.append(dict(party=p, election=k.replace('elec', ''), n_pre=len(pre), n_post=len(post), mean_pre=pre.emi_full.mean(), mean_post=post.emi_full.mean(), diff=post.emi_full.mean() - pre.emi_full.mean()))
T = pd.DataFrame(rows); T.to_csv(os.path.join(D, 'breakpoint7_by_party.csv'), index=False)
P = pd.DataFrame(pp); P.to_csv(os.path.join(D, 'prepost_by_party.csv'), index=False)
print(T.pivot(index='party', columns='election', values='level').round(3))
print(P.pivot(index='party', columns='election', values='diff').round(3))

# ---- figure: one forest panel per party ----
parties = [p for p in PARTIES if p in set(T.party)]
fig, axes = plt.subplots(1, len(parties), figsize=(3.6 * len(parties), 5.2), sharex=True, sharey=True)
years = [k.replace('elec', '') for k in BP7]; yy = {y: i for i, y in enumerate(years[::-1])}
for ax, p in zip(axes, parties):
    t = T[T.party == p]
    for _, r in t.iterrows():
        c = HL if r.election == '2017' else PARTY_COL[p]
        ax.errorbar(r.level, yy[r.election], xerr=[[r.level - r.lo], [r.hi - r.level]], fmt='o', color=c, capsize=3, ms=8 if r.election == '2017' else 6, lw=1.8)
    for y in years:
        if y not in set(t.election): ax.text(0.0, yy[y], 'n/a', ha='center', va='center', fontsize=PLOT_TICK_SIZE - 2, color='0.5')
    ax.axvline(0, color='grey', lw=0.8); ax.set_title(SHORT[p], color=PARTY_COL[p], fontsize=PLOT_LABEL_SIZE, fontweight='bold')
    ax.set_yticks(list(yy.values())); ax.set_yticklabels(list(yy.keys())); ax.tick_params(labelsize=PLOT_TICK_SIZE); ax.grid(axis='x', alpha=0.25)
axes[0].set_ylabel('Federal election', fontsize=PLOT_LABEL_SIZE - 2, fontweight='bold')
fig.supxlabel('Level shift at the election (EMI, 95% CI)', fontsize=PLOT_LABEL_SIZE - 2, fontweight='bold')
fig.tight_layout()
for ext in ('pdf', 'png'):
    out = os.path.join(D, f'fig_supp_afd_by_party.{ext}')
    try: fig.savefig(out, dpi=300, bbox_inches='tight')
    except PermissionError: fig.savefig(out.replace(f'.{ext}', f'_tmp.{ext}'), dpi=300, bbox_inches='tight'); print('locked, wrote _tmp')
print('written', D)
