"""Fine-grained timing of the post-2017 shift and the exposure (electoral threat) hypothesis.

Timing at speech resolution: for each continuing MdB, the first date after the constitution of the 19th Bundestag
(24 Oct 2017) on which the running mean of their last five speeches falls below their own pre-2017 level minus half a
within-person SD (threshold as in the quarterly version). Time-to-adoption in days; censored at 31 Dec 2019.
Exposure: AfD second-vote share 2017 in the MdB's constituency, its change since 2013, AfD minus own party share.
Tests: tercile comparison (overall and within party), Spearman correlations, Cox proportional hazards stratified by
party (statsmodels PHReg), monthly event study with MdB and party x month effects, and the drop (extent) on exposure.
Output: results/revision/09_mechanism_2017/finegrained_*.csv/png
"""
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))   # German-EMI repository root: data in data/, outputs in results/
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.duration.hazard_regression import PHReg
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = REPO_ROOT; OUT = os.path.join(REPO_ROOT, 'results', 'revision', '09_mechanism_2017')
EMI = 'emi_w2vparliament'; T0 = pd.Timestamp('2017-10-24', tz='UTC'); TEND = pd.Timestamp('2019-12-31', tz='UTC')
par = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'parliament_nhb.csv'), usecols=['date', 'politicianId', 'party', EMI]).dropna(subset=['politicianId', EMI])
par['politicianId'] = par.politicianId.astype(int).astype(str); par['date'] = pd.to_datetime(par.date, utc=True); par = par.sort_values('date')
panel = pd.read_csv(os.path.join(OUT, 'panel_2017.csv'), dtype={'politicianId': str})
exp = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'revision', 'mdb_exposure_wp19.csv'), dtype={'politicianId': str})
p = panel[['politicianId', 'party', 'dist_afd', 'change', 'emi_pre', 'emi_post']].merge(
    exp[['politicianId', 'afd_share_2017', 'afd_change', 'afd_minus_own', 'east', 'direct', 'land']], on='politicianId', how='left')
p['drop'] = -p.change

# --- speech-level time to adoption ----------------------------------------------------------------
q = par[par.politicianId.isin(p.politicianId)].copy(); q['q'] = q.date.dt.tz_localize(None).dt.to_period('Q')
pre_q = q[(q.q >= pd.Period('2015Q1')) & (q.q <= pd.Period('2017Q2'))].groupby(['politicianId', 'q'])[EMI].mean().groupby('politicianId').agg(['mean', 'std', 'size'])
gsd = pre_q['std'].median()
rows = []
for pid, g in q[q.date >= T0].groupby('politicianId'):
    if pid not in pre_q.index or pre_q.loc[pid, 'size'] < 3: continue
    thr = pre_q.loc[pid, 'mean'] - 0.5 * (pre_q.loc[pid, 'std'] if pre_q.loc[pid, 'std'] > 0 else gsd)
    g = g[g.date <= TEND].sort_values('date'); roll = g[EMI].rolling(5, min_periods=3).mean()
    hit = g.date[roll < thr]
    if len(hit): rows.append(dict(politicianId=pid, days=(hit.iloc[0] - T0).days, event=1, n_post=len(g)))
    else: rows.append(dict(politicianId=pid, days=(min(g.date.max(), TEND) - T0).days if len(g) else 0, event=0, n_post=len(g)))
tt = pd.DataFrame(rows); p = p.merge(tt, on='politicianId', how='inner'); p = p[p.n_post >= 5]
for c in ['afd_share_2017', 'afd_change', 'afd_minus_own']:
    p[f'{c}_z'] = (p[c] - p[c].mean()) / p[c].std(); p[f'{c}_wz'] = p.groupby('party')[c].transform(lambda s: (s - s.mean()) / s.std())
print(f'N = {len(p)} | adopted by end-2019: {p.event.mean():.0%} | median days to adoption among adopters: {p[p.event==1].days.median():.0f}')
print('days to adoption quantiles (adopters):', p[p.event == 1].days.quantile([.1, .25, .5, .75, .9]).round(0).to_dict())

# --- 1. terciles of exposure, overall and within party --------------------------------------------
res = []
def tert(d, col, label):
    d = d.dropna(subset=[col]).copy(); d['tercile'] = pd.qcut(d[col], 3, labels=['low', 'mid', 'high'])
    t = d.groupby('tercile', observed=True).agg(n=('drop', 'size'), mean_drop=('drop', 'mean'), adopted=('event', 'mean'),
                                                median_days=('days', lambda s: s[d.loc[s.index, 'event'] == 1].median()))
    lo, hi = d[d.tercile == 'low'], d[d.tercile == 'high']
    tw = stats.ttest_ind(lo['drop'], hi['drop'], equal_var=False); fe = stats.fisher_exact([[lo.event.sum(), len(lo) - lo.event.sum()], [hi.event.sum(), len(hi) - hi.event.sum()]])[1]
    print(f'\n{label} ({col}):'); print(t.round(3).to_string())
    print(f'   high vs low tercile: drop {hi["drop"].mean():.3f} vs {lo["drop"].mean():.3f} (p={tw.pvalue:.2f}); adopted {hi.event.mean():.0%} vs {lo.event.mean():.0%} (p={fe:.2f}); '
          f'median days {hi[hi.event==1].days.median():.0f} vs {lo[lo.event==1].days.median():.0f}')
    res.append(dict(level=label, exposure=col, drop_high=hi['drop'].mean(), drop_low=lo['drop'].mean(), p_drop=tw.pvalue, adopt_high=hi.event.mean(), adopt_low=lo.event.mean(), p_adopt=fe))
for col in ['afd_share_2017', 'afd_change']:
    tert(p, col, 'ALL non-AfD MdBs')
for pty, g in p.groupby('party'):
    if len(g) >= 40: tert(g, 'afd_share_2017', pty)

# --- 2. correlations and Cox model -------------------------------------------------------------------
print('\nCorrelations (Spearman):')
for col in ['afd_share_2017', 'afd_change', 'afd_minus_own', 'dist_afd']:
    d = p.dropna(subset=[col]); a = d[d.event == 1]
    print(f'   {col:16s} drop: rho={stats.spearmanr(d["drop"], d[col]).correlation:+.3f} (p={stats.spearmanr(d["drop"], d[col]).pvalue:.2f}) | days to adoption: rho={stats.spearmanr(a.days, a[col]).correlation:+.3f} (p={stats.spearmanr(a.days, a[col]).pvalue:.2f})')
print('\nCox proportional hazards of adoption (stratified by party; hazard ratio per SD; HR > 1 = faster):')
cox = []
for col in ['afd_share_2017_z', 'afd_change_z', 'afd_minus_own_z', 'afd_share_2017_wz']:
    d = p.dropna(subset=[col, 'east'])
    m = PHReg(d.days.values + 0.5, d[[col, 'east', 'direct']].values.astype(float), status=d.event.values, strata=d.party.values).fit()
    hr, lo_, hi_ = np.exp(m.params[0]), np.exp(m.params[0] - 1.96 * m.bse[0]), np.exp(m.params[0] + 1.96 * m.bse[0])
    print(f'   {col:20s} HR = {hr:.2f} [{lo_:.2f}, {hi_:.2f}]  p = {m.pvalues[0]:.2f}  n = {len(d)}')
    cox.append(dict(exposure=col, hr=hr, lo=lo_, hi=hi_, p=m.pvalues[0], n=len(d)))
pd.DataFrame(cox).to_csv(os.path.join(OUT, 'finegrained_cox.csv'), index=False)

# --- 3. monthly event study: EMI_im = a_i + g_party,m + b * exposure * post + e ----------------------
mm = q[(q.date >= '2016-01-01') & (q.date <= '2019-12-31')].copy(); mm['m'] = mm.date.dt.tz_localize(None).dt.to_period('M')
mm = mm.groupby(['politicianId', 'm'])[EMI].agg(['mean', 'size']).reset_index().rename(columns={'mean': 'y'}); mm = mm[mm['size'] >= 1]
mm = mm.merge(p[['politicianId', 'party', 'afd_share_2017_wz', 'afd_change_z', 'dist_afd']], on='politicianId'); mm['ms'] = mm.m.astype(str)
mm['post'] = (mm.m >= pd.Period('2017-10')).astype(int)
print('\nMonthly event study (MdB FE + party x month FE), post x exposure:')
for col in ['afd_share_2017_wz', 'afd_change_z']:
    d = mm.dropna(subset=[col]).copy(); d['x_post'] = d[col] * d.post
    mod = smf.ols('y ~ C(politicianId) + C(party):C(ms) + x_post', data=d); r = mod.fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d.politicianId.loc[mod.data.row_labels])[0]})
    print(f'   {col:20s} coef = {r.params["x_post"]:+.4f} (se {r.bse["x_post"]:.4f}, p = {r.pvalues["x_post"]:.2f}), MdB-months = {int(r.nobs)}')
    # speed version: exposure x months-since-entry within the first year
    d['since'] = ((d.m - pd.Period('2017-10')).apply(lambda x: x.n)).clip(lower=0, upper=12) / 12.0; d['x_since'] = d[col] * d.since * d.post
    mod2 = smf.ols('y ~ C(politicianId) + C(party):C(ms) + x_post + x_since', data=d); r2 = mod2.fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d.politicianId.loc[mod2.data.row_labels])[0]})
    print(f'   {"":20s} with ramp over first year: post x exposure = {r2.params["x_post"]:+.4f} (p={r2.pvalues["x_post"]:.2f}), ramp x exposure = {r2.params["x_since"]:+.4f} (p={r2.pvalues["x_since"]:.2f})')

# --- 4. extent on exposure, simple OLS -----------------------------------------------------------------
for col in ['afd_share_2017_z', 'afd_change_z', 'afd_minus_own_z']:
    d = p.dropna(subset=[col, 'east']); r = smf.ols(f'drop ~ {col} + C(party) + east + direct', data=d).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d.land.fillna('NA'))[0]})
    print(f'drop ~ {col:18s}: {r.params[col]:+.4f} per SD (p = {r.pvalues[col]:.2f}, n = {int(r.nobs)})')
p.to_csv(os.path.join(OUT, 'finegrained_panel.csv'), index=False); pd.DataFrame(res).to_csv(os.path.join(OUT, 'finegrained_terciles.csv'), index=False)

# --- figure: Kaplan-Meier-style adoption curves by exposure tercile, and drop vs exposure ---------------
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
d = p.dropna(subset=['afd_share_2017']).copy(); d['tercile'] = pd.qcut(d.afd_share_2017, 3, labels=['low AfD share', 'mid', 'high AfD share'])
grid = np.arange(0, 800, 7)
for lab, col in [('low AfD share', '#56B4E9'), ('mid', 'grey'), ('high AfD share', '#D55E00')]:
    g = d[d.tercile == lab]; curve = [((g.days <= t) & (g.event == 1)).mean() for t in grid]
    ax[0].step(grid, curve, where='post', color=col, lw=2, label=f'{lab} (n={len(g)})')
ax[0].set_xlabel('days since the 19th Bundestag convened (24 Oct 2017)'); ax[0].set_ylabel('share of MdBs that have adopted'); ax[0].legend(frameon=False); ax[0].grid(alpha=.3); ax[0].set_title('A  Speed of adoption by constituency AfD exposure')
PC = {'CDU/CSU': '#000000', 'SPD': '#E3000F', 'Greens': '#46962B', 'Left': '#BE3075'}
for pty, g in d.groupby('party'):
    ax[1].scatter(g.afd_share_2017, g['drop'], s=14, alpha=.5, color=PC[pty], label=f'{pty} (r={stats.pearsonr(g["drop"], g.afd_share_2017)[0]:+.2f})')
    b = np.polyfit(g.afd_share_2017, g['drop'], 1); xs = np.linspace(g.afd_share_2017.min(), g.afd_share_2017.max(), 10); ax[1].plot(xs, np.polyval(b, xs), color=PC[pty], lw=2)
ax[1].axhline(0, color='grey', lw=.8); ax[1].set_xlabel('AfD second-vote share 2017 in own constituency'); ax[1].set_ylabel('drop in EMI after 2017'); ax[1].legend(frameon=False, fontsize=9); ax[1].grid(alpha=.3); ax[1].set_title('B  Size of the drop by exposure')
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'finegrained_exposure.png'), dpi=150, bbox_inches='tight'); plt.savefig(os.path.join(OUT, 'finegrained_exposure.pdf'), bbox_inches='tight'); print('\nfigure: finegrained_exposure.png')
