"""Specification curve for the reply to an AfD speech: every combination of outcome, covariate set and estimator,
reported in full. Unit: a non-AfD reply j whose preceding other-party speech is i; treatment: i is an AfD speech.
Outcomes: level = EMI of j; effect = EMI of j minus EMI of the speech before i (the raw 'reply effect').
Covariate sets (cumulative): none; position; +year +party; +EMI of the speech before i (b); +debate register measured
BEFORE the preceding speech (mean EMI of earlier non-AfD speeches in the same item, excluding b); and two over-adjusting
versions shown for completeness: the register including speeches after the reply (post-treatment) and the AfD speech's own EMI.
Estimators: OLS adjustment; speaker FE (+ covariates); coarsened exact matching on the coarsened covariates;
IPW-ATT on a logit propensity score; 1:1 nearest-neighbour matching on the propensity score (caliper 0.02 s.d.).
Output: results/parliament/tables/contagion_spec_curve.csv, results/contagion_spec_curve.png
"""
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))   # German-EMI repository root: data in data/, outputs in results/
import os, numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
os.chdir(REPO_ROOT)
src = open('code/04_analysis/contagion/0411_contagion_sequence.py', encoding='utf-8').read().split('def mdb_mention_network')[0].replace('from matplotlib.patches', 'from matplotlib.patches')
party_colors = {}; exec(src)
EMI = 'emi_w2vparliament'
raw = pd.read_csv('data/revision/bundestag_speeches_2000_2025_raw.csv', usecols=['Source_ID', 'Date', 'Sitting', 'Agenda', 'Party', 'Role'], dtype=str)
raw = raw[(raw.Role == 'Regular') & (raw.Date >= '2017-10-24')]
sc = pd.read_csv('data/parliament_nhb.csv', usecols=['id', EMI, 'politicianId', 'length'], dtype={'id': str})
d = raw.merge(sc, left_on='Source_ID', right_on='id').dropna(subset=[EMI])
d['party'] = d.Party.map(_SEQ_RAW_TO_NHB); d['order'] = pd.to_numeric(d.Source_ID.str.extract(r'ID(\d+)')[0]); d = d.sort_values(['Sitting', 'order']).reset_index(drop=True)
d['item'] = d.Sitting + '|' + d.Agenda.fillna(''); d['pos'] = d.groupby('item').cumcount().clip(upper=6); d['year'] = d.Date.str[:4].astype(int)
rows = []
for sit, g in d.groupby('Sitting', sort=False):
    p, e, pos, pid, item, yr = g.party.values, g[EMI].values, g.pos.values, g.politicianId.values, g.item.values, g.year.values; n = len(g)
    for j in range(n):
        Y = p[j]
        if Y not in _SEQ_PARTIES or Y == 'AFD': continue
        i = next((k for k in range(j - 1, -1, -1) if p[k] != Y), None)
        if i is None or p[i] not in _SEQ_PARTIES: continue
        b = next((k for k in range(i - 1, -1, -1) if p[k] != p[i]), None)
        if b is None: continue
        pre = [e[k] for k in range(n) if item[k] == item[i] and k < i and k != b and p[k] != 'AFD']      # register BEFORE the preceding speech (pre-treatment)
        post = [e[k] for k in range(n) if item[k] == item[j] and k not in (i, j) and p[k] != 'AFD']     # whole item incl. later speeches (post-treatment, over-adjusts)
        rows.append(dict(sitting=sit, pid=pid[j], party=Y, level=e[j], effect=e[j] - e[b], T=int(p[i] == 'AFD'), before=e[b], register=np.mean(pre) if len(pre) >= 2 else np.nan,
                         register_post=np.mean(post) if len(post) >= 2 else np.nan, prev=e[i], pos=pos[j], year=yr[j]))
W = pd.DataFrame(rows).dropna(subset=['pid', 'register', 'register_post']).reset_index(drop=True); W['g'] = pd.factorize(W.sitting)[0]
print(f'replies {len(W):,}, treated {W["T"].sum():,}')
COV = {'none': [], 'position': ['C(pos)'], 'position+year+party': ['C(pos)', 'C(year)', 'C(party)'],
       '+EMI of speech before i': ['C(pos)', 'C(year)', 'C(party)', 'before'], '+debate register (pre-treatment)': ['C(pos)', 'C(year)', 'C(party)', 'before', 'register'],
       '+register incl. later speeches (post-treatment, over-adjusts)': ['C(pos)', 'C(year)', 'C(party)', 'before', 'register_post'],
       '+AfD speech own EMI (over-adjusts)': ['C(pos)', 'C(year)', 'C(party)', 'before', 'register', 'prev']}
COARSE = {'none': [], 'position': ['pos'], 'position+year+party': ['pos', 'year', 'party'], '+EMI of speech before i': ['pos', 'year', 'party', 'q_before'],
          '+debate register (pre-treatment)': ['pos', 'year', 'party', 'q_before', 'q_register'], '+register incl. later speeches (post-treatment, over-adjusts)': ['pos', 'year', 'party', 'q_before', 'q_register_post'],
          '+AfD speech own EMI (over-adjusts)': ['pos', 'year', 'party', 'q_before', 'q_register', 'q_prev']}
for c in ['before', 'register', 'register_post', 'prev']: W['q_' + c] = pd.qcut(W[c], 5, labels=False)
def cl(fit, name='T'): return fit.params[name], fit.bse[name], fit.pvalues[name]
out = []
for y in ['level', 'effect']:
    for cname, cov in COV.items():
        # OLS
        r = smf.ols(f'{y} ~ T' + ''.join(' + ' + c for c in cov), W).fit(cov_type='cluster', cov_kwds={'groups': W.g.values}); est, se, p = cl(r); out.append(dict(outcome=y, covariates=cname, estimator='OLS adjustment', est=est, se=se, p=p, n=int(r.nobs)))
        # speaker FE: demean y and T (and continuous covariates) within speaker, then OLS with categorical covariates
        cont = [c for c in cov if not c.startswith('C(')]; cats = [c for c in cov if c.startswith('C(')]
        Wf = W.copy()
        for c in [y, 'T'] + cont: Wf[c + '_d'] = Wf[c] - Wf.groupby('pid')[c].transform('mean')
        r = smf.ols(f'{y}_d ~ T_d' + ''.join(' + ' + c + '_d' for c in cont) + ''.join(' + ' + c for c in cats), Wf).fit(cov_type='cluster', cov_kwds={'groups': Wf.g.values}); est, se, p = cl(r, 'T_d'); out.append(dict(outcome=y, covariates=cname, estimator='speaker FE', est=est, se=se, p=p, n=int(r.nobs)))
        # CEM
        keys = COARSE[cname]
        if keys:
            Wc = W.copy(); Wc['s'] = Wc[keys].astype(str).agg('|'.join, axis=1); cnt = Wc.groupby('s')['T'].agg(['sum', 'size']); ok = cnt[(cnt['sum'] > 0) & (cnt['sum'] < cnt['size'])].index; Wc = Wc[Wc.s.isin(ok)]
            nt = Wc.groupby('s')['T'].transform('sum'); nc = Wc.groupby('s')['T'].transform('size') - nt; Wc['w'] = np.where(Wc['T'] == 1, 1.0, nt / nc)
            r = sm.WLS(Wc[y], sm.add_constant(Wc['T']), weights=Wc.w).fit(cov_type='cluster', cov_kwds={'groups': Wc.g.values}); est, se, p = cl(r); out.append(dict(outcome=y, covariates=cname, estimator='CEM', est=est, se=se, p=p, n=int(len(Wc)), n_treated=int(Wc['T'].sum())))
        # propensity score: IPW-ATT and 1:1 NN with caliper
        if cov:
            ps = smf.logit('T ~ ' + ' + '.join(cov), W).fit(disp=0); Wp = W.copy(); Wp['ps'] = ps.predict(Wp); Wp = Wp[(Wp.ps > 0.01) & (Wp.ps < 0.99)]
            Wp['w'] = np.where(Wp['T'] == 1, 1.0, Wp.ps / (1 - Wp.ps))
            r = sm.WLS(Wp[y], sm.add_constant(Wp['T']), weights=Wp.w).fit(cov_type='cluster', cov_kwds={'groups': Wp.g.values}); est, se, p = cl(r); out.append(dict(outcome=y, covariates=cname, estimator='IPW-ATT', est=est, se=se, p=p, n=int(len(Wp)), n_treated=int(Wp['T'].sum())))
            # greedy 1:1 nearest neighbour on the logit of the PS without replacement, caliper 0.02 sd
            Wp['lps'] = np.log(Wp.ps / (1 - Wp.ps)); cal = 0.02 * Wp.lps.std()
            tr = Wp[Wp['T'] == 1].sort_values('lps'); co = Wp[Wp['T'] == 0].sort_values('lps'); co_l = co.lps.values; used = np.zeros(len(co), bool); pairs = []
            for t_idx, t_l in zip(tr.index, tr.lps.values):
                k = np.searchsorted(co_l, t_l); best, bd = None, cal
                for kk in (k - 2, k - 1, k, k + 1, k + 2):
                    if 0 <= kk < len(co) and not used[kk] and abs(co_l[kk] - t_l) <= bd: best, bd = kk, abs(co_l[kk] - t_l)
                if best is not None: used[best] = True; pairs.append((t_idx, co.index[best]))
            if len(pairs) > 100:
                ti, ci = zip(*pairs); diffs = Wp.loc[list(ti), y].values - Wp.loc[list(ci), y].values; gg = Wp.loc[list(ti), 'g'].values
                est = diffs.mean(); se = _cluster_se(diffs, gg); p = 2 * (1 - __import__('scipy').stats.norm.cdf(abs(est / se)))
                out.append(dict(outcome=y, covariates=cname, estimator='1:1 NN on PS (caliper)', est=est, se=se, p=p, n=int(2 * len(pairs)), n_treated=int(len(pairs))))
R = pd.DataFrame(out); R['lo'] = R.est - 1.96 * R.se; R['hi'] = R.est + 1.96 * R.se; R.to_csv('results/parliament/tables/contagion_spec_curve.csv', index=False)
pd.set_option('display.width', 220); print(R[['outcome', 'covariates', 'estimator', 'est', 'se', 'p', 'n']].round(3).to_string(index=False))
print(f"\nsignificant negative: {int(((R.p < 0.05) & (R.est < 0)).sum())} of {len(R)}; significant positive: {int(((R.p < 0.05) & (R.est > 0)).sum())}; null: {int((R.p >= 0.05).sum())}")
print("by covariate set, level outcome (mean estimate):\n" + R[R.outcome == 'level'].groupby('covariates').est.mean().round(3).to_string())
# figure
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
cols = {'OLS adjustment': '#4D4D4D', 'speaker FE': '#0072B2', 'CEM': '#D55E00', 'IPW-ATT': '#009E73', '1:1 NN on PS (caliper)': '#CC79A7'}
for ax, y in zip(axes, ['level', 'effect']):
    S = R[R.outcome == y].sort_values('est').reset_index(drop=True)
    for i, r in S.iterrows():
        ax.plot([i, i], [r.lo, r.hi], color=cols[r.estimator], lw=1.2); ax.plot(i, r.est, 'o', color=cols[r.estimator], ms=5, mfc=cols[r.estimator] if 'over-adjusts' in r.covariates else ('grey' if 'pre-treatment' in r.covariates else 'white'))
    ax.axhline(0, color='black', lw=0.8); ax.set_title(f'outcome: reply {y}', loc='left'); ax.set_xlabel('specification, sorted'); ax.set_ylabel('AfD vs other preceding speaker (EMI)'); ax.grid(alpha=.3)
handles = [plt.Line2D([0], [0], marker='o', color=c, label=k, lw=0) for k, c in cols.items()] + [plt.Line2D([0], [0], marker='o', color='grey', mfc='white', label='open: no register control', lw=0), plt.Line2D([0], [0], marker='o', color='grey', mfc='grey', label='grey: pre-treatment register', lw=0), plt.Line2D([0], [0], marker='o', color='grey', label='filled: post-treatment covariate (over-adjusts)', lw=0)]
axes[0].legend(handles=handles, frameon=False, fontsize=8, loc='upper left'); fig.tight_layout(); fig.savefig('results/contagion_spec_curve.png', dpi=200, bbox_inches='tight'); print('figure: results/contagion_spec_curve.png')
