"""Per-person ideology (Breunig & Guinaudeau 2025 positions, WP19) and EMI.
 1. LEVEL: do MdBs closer to the AfD centroid have higher (or lower) EMI? Per-MdB mean EMI in WP18 (pre-entry) and WP19 (post-entry),
    regressed on distance to the AfD centroid: pooled, with party fixed effects, and within each party.
 2. SHOCK: was the pre/post-entry drop bigger for MdBs closer to the AfD? Change = mean EMI in the six quarters after entry
    (2017Q4-2019Q1) minus the six quarters before (2016Q1-2017Q2), regressed on distance: pooled, party FE, within party, terciles.
Output: results/09_mechanism_2017/ideology_level_shock.{csv,png,pdf}, ideology_level_shock_numbers.json
"""
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))   # German-EMI repository root: data in data/, outputs in results/
import os, json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt, statsmodels.api as sm, statsmodels.formula.api as smf
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = REPO_ROOT; OUT = os.path.join(REPO_ROOT, 'results', 'revision', '09_mechanism_2017')
EMI = 'emi_w2vparliament'
sc = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'parliament_nhb.csv'), usecols=['date', 'politicianId', 'party', EMI]).dropna(subset=['politicianId', EMI])
sc['politicianId'] = sc.politicianId.astype(int); sc['date'] = pd.to_datetime(sc.date, utc=True).dt.tz_localize(None); sc['q'] = sc.date.dt.to_period('Q')
pos = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'revision', 'breunig_guinaudeau', 'mp_positions_wp19.csv'))[['politicianId', 'party', 'position', 'dist_afd']].rename(columns={'party': 'party_bg'})
pos['dist_sd'] = (pos.dist_afd - pos.dist_afd.mean()) / pos.dist_afd.std()
sc = sc.merge(pos, on='politicianId'); print(f'speeches with a position: {len(sc):,} by {sc.politicianId.nunique()} MdBs')
MIN_N = 5
def per_mdb(lo, hi):
    d = sc[(sc.date >= lo) & (sc.date < hi)]; g = d.groupby('politicianId').agg(emi=(EMI, 'mean'), n=(EMI, 'size'))
    return g[g.n >= MIN_N]
def fit(formula, d, w=None):
    m = smf.wls(formula, d, weights=w) if w is not None else smf.ols(formula, d); r = m.fit(cov_type='HC1'); return r
def row(label, r, term='dist_sd', n=None): return dict(test=label, coef=r.params[term], se=r.bse[term], lo=r.conf_int().loc[term, 0], hi=r.conf_int().loc[term, 1], p=r.pvalues[term], n=int(r.nobs))
rows = []; NUM = {}
P = pos.set_index('politicianId')

# ---------------- 1. LEVEL ------------------------------------------------------------------------------
print('\n1) LEVEL: per-MdB mean EMI vs distance to the AfD centroid (coef per SD of distance = 4.3 units; positive = farther from AfD -> higher EMI)')
for tag, lo, hi in [('WP18 (2013-10 to 2017-09, before entry)', '2013-10-22', '2017-09-25'), ('WP19 (2017-10 to 2021-09, after entry)', '2017-10-24', '2021-09-27')]:
    L = per_mdb(lo, hi).join(P); L['party'] = L.party_bg
    non = L[L.party != 'AfD']
    r_all = fit('emi ~ dist_sd', L); r_non = fit('emi ~ dist_sd', non); r_fe = fit('emi ~ dist_sd + C(party)', non)
    print(f'   {tag}: n={len(L)} MdBs ({len(non)} non-AfD)')
    print(f'      pooled incl. AfD       : {r_all.params.dist_sd:+.3f} (se {r_all.bse.dist_sd:.3f}, p={r_all.pvalues.dist_sd:.3f}) | AfD mean {L[L.party=="AfD"].emi.mean():+.2f} vs others {non.emi.mean():+.2f}')
    print(f'      pooled non-AfD         : {r_non.params.dist_sd:+.3f} (se {r_non.bse.dist_sd:.3f}, p={r_non.pvalues.dist_sd:.3f})')
    print(f'      non-AfD, party FE      : {r_fe.params.dist_sd:+.3f} (se {r_fe.bse.dist_sd:.3f}, p={r_fe.pvalues.dist_sd:.3f})')
    rows += [row(f'level {tag}: pooled incl. AfD', r_all), row(f'level {tag}: pooled non-AfD', r_non), row(f'level {tag}: non-AfD party FE', r_fe)]
    for p_, g in L.groupby('party'):
        if len(g) < 15: continue
        rho, pv = spearmanr(g.dist_afd, g.emi); r_p = fit('emi ~ dist_sd', g)
        print(f'         within {p_:8s} n={len(g):3d}: slope {r_p.params.dist_sd:+.3f} (p={r_p.pvalues.dist_sd:.2f}), Spearman rho {rho:+.2f} (p={pv:.2f})')
        rows.append(dict(test=f'level {tag}: within {p_}', coef=r_p.params.dist_sd, se=r_p.bse.dist_sd, lo=r_p.conf_int().loc['dist_sd', 0], hi=r_p.conf_int().loc['dist_sd', 1], p=r_p.pvalues.dist_sd, n=len(g), rho=rho, rho_p=pv))
    NUM[f'level_{tag[:4]}'] = dict(n=len(L), pooled_all=r_all.params.dist_sd, pooled_non=r_non.params.dist_sd, party_fe=r_fe.params.dist_sd, party_fe_p=r_fe.pvalues.dist_sd)
    if tag.startswith('WP19'): L19 = L
    else: L18 = L
# left-right position instead of distance (within non-AfD, party FE)
r_lr = fit('emi ~ position + C(party)', L19[L19.party != 'AfD']); print(f'      WP19, left-right position (per unit, party FE): {r_lr.params.position:+.4f} (p={r_lr.pvalues.position:.3f})')

# ---------------- 2. SHOCK ------------------------------------------------------------------------------
print('\n2) SHOCK: change in mean EMI, six quarters after entry (2017Q4-2019Q1) minus six quarters before (2016Q1-2017Q2), non-AfD MdBs')
pre = per_mdb('2016-01-01', '2017-07-01').rename(columns={'emi': 'emi_pre', 'n': 'n_pre'}); post = per_mdb('2017-10-24', '2019-04-01').rename(columns={'emi': 'emi_post', 'n': 'n_post'})
S = pre.join(post, how='inner').join(P); S['party'] = S.party_bg; S = S[S.party != 'AfD'].copy(); S['change'] = S.emi_post - S.emi_pre; S['w'] = np.minimum(S.n_pre, S.n_post)
print(f'   n={len(S)} MdBs with >= {MIN_N} speeches in both windows; mean change {S.change.mean():+.3f} (sd {S.change.std():.2f}); {(S.change<0).mean():.0%} fell')
r_s = fit('change ~ dist_sd', S); r_sw = fit('change ~ dist_sd', S, S.w); r_sfe = fit('change ~ dist_sd + C(party)', S); r_sfew = fit('change ~ dist_sd + C(party)', S, S.w); r_pre = fit('change ~ dist_sd + emi_pre + C(party)', S)
for lab, r in [('pooled', r_s), ('pooled, weighted by speeches', r_sw), ('party FE', r_sfe), ('party FE, weighted', r_sfew), ('party FE + pre level', r_pre)]:
    print(f'      {lab:30s}: {r.params.dist_sd:+.3f} per SD (se {r.bse.dist_sd:.3f}, 95% CI {r.conf_int().loc["dist_sd",0]:+.3f} to {r.conf_int().loc["dist_sd",1]:+.3f}, p={r.pvalues.dist_sd:.3f})'); rows.append(row(f'shock: {lab}', r))
for p_, g in S.groupby('party'):
    if len(g) < 15: continue
    rho, pv = spearmanr(g.dist_afd, g.change); r_p = fit('change ~ dist_sd', g)
    print(f'         within {p_:8s} n={len(g):3d}: slope {r_p.params.dist_sd:+.3f} (p={r_p.pvalues.dist_sd:.2f}), rho {rho:+.2f} (p={pv:.2f}), mean change {g.change.mean():+.2f}')
    rows.append(dict(test=f'shock: within {p_}', coef=r_p.params.dist_sd, se=r_p.bse.dist_sd, lo=r_p.conf_int().loc['dist_sd', 0], hi=r_p.conf_int().loc['dist_sd', 1], p=r_p.pvalues.dist_sd, n=len(g), rho=rho, rho_p=pv, mean_change=g.change.mean()))
# full-WP change (WP19 mean minus WP18 mean) as a robustness window
W = L18[['emi']].rename(columns={'emi': 'e18'}).join(L19[['emi']].rename(columns={'emi': 'e19'}), how='inner').join(P); W['party'] = W.party_bg; W = W[W.party != 'AfD']; W['change'] = W.e19 - W.e18
r_w = fit('change ~ dist_sd + C(party)', W); print(f'   robustness, WP19 mean minus WP18 mean (n={len(W)}), party FE: {r_w.params.dist_sd:+.3f} (p={r_w.pvalues.dist_sd:.3f})'); rows.append(row('shock: WP19 minus WP18, party FE', r_w))
# left-right position
r_lrs = fit('change ~ position + C(party)', S); print(f'   left-right position instead of distance (party FE): {r_lrs.params.position:+.4f} per unit (p={r_lrs.pvalues.position:.3f})')
NUM['shock'] = dict(n=len(S), mean_change=S.change.mean(), pooled=r_s.params.dist_sd, pooled_p=r_s.pvalues.dist_sd, party_fe=r_sfe.params.dist_sd, party_fe_lo=r_sfe.conf_int().loc['dist_sd', 0], party_fe_hi=r_sfe.conf_int().loc['dist_sd', 1], party_fe_p=r_sfe.pvalues.dist_sd)
# ---------------- continuous only: correlations with distance and left-right position ----------------------
print('\n3) CONTINUOUS: Pearson r / Spearman rho of per-MdB EMI with distance to the AfD and with left-right position (no party classes)')
from scipy.stats import pearsonr
for lab, d, ycol in [('level before entry (WP18)', L18, 'emi'), ('level after entry (WP19), non-AfD', L19[L19.party != 'AfD'], 'emi'), ('level after entry (WP19), incl. AfD', L19, 'emi'), ('change after minus before', S, 'change')]:
    for xcol in ['dist_afd', 'position']:
        r, pr = pearsonr(d[xcol], d[ycol]); rho, ps = spearmanr(d[xcol], d[ycol])
        print(f'   {lab:38s} ~ {xcol:9s}: n={len(d):3d}  r={r:+.3f} (p={pr:.3f})  rho={rho:+.3f} (p={ps:.3f})')
    NUM[f'cont_{ycol}_{lab[:12]}'] = dict(n=len(d), r_dist=pearsonr(d.dist_afd, d[ycol])[0], p_dist=pearsonr(d.dist_afd, d[ycol])[1], r_pos=pearsonr(d.position, d[ycol])[0], p_pos=pearsonr(d.position, d[ycol])[1])
r_q = fit('change ~ dist_afd + I(dist_afd**2)', S); print(f'   change ~ distance + distance^2: lin {r_q.params.dist_afd:+.4f} (p={r_q.pvalues.dist_afd:.2f}), quad {r_q.params["I(dist_afd ** 2)"]:+.4f} (p={r_q.pvalues["I(dist_afd ** 2)"]:.2f})')
pd.DataFrame(rows).to_csv(os.path.join(OUT, 'ideology_level_shock.csv'), index=False); json.dump(NUM, open(os.path.join(OUT, 'ideology_level_shock_numbers.json'), 'w'), indent=1, default=float)

# ---------------- figure ---------------------------------------------------------------------------------
PC = {'CDU/CSU': '#1A1A1A', 'SPD': '#E3000F', 'FDP': '#C9A800', 'Greens': '#46962B', 'Left': '#BE3075', 'AfD': '#009EE0'}
fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))
for a, d, ycol, ttl in [(ax[0], L18, 'emi', 'A  Level before entry (WP18 mean EMI)'), (ax[1], L19, 'emi', 'B  Level after entry (WP19 mean EMI)'), (ax[2], S, 'change', 'C  Change: 6 quarters after minus 6 before')]:
    for p_, g in d.groupby('party'):
        a.scatter(g.dist_afd, g[ycol], s=12, color=PC.get(p_, 'grey'), alpha=0.6, label=f'{p_} (n={len(g)})')
        if len(g) >= 15 and p_ != 'AfD':
            b = np.polyfit(g.dist_afd, g[ycol], 1); xx = np.array([g.dist_afd.min(), g.dist_afd.max()]); a.plot(xx, np.polyval(b, xx), color=PC.get(p_, 'grey'), lw=1.5)
    a.axhline(0, color='grey', lw=0.6); a.set_xlabel('distance to AfD centroid (Breunig & Guinaudeau)'); a.set_title(ttl, loc='left', fontsize=10); a.grid(alpha=.3)
ax[0].set_ylabel('mean EMI'); ax[2].set_ylabel('change in mean EMI'); ax[0].legend(frameon=False, fontsize=7); ax[2].legend(frameon=False, fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'ideology_level_shock.png'), dpi=150, bbox_inches='tight'); plt.savefig(os.path.join(OUT, 'ideology_level_shock.pdf'), bbox_inches='tight'); print('\nfigure: ideology_level_shock.png')
