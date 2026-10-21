"""Matched / propensity-weighted estimates of the reply to an AfD speech.
Unit: a non-AfD speech j whose preceding speech by another party is i. Treatment: i is an AfD speech.
Outcome: EMI of j (level) and the reply effect (EMI_j minus EMI of the speech before i by another party).
Covariates: EMI of i (register of the speech being replied to), position of j within the agenda item, year,
party of j, speaker of j. Estimators: (a) OLS with the covariates and speaker fixed effects, (b) coarsened exact
matching on party x position x year x quintile of EMI_i, (c) IPW-ATT on a logistic propensity score.
Standard errors clustered by sitting. Output: results/parliament/tables/contagion_matching.csv
"""
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))   # German-EMI repository root: data in data/, outputs in results/
import os, re, numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
os.chdir(REPO_ROOT)
src = open('code/04_analysis/contagion/0411_contagion_sequence.py', encoding='utf-8').read().split('def mdb_mention_network')[0].replace('from matplotlib.patches', 'import matplotlib; matplotlib.use("Agg")\nfrom matplotlib.patches')
party_colors = {}; exec(src)
EMI = 'emi_w2vparliament'
raw = pd.read_csv('data/revision/bundestag_speeches_2000_2025_raw.csv', usecols=['Source_ID', 'Date', 'Sitting', 'Agenda', 'Party', 'Role'], dtype=str)
raw = raw[(raw.Role == 'Regular') & (raw.Date >= '2017-10-24')]
sc = pd.read_csv('data/parliament_nhb.csv', usecols=['id', EMI, 'politicianId', 'text', 'length'], dtype={'id': str})
sc['m'] = sc.text.fillna('').str.lower().str.contains(r'\bafd\b|alternative für deutschland').astype(int)
d = raw.merge(sc[['id', EMI, 'politicianId', 'm', 'length']], left_on='Source_ID', right_on='id').dropna(subset=[EMI])
d['party'] = d.Party.map(_SEQ_RAW_TO_NHB); d['order'] = pd.to_numeric(d.Source_ID.str.extract(r'ID(\d+)')[0]); d = d.sort_values(['Sitting', 'order']).reset_index(drop=True)
d['item'] = d.Sitting + '|' + d.Agenda.fillna(''); d['pos'] = d.groupby('item').cumcount().clip(upper=6); d['year'] = d.Date.str[:4].astype(int)
rows = []
for sit, g in d.groupby('Sitting', sort=False):
    p, e, pos, pid, m, yr, ln = g.party.values, g[EMI].values, g.pos.values, g.politicianId.values, g.m.values, g.year.values, g.length.values; n = len(g)
    for j in range(n):
        Y = p[j]
        if Y not in _SEQ_PARTIES or Y == 'AFD': continue
        i = next((k for k in range(j - 1, -1, -1) if p[k] != Y), None)
        if i is None or p[i] not in _SEQ_PARTIES: continue
        b = next((k for k in range(i - 1, -1, -1) if p[k] != p[i]), None)
        rows.append(dict(sitting=sit, pid=pid[j], party=Y, y=e[j], prev_afd=int(p[i] == 'AFD'), prev_emi=e[i], prev_party=p[i], pos=pos[j], year=yr[j],
                         names_afd=m[j], diff=(e[j] - e[b]) if b is not None else np.nan, loglen=np.log1p(ln[j]) if not np.isnan(ln[j]) else np.nan))
W = pd.DataFrame(rows).dropna(subset=['pid']); W['g'] = pd.factorize(W.sitting)[0]
print(f'reply speeches: {len(W):,}; after AfD: {W.prev_afd.sum():,} ({W.prev_afd.mean():.1%})')
print(f'raw: level after AfD {W[W.prev_afd==1].y.mean():+.3f} vs after others {W[W.prev_afd==0].y.mean():+.3f}; preceding-speech EMI: AfD {W[W.prev_afd==1].prev_emi.mean():+.3f} vs others {W[W.prev_afd==0].prev_emi.mean():+.3f}')
out = []
def fe(df, y, x, groups):
    yv, xv = df[y].astype(float).copy(), df[x].astype(float).copy()
    for _ in range(10):
        for gcol in groups: yv = yv - yv.groupby(df[gcol]).transform('mean'); xv = xv - xv.groupby(df[gcol]).transform('mean')
    return sm.OLS(yv.values, xv.values).fit(cov_type='cluster', cov_kwds={'groups': df.g.values})
# (a) regression adjustment
for lab, f in [('OLS: prev speech EMI + position + year + party', 'y ~ prev_afd + prev_emi + C(pos) + C(year) + C(party)'),
               ('OLS: + preceding-speech EMI x position', 'y ~ prev_afd + prev_emi * C(pos) + C(year) + C(party)')]:
    r = smf.ols(f, W).fit(cov_type='cluster', cov_kwds={'groups': W.g.values}); out.append(dict(estimator=lab, outcome='reply level', att=r.params['prev_afd'], se=r.bse['prev_afd'], p=r.pvalues['prev_afd'], n=int(r.nobs))); print(f"{lab:55s}: {r.params['prev_afd']:+.3f} (se {r.bse['prev_afd']:.3f}, p={r.pvalues['prev_afd']:.3f})")
Wr = W.copy(); Wr['prev_emi_res'] = Wr.prev_emi - Wr.groupby('pos').prev_emi.transform('mean')
r = fe(Wr.assign(x=Wr.prev_afd), 'y', 'x', ['pid', 'pos']); base = r
Wr['y_adj'] = Wr.y - smf.ols('y ~ prev_emi', Wr).fit().params['prev_emi'] * Wr.prev_emi
r = fe(Wr, 'y_adj', 'prev_afd', ['pid', 'pos']); out.append(dict(estimator='speaker + position FE, outcome adjusted for prev speech EMI', outcome='reply level', att=r.params[0], se=r.bse[0], p=r.pvalues[0], n=len(Wr)))
print(f"{'speaker + position FE, adjusted for prev speech EMI':55s}: {r.params[0]:+.3f} (se {r.bse[0]:.3f}, p={r.pvalues[0]:.3f})")
# (b) coarsened exact matching: party x position x year x quintile of preceding-speech EMI
W['q_prev'] = pd.qcut(W.prev_emi, 5, labels=False); W['stratum'] = W.party + '|' + W.pos.astype(str) + '|' + W.year.astype(str) + '|' + W.q_prev.astype(str)
cnt = W.groupby('stratum').prev_afd.agg(['sum', 'size']); ok = cnt[(cnt['sum'] > 0) & (cnt['sum'] < cnt['size'])].index; Wc = W[W.stratum.isin(ok)].copy()
nt = Wc.groupby('stratum').prev_afd.transform('sum'); nc = Wc.groupby('stratum').prev_afd.transform('size') - nt
Wc['w'] = np.where(Wc.prev_afd == 1, 1.0, nt / nc)
for ycol, lab in [('y', 'reply level'), ('diff', 'reply effect')]:
    Wd = Wc.dropna(subset=[ycol]); r = sm.WLS(Wd[ycol], sm.add_constant(Wd.prev_afd), weights=Wd.w).fit(cov_type='cluster', cov_kwds={'groups': Wd.g.values})
    out.append(dict(estimator='CEM: party x position x year x prev-EMI quintile', outcome=lab, att=r.params['prev_afd'], se=r.bse['prev_afd'], p=r.pvalues['prev_afd'], n=int(len(Wd)), n_treated=int(Wd.prev_afd.sum())))
    print(f"CEM ATT on {lab:12s} ({int(Wd.prev_afd.sum()):,} treated matched in {Wd.stratum.nunique():,} strata): {r.params['prev_afd']:+.3f} (se {r.bse['prev_afd']:.3f}, p={r.pvalues['prev_afd']:.3f})")
# (c) propensity score IPW-ATT
ps = smf.logit('prev_afd ~ prev_emi + C(pos) + C(year) + C(party) + loglen', W.dropna(subset=['loglen'])).fit(disp=0); Wp = W.dropna(subset=['loglen']).copy(); Wp['ps'] = ps.predict(Wp)
Wp = Wp[(Wp.ps > 0.02) & (Wp.ps < 0.98)]; Wp['w'] = np.where(Wp.prev_afd == 1, 1.0, Wp.ps / (1 - Wp.ps))
for ycol, lab in [('y', 'reply level'), ('diff', 'reply effect')]:
    Wd = Wp.dropna(subset=[ycol]); r = sm.WLS(Wd[ycol], sm.add_constant(Wd.prev_afd), weights=Wd.w).fit(cov_type='cluster', cov_kwds={'groups': Wd.g.values})
    out.append(dict(estimator='IPW-ATT on propensity score', outcome=lab, att=r.params['prev_afd'], se=r.bse['prev_afd'], p=r.pvalues['prev_afd'], n=int(len(Wd)), n_treated=int(Wd.prev_afd.sum())))
    print(f"IPW ATT on {lab:12s}: {r.params['prev_afd']:+.3f} (se {r.bse['prev_afd']:.3f}, p={r.pvalues['prev_afd']:.3f}); balance after weighting: prev EMI treated {np.average(Wd[Wd.prev_afd==1].prev_emi):+.3f} vs weighted controls {np.average(Wd[Wd.prev_afd==0].prev_emi, weights=Wd[Wd.prev_afd==0].w):+.3f}")
# naming the AfD: matched comparison within replies after AfD vs after others
r = smf.ols('names_afd ~ prev_afd + C(pos) + C(year) + C(party)', W).fit(cov_type='cluster', cov_kwds={'groups': W.g.values}); print(f"share of replies naming the AfD, after AfD vs after others (adjusted): {r.params['prev_afd']:+.3f} (se {r.bse['prev_afd']:.3f}); raw {W[W.prev_afd==1].names_afd.mean():.3f} vs {W[W.prev_afd==0].names_afd.mean():.3f}")
pd.DataFrame(out).to_csv('results/parliament/tables/contagion_matching.csv', index=False); print('saved results/parliament/tables/contagion_matching.csv')

# ---- topic register of the debate as the covariate (pre-treatment): mean EMI of the OTHER speeches in the same agenda item,
#      excluding the reply j and the preceding speech i, and the EMI of the speech before i (b). Avoids conditioning on the
#      AfD speech's own register, which is a trait of the speaker rather than of the debate.
rows2 = []
for sit, g in d.groupby('Sitting', sort=False):
    p, e, pos, pid, item, yr = g.party.values, g[EMI].values, g.pos.values, g.politicianId.values, g.item.values, g.year.values; n = len(g)
    for j in range(n):
        Y = p[j]
        if Y not in _SEQ_PARTIES or Y == 'AFD': continue
        i = next((k for k in range(j - 1, -1, -1) if p[k] != Y), None)
        if i is None or p[i] not in _SEQ_PARTIES: continue
        b = next((k for k in range(i - 1, -1, -1) if p[k] != p[i]), None)
        others = [e[k] for k in range(n) if item[k] == item[j] and k not in (i, j) and p[k] != 'AFD']
        rows2.append(dict(sitting=sit, pid=pid[j], party=Y, y=e[j], prev_afd=int(p[i] == 'AFD'), item_emi=np.mean(others) if len(others) >= 2 else np.nan, before_emi=e[b] if b is not None else np.nan, pos=pos[j], year=yr[j]))
V = pd.DataFrame(rows2).dropna(subset=['pid', 'item_emi']); V['g'] = pd.factorize(V.sitting)[0]
print(f"\nregister of the debate: item mean EMI (excl. i, j) treated {V[V.prev_afd==1].item_emi.mean():+.3f} vs controls {V[V.prev_afd==0].item_emi.mean():+.3f}")
for lab, f, data in [('OLS: debate register (item mean) + position + year + party', 'y ~ prev_afd + item_emi + C(pos) + C(year) + C(party)', V),
                     ('OLS: + EMI of the speech before the preceding one', 'y ~ prev_afd + item_emi + before_emi + C(pos) + C(year) + C(party)', V.dropna(subset=['before_emi']))]:
    r = smf.ols(f, data).fit(cov_type='cluster', cov_kwds={'groups': data.g.values}); out.append(dict(estimator=lab, outcome='reply level', att=r.params['prev_afd'], se=r.bse['prev_afd'], p=r.pvalues['prev_afd'], n=int(r.nobs))); print(f"{lab:60s}: {r.params['prev_afd']:+.3f} (se {r.bse['prev_afd']:.3f}, p={r.pvalues['prev_afd']:.3f})")
Vr = V.copy(); Vr['y_adj'] = Vr.y - smf.ols('y ~ item_emi', Vr).fit().params['item_emi'] * Vr.item_emi
r = fe(Vr, 'y_adj', 'prev_afd', ['pid', 'pos']); out.append(dict(estimator='speaker + position FE, adjusted for debate register', outcome='reply level', att=r.params[0], se=r.bse[0], p=r.pvalues[0], n=len(Vr))); print(f"{'speaker + position FE, adjusted for debate register':60s}: {r.params[0]:+.3f} (se {r.bse[0]:.3f}, p={r.pvalues[0]:.3f})")
V['q_item'] = pd.qcut(V.item_emi, 5, labels=False); V['stratum'] = V.party + '|' + V.pos.astype(str) + '|' + V.year.astype(str) + '|' + V.q_item.astype(str)
cnt = V.groupby('stratum').prev_afd.agg(['sum', 'size']); ok = cnt[(cnt['sum'] > 0) & (cnt['sum'] < cnt['size'])].index; Vc = V[V.stratum.isin(ok)].copy()
nt = Vc.groupby('stratum').prev_afd.transform('sum'); nc = Vc.groupby('stratum').prev_afd.transform('size') - nt; Vc['w'] = np.where(Vc.prev_afd == 1, 1.0, nt / nc)
r = sm.WLS(Vc.y, sm.add_constant(Vc.prev_afd), weights=Vc.w).fit(cov_type='cluster', cov_kwds={'groups': Vc.g.values}); out.append(dict(estimator='CEM: party x position x year x debate-register quintile', outcome='reply level', att=r.params['prev_afd'], se=r.bse['prev_afd'], p=r.pvalues['prev_afd'], n=int(len(Vc)), n_treated=int(Vc.prev_afd.sum())))
print(f"CEM on debate register ({int(Vc.prev_afd.sum()):,} treated): {r.params['prev_afd']:+.3f} (se {r.bse['prev_afd']:.3f}, p={r.pvalues['prev_afd']:.3f})")
pd.DataFrame(out).to_csv('results/parliament/tables/contagion_matching.csv', index=False); print('saved')
