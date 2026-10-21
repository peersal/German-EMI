"""Battery of mechanism tests: does ideological distance to the AfD (Breunig & Guinaudeau 2025) or constituency AfD
exposure predict the extent, speed or form of within-MdB rhetorical change? Writes results/revision/10_mechanism_tests/.

T3  2021 transition on the stayer panel (positions carried forward)
T4  speed: discrete-time hazard of 'adopting' after 2017
T5  long-run drift 2015-2025: distance x time with MdB and party x quarter FE
T7  similarity of an MdB's speech embedding to the AfD centroid, change across 2017
T8  contact with the AfD in debates (same agenda item; speaking right after an AfD speaker) and contact x distance
T9  intra-party wings (Sälzer codes in the Breunig data)
T10 placebo: the 2013 transition (no AfD in the chamber) with the same positions
T12 lead-lag: AfD's lagged quarterly EMI x distance
(T6 gap-to-AfD is algebraically T5 with quarter FE; T11 needs the attribution coding; T13 needs a Twitter-MdB map.)
"""
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))   # German-EMI repository root: data in data/, outputs in results/
import os, sys, json, warnings, numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = REPO_ROOT
OUT = os.path.join(REPO_ROOT, 'results', 'revision', '10_mechanism_tests'); os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(REPO_ROOT, 'code', '03_scoring'))
EMI = 'emi_w2vparliament'
SUMMARY = []
def add(test, outcome, predictor, r, term, n=None, note=''):
    ci = r.conf_int()
    SUMMARY.append(dict(test=test, outcome=outcome, predictor=predictor, coef=float(r.params[term]), se=float(r.bse[term]),
                        p=float(r.pvalues[term]), lo=float(ci.loc[term, 0]), hi=float(ci.loc[term, 1]), n=int(n if n is not None else r.nobs), note=note))
    print(f'{test:4s} {outcome:28s} {predictor:34s} coef={r.params[term]:+.4f} se={r.bse[term]:.4f} p={r.pvalues[term]:.3f} n={int(n if n is not None else r.nobs)}', flush=True)
def ols_cl(formula, d, cluster):
    m = smf.ols(formula, data=d); g = pd.factorize(d[cluster].astype(str).loc[m.data.row_labels])[0]
    return m.fit(cov_type='cluster', cov_kwds={'groups': g})

# ---------------------------------------------------------------- data
par = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'parliament_nhb.csv'), usecols=['id', 'date', 'politicianId', 'party', 'leaning', 'text_clean', EMI], dtype={'id': str})
par = par.dropna(subset=['politicianId', EMI]); par['politicianId'] = par.politicianId.astype(int).astype(str)
par['date'] = pd.to_datetime(par.date, utc=True); par['q'] = par.date.dt.tz_localize(None).dt.to_period('Q')
pos = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'revision', 'breunig_guinaudeau', 'mp_positions_wp19.csv'), dtype={'politicianId': str})
exp = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'revision', 'mdb_exposure_wp19.csv'), dtype={'politicianId': str})
cov = pos[['politicianId', 'party', 'position', 'dist_afd', 'faction', 'gender']].merge(exp[['politicianId', 'east', 'direct', 'land', 'afd_share_2017', 'afd_change']], on='politicianId', how='left')
cov = cov[cov.party != 'AfD'].copy()
cov['dist_wz'] = cov.groupby('party').dist_afd.transform(lambda s: (s - s.mean()) / s.std())
cov['afd_wz'] = cov.groupby('party').afd_share_2017.transform(lambda s: (s - s.mean()) / s.std())
cov['land'] = cov.land.fillna('NA')
def window_change(df, pre, post, y=EMI, min_n=5):
    a = df[(df.q >= pd.Period(pre[0])) & (df.q <= pd.Period(pre[1]))].groupby('politicianId')[y].agg(['mean', 'size'])
    b = df[(df.q >= pd.Period(post[0])) & (df.q <= pd.Period(post[1]))].groupby('politicianId')[y].agg(['mean', 'size'])
    j = a.join(b, lsuffix='_pre', rsuffix='_post', how='inner'); j = j[(j.size_pre >= min_n) & (j.size_post >= min_n)]
    return (j.mean_post - j.mean_pre).rename('change').reset_index()
BASE = ' + C(party) + east + direct + C(gender)'

# ---------------------------------------------------------------- T3: 2021 transition
ch21 = window_change(par, ('2020Q1', '2021Q2'), ('2022Q1', '2023Q2')).merge(cov, on='politicianId')
print('\nT3 panel (2021 transition):', len(ch21), ch21.groupby('party').size().to_dict())
print('mean change by party:', ch21.groupby('party').change.mean().round(3).to_dict())
for pred in ['dist_wz', 'afd_wz']:
    r = ols_cl(f'change ~ {pred}{BASE}', ch21.dropna(subset=[pred]), 'land'); add('T3', 'change across 2021', pred, r, pred)
r = ols_cl('change ~ dist_wz*C(party) + east + direct + C(gender)', ch21, 'land')
for t in r.params.index:
    if t.startswith('dist_wz:'): add('T3', 'change across 2021', 'dist_wz x ' + t.split('T.')[1].rstrip(']'), r, t, note='interaction vs CDU/CSU')
# event study 2019Q1-2023Q4
ev = par[par.politicianId.isin(ch21.politicianId) & (par.q >= pd.Period('2019Q1')) & (par.q <= pd.Period('2023Q4'))]
ev = ev.groupby(['politicianId', 'q'])[EMI].agg(['mean', 'size']).reset_index().rename(columns={'mean': 'y', 'size': 'n'})
ev = ev[ev.n >= 2].merge(cov[['politicianId', 'party', 'dist_wz', 'afd_wz']], on='politicianId'); ev['qs'] = ev.q.astype(str)
ev['post'] = (ev.q > pd.Period('2021Q3')).astype(int)
for pred in ['dist_wz', 'afd_wz']:
    d = ev.dropna(subset=[pred]).copy(); d['x_post'] = d[pred] * d.post
    r = ols_cl('y ~ C(politicianId) + C(party):C(qs) + x_post', d, 'politicianId'); add('T3', 'event study 2021, post x pred', pred, r, 'x_post', n=d.politicianId.nunique())
ch21.to_csv(os.path.join(OUT, 'T3_panel_2021.csv'), index=False)

# ---------------------------------------------------------------- T4: speed (discrete-time hazard after 2017)
pan17 = pd.read_csv(os.path.join(REPO_ROOT, 'results', 'revision', '09_mechanism_2017', 'panel_2017.csv'), dtype={'politicianId': str})
qm = par[par.politicianId.isin(pan17.politicianId)].groupby(['politicianId', 'q'])[EMI].agg(['mean', 'size']).reset_index().rename(columns={'mean': 'y'})
qm = qm[qm['size'] >= 2]
pre_stats = qm[(qm.q >= pd.Period('2015Q1')) & (qm.q <= pd.Period('2017Q2'))].groupby('politicianId').y.agg(['mean', 'std', 'size'])
pre_stats = pre_stats[pre_stats['size'] >= 3]; gsd = float(pre_stats['std'].median())
rows = []
for pid, g in qm[(qm.q >= pd.Period('2017Q4')) & (qm.q <= pd.Period('2019Q4'))].groupby('politicianId'):
    if pid not in pre_stats.index: continue
    thr = pre_stats.loc[pid, 'mean'] - 0.5 * (pre_stats.loc[pid, 'std'] if pre_stats.loc[pid, 'std'] > 0 else gsd)
    g = g.sort_values('q'); roll = g.y.rolling(2, min_periods=1).mean().values
    event_seen = False
    for k, (qq, rv) in enumerate(zip(g.q, roll)):
        ev_ = int(rv < thr)
        rows.append(dict(politicianId=pid, q=str(qq), t=k + 1, event=ev_))
        if ev_: event_seen = True; break
haz = pd.DataFrame(rows).merge(cov[['politicianId', 'party', 'dist_wz', 'afd_wz', 'east', 'direct', 'gender', 'land']], on='politicianId')
print('\nT4 hazard rows', len(haz), '| MdBs', haz.politicianId.nunique(), '| adopted by 2019Q4:', haz.groupby('politicianId').event.max().mean().round(3))
for pred in ['dist_wz', 'afd_wz']:
    d = haz.dropna(subset=[pred])
    m = smf.logit(f'event ~ {pred} + C(party) + C(t) + east + direct', data=d)
    r = m.fit(disp=0, cov_type='cluster', cov_kwds={'groups': pd.factorize(d.politicianId.loc[m.data.row_labels])[0]})
    add('T4', 'hazard of adoption 2017Q4-2019Q4', pred, r, pred, n=d.politicianId.nunique(), note='logit, log-odds per within-party SD')
med = haz.merge(cov[['politicianId']], on='politicianId').copy()
med['tercile'] = med.groupby('party').dist_wz.transform(lambda s: pd.qcut(s, 3, labels=['close', 'middle', 'far']))
first = med[med.event == 1].groupby('politicianId').t.min()
adopt = med.groupby(['politicianId', 'party', 'tercile'], observed=True).event.max().reset_index()
adopt['t_first'] = adopt.politicianId.map(first)
tab = adopt.groupby(['party', 'tercile'], observed=True).agg(n=('politicianId', 'size'), share_adopted=('event', 'mean'), median_quarters=('t_first', 'median')).round(2)
tab.to_csv(os.path.join(OUT, 'T4_adoption_by_distance_tercile.csv')); print(tab.to_string())

# ---------------------------------------------------------------- T5: long-run drift 2015-2025, distance x time
qa = par.groupby(['politicianId', 'q'])[EMI].agg(['mean', 'size']).reset_index().rename(columns={'mean': 'y', 'size': 'n'})
qa = qa[qa.n >= 2].merge(cov[['politicianId', 'party', 'dist_wz', 'afd_wz']], on='politicianId')
qa['qs'] = qa.q.astype(str); qa['t'] = (qa.q - pd.Period('2015Q1')).apply(lambda x: x.n) / 4.0     # years since 2015
for pred in ['dist_wz', 'afd_wz']:
    d = qa.dropna(subset=[pred]).copy(); d['x_t'] = d[pred] * d.t
    r = ols_cl('y ~ C(politicianId) + C(party):C(qs) + x_t', d, 'politicianId'); add('T5', 'drift 2015-2025 (per year)', pred + ' x time', r, 'x_t', n=d.politicianId.nunique())
    for lo, hi, lab in [('2015Q1', '2019Q4', '2015-2019'), ('2020Q1', '2025Q2', '2020-2025')]:
        dd = d[(d.q >= pd.Period(lo)) & (d.q <= pd.Period(hi))]
        r = ols_cl('y ~ C(politicianId) + C(party):C(qs) + x_t', dd, 'politicianId'); add('T5', f'drift {lab} (per year)', pred + ' x time', r, 'x_t', n=dd.politicianId.nunique())

# ---------------------------------------------------------------- T7: similarity to the AfD's language
import importlib; S = importlib.import_module('0304_score_emi')
kv = S.load_model(os.path.join(REPO_ROOT, 'data', 'parliament_model')); tok_stop = S.nltk_german_stopwords()
sub = par[(par.q >= pd.Period('2016Q1')) & (par.q <= pd.Period('2019Q2'))].dropna(subset=['text_clean']).copy()
print('\nT7 encoding', len(sub), 'speeches 2016Q1-2019Q2', flush=True)
V, _ = S.doc_vectors(sub.text_clean.tolist(), kv, tok_stop)
sub = sub.reset_index(drop=True)
ref_mask = ((sub.party == 'AFD') & (sub.q >= pd.Period('2017Q4')) & (sub.q <= pd.Period('2018Q4'))).values
ref = V[ref_mask].mean(0); print('AfD reference speeches', int(ref_mask.sum()))
def cos_rows(M, u): return (M @ u) / (np.linalg.norm(M, axis=1) * np.linalg.norm(u) + 1e-12)
sub['sim_afd'] = cos_rows(V, ref)
# topic-neutral variant: similarity to AfD minus similarity to the rest of the chamber in the same window
ref_other = V[((sub.party != 'AFD') & (sub.q >= pd.Period('2017Q4')) & (sub.q <= pd.Period('2018Q4'))).values].mean(0)
sub['sim_afd_rel'] = sub.sim_afd - cos_rows(V, ref_other)
for ycol, lab in [('sim_afd', 'similarity to AfD centroid'), ('sim_afd_rel', 'similarity to AfD minus to chamber')]:
    ch = window_change(sub, ('2016Q1', '2017Q2'), ('2018Q1', '2019Q2'), y=ycol).merge(cov, on='politicianId')
    print(f'{lab}: chamber mean change {ch.change.mean():+.4f} (n={len(ch)}); by party', ch.groupby('party').change.mean().round(4).to_dict())
    for pred in ['dist_wz', 'afd_wz']:
        r = ols_cl(f'change ~ {pred}{BASE}', ch.dropna(subset=[pred]), 'land'); add('T7', f'change in {lab}', pred, r, pred)
    ch.to_csv(os.path.join(OUT, f'T7_change_{ycol}.csv'), index=False)
# does similarity to the AfD co-move with EMI at the MdB level?
ch_e = window_change(sub, ('2016Q1', '2017Q2'), ('2018Q1', '2019Q2')).rename(columns={'change': 'd_emi'})
ch_s = window_change(sub, ('2016Q1', '2017Q2'), ('2018Q1', '2019Q2'), y='sim_afd_rel').rename(columns={'change': 'd_sim'})
cs = ch_e.merge(ch_s, on='politicianId').merge(cov, on='politicianId')
r = ols_cl('d_emi ~ d_sim + C(party)', cs, 'land'); add('T7', 'EMI change ~ similarity change', 'd_sim (within party)', r, 'd_sim', note='do MdBs whose language moved towards the AfD also drop more in EMI?')

# ---------------------------------------------------------------- T8: contact with the AfD in debates
raw = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'revision', 'bundestag_speeches_2000_2025_raw.csv'), usecols=['Source_ID', 'Date', 'Sitting', 'Agenda', 'Party', 'Speaker_ID', 'Role'], dtype=str)
raw = raw[raw.Date >= '2017-10-01'].copy(); raw['Date'] = pd.to_datetime(raw.Date, utc=True)
raw['order'] = pd.to_numeric(raw.Source_ID.str.extract(r'ID(\d+)')[0], errors='coerce'); raw = raw.sort_values(['Sitting', 'order'])
raw['prev_party'] = raw.groupby('Sitting').Party.shift(1)
raw['after_afd'] = (raw.prev_party == 'AfD').astype(int)
afd_items = set(zip(raw[raw.Party == 'AfD'].Sitting, raw[raw.Party == 'AfD'].Agenda.fillna('')))
raw['same_item_afd'] = [int((s, a) in afd_items) for s, a in zip(raw.Sitting, raw.Agenda.fillna(''))]
post = raw[(raw.Date >= '2017-10-24') & (raw.Date < '2019-07-01') & raw.Speaker_ID.notna()]
contact = post.groupby('Speaker_ID').agg(n_speeches=('Source_ID', 'size'), share_after_afd=('after_afd', 'mean'), share_same_item=('same_item_afd', 'mean')).reset_index().rename(columns={'Speaker_ID': 'politicianId'})
c8 = pan17[['politicianId', 'change']].merge(contact, on='politicianId').merge(cov, on='politicianId')
c8 = c8[c8.n_speeches >= 5]
for c in ['share_after_afd', 'share_same_item']:
    c8[c + '_wz'] = c8.groupby('party')[c].transform(lambda s: (s - s.mean()) / s.std())
print('\nT8 contact panel', len(c8), '| mean share after AfD %.3f, same item %.3f' % (c8.share_after_afd.mean(), c8.share_same_item.mean()))
for c in ['share_after_afd_wz', 'share_same_item_wz']:
    r = ols_cl(f'change ~ {c}{BASE}', c8, 'land'); add('T8', 'change across 2017', c, r, c)
    r = ols_cl(f'change ~ {c}*dist_wz{BASE}', c8, 'land'); add('T8', 'change across 2017', f'{c} x dist_wz', r, f'{c}:dist_wz')
c8.to_csv(os.path.join(OUT, 'T8_contact_panel.csv'), index=False)

# ---------------------------------------------------------------- T9: intra-party wings
w = pan17.merge(cov[['politicianId', 'faction']], on='politicianId', suffixes=('', '_cov'))
w['faction'] = w.faction.fillna('none')
WINGS = {'CDU/CSU': ['MIT'], 'SPD': ['SEEH', 'PL', 'NB'], 'Greens': ['REAL', 'FUNDI']}
for pty, codes in WINGS.items():
    d = w[w.party == pty].copy()
    for code in codes:
        d[f'w_{code}'] = (d.faction == code).astype(int)
        if d[f'w_{code}'].sum() >= 5:
            r = ols_cl(f'change ~ w_{code} + east + direct + C(gender)', d, 'land'); add('T9', f'change across 2017, {pty}', f'wing {code} (n={int(d[f"w_{code}"].sum())}) vs rest', r, f'w_{code}')

# ---------------------------------------------------------------- T10: placebo at the 2013 transition
h = pd.read_csv(os.path.join(REPO_ROOT, 'results', 'revision', '02_pre2015', 'bundestag_2000_2025_emi_harmonised.csv'), usecols=['politicianId', 'date', 'emi_full'], dtype={'politicianId': str})
h['date'] = pd.to_datetime(h.date, utc=True); h['q'] = h.date.dt.tz_localize(None).dt.to_period('Q')
h['politicianId'] = h.politicianId.str.replace(r'\.0$', '', regex=True)
ch13 = window_change(h, ('2012Q1', '2013Q2'), ('2014Q1', '2015Q2'), y='emi_full').merge(cov, on='politicianId')
print('\nT10 placebo panel 2013:', len(ch13), ch13.groupby('party').size().to_dict(), '| mean change', round(ch13.change.mean(), 3))
for pred in ['dist_wz', 'afd_wz']:
    r = ols_cl(f'change ~ {pred}{BASE}', ch13.dropna(subset=[pred]), 'land'); add('T10', 'change across 2013 (placebo)', pred, r, pred)
# and the 2017 change measured on the same re-scored series, for comparability
ch17h = window_change(h, ('2016Q1', '2017Q2'), ('2018Q1', '2019Q2'), y='emi_full').merge(cov, on='politicianId')
r = ols_cl('change ~ dist_wz' + BASE, ch17h, 'land'); add('T10', 'change across 2017 (re-scored series)', 'dist_wz', r, 'dist_wz')
ch13.to_csv(os.path.join(OUT, 'T10_placebo_2013.csv'), index=False)

# ---------------------------------------------------------------- T12: lead-lag with the AfD's quarterly EMI
afd_q = par[par.party == 'AFD'].groupby('q')[EMI].mean().rename('afd_emi')
ll = qa[(qa.q >= pd.Period('2017Q4')) & (qa.q <= pd.Period('2021Q2'))].copy()
ll['y_lag'] = ll.groupby('politicianId').y.shift(1); ll['q_lag'] = ll.q - 1
ll['afd_lag'] = ll.q_lag.map(afd_q); ll = ll.dropna(subset=['y_lag', 'afd_lag'])
for pred in ['dist_wz', 'afd_wz']:
    d = ll.dropna(subset=[pred]).copy(); d['afd_x'] = (d.afd_lag - d.afd_lag.mean()) * d[pred]
    r = ols_cl('y ~ y_lag + C(politicianId) + C(party):C(qs) + afd_x', d, 'politicianId'); add('T12', 'EMI_t ~ AfD EMI_{t-1} x pred', pred, r, 'afd_x', n=d.politicianId.nunique())

# ---------------------------------------------------------------- summary
summ = pd.DataFrame(SUMMARY); summ['sig05'] = summ.p < 0.05
summ.to_csv(os.path.join(OUT, 'tests_summary.csv'), index=False)
json.dump(SUMMARY, open(os.path.join(OUT, 'numbers.json'), 'w'), indent=1, default=float)
pd.set_option('display.width', 220)
print('\n==== SUMMARY (p < 0.05 flagged) ====')
print(summ[['test', 'outcome', 'predictor', 'coef', 'se', 'p', 'n', 'sig05']].round(4).to_string(index=False))
