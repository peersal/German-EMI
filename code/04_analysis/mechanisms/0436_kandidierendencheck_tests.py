"""Issue-space distance to the AfD from the abgeordnetenwatch Kandidierenden-Check, and the contagion tests with it.

For each election check (2017: period 111 / check 12; 2021: 132 / 58; 2025: 161 / 72):
  1. answers -> matrix MdB x thesis coded +1 agree / 0 neutral / -1 disagree
  2. AfD modal answer per thesis (among AfD MdBs who answered); issue distance = share of theses on which the MdB
     answers unlike the AfD modal answer (neutral counts half); also a first principal component of the answer matrix
  3. match MdBs to politicianId through the Bundestag master data by name (same normalisation as the Breunig match)
  4. validation: 2017 issue distance vs Breunig & Guinaudeau expert distance, pooled and within party
  5. tests: within-MdB EMI change across the election on issue distance (party FE, East/West, mandate, gender),
     plus the quarterly event study, as in sections 9/10
Output: results/revision/11_kandidierendencheck/
"""
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))   # German-EMI repository root: data in data/, outputs in results/
import os, re, sys, json, unicodedata, numpy as np, pandas as pd, xml.etree.ElementTree as ET
import statsmodels.formula.api as smf
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = REPO_ROOT
D = os.path.join(REPO_ROOT, 'data', 'revision', 'abgeordnetenwatch'); OUT = os.path.join(REPO_ROOT, 'results', 'revision', '11_kandidierendencheck'); os.makedirs(OUT, exist_ok=True)
EMI = 'emi_w2vparliament'
CHECKS = {2017: (111, 12, '19'), 2021: (132, 58, '20'), 2025: (161, 72, '21')}
WINDOWS = {2017: (('2016Q1', '2017Q2'), ('2018Q1', '2019Q2')), 2021: (('2020Q1', '2021Q2'), ('2022Q1', '2023Q2')), 2025: (('2023Q3', '2024Q4'), ('2025Q3', '2026Q4'))}
CODE = {'Ich stimme zu': 1.0, 'Ich sehe es neutral': 0.0, 'Ich stimme dagegen': -1.0}
PARTY_MAP = {'CDU/CSU': 'CDU/CSU', 'SPD': 'SPD', 'FDP': 'FDP', 'DIE GRÜNEN': 'Greens', 'BÜNDNIS 90/DIE GRÜNEN': 'Greens', 'Bündnis 90/Die Grünen': 'Greens',
             'DIE LINKE': 'Left', 'Die Linke': 'Left', 'AfD': 'AfD', 'BSW': 'BSW'}
SUMMARY = []

def norm(s):
    s = str(s).replace('ß', 'ss'); s = unicodedata.normalize('NFKD', s); s = ''.join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r'\b(Dr|Prof|h\.c|med|jur|rer|nat|phil|Dipl|Ing|habil)\.?\s*', '', s); return re.sub(r'[^a-z ]', '', s.lower()).strip()

def stamm_names(wp):
    t = ET.parse(os.path.join(REPO_ROOT, 'data', 'revision', 'wahlkreise', 'MDB_STAMMDATEN.XML')).getroot(); rows = []
    for m in t.findall('MDB'):
        if wp not in {w.findtext('WP') for w in m.find('WAHLPERIODEN').findall('WAHLPERIODE')}: continue
        for n in m.find('NAMEN').findall('NAME'):
            v, nn = n.findtext('VORNAME') or '', n.findtext('NACHNAME') or ''
            rows.append((m.findtext('ID'), norm(f'{v} {nn}'))); rows.append((m.findtext('ID'), norm(f'{v.split()[0] if v else ""} {nn}')))
    return pd.DataFrame(rows, columns=['politicianId', 'key']).drop_duplicates()

def ols_cl(formula, d, cluster):
    m = smf.ols(formula, data=d); g = pd.factorize(d[cluster].astype(str).loc[m.data.row_labels])[0]
    return m.fit(cov_type='cluster', cov_kwds={'groups': g})
def add(test, outcome, predictor, r, term, n=None, note=''):
    ci = r.conf_int(); SUMMARY.append(dict(test=test, outcome=outcome, predictor=predictor, coef=float(r.params[term]), se=float(r.bse[term]), p=float(r.pvalues[term]),
                                          lo=float(ci.loc[term, 0]), hi=float(ci.loc[term, 1]), n=int(n or r.nobs), note=note))
    print(f'{test:14s} {outcome:30s} {predictor:26s} coef={r.params[term]:+.4f} se={r.bse[term]:.4f} p={r.pvalues[term]:.3f} n={int(n or r.nobs)}', flush=True)

par = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'parliament_nhb.csv'), usecols=['date', 'politicianId', 'party', EMI]).dropna(subset=['politicianId', EMI])
par['politicianId'] = par.politicianId.astype(int).astype(str); par['date'] = pd.to_datetime(par.date, utc=True); par['q'] = par.date.dt.tz_localize(None).dt.to_period('Q')
pos = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'revision', 'breunig_guinaudeau', 'mp_positions_wp19.csv'), dtype={'politicianId': str})
exp = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'revision', 'mdb_exposure_wp19.csv'), dtype={'politicianId': str})

for year, (period, check, wp) in CHECKS.items():
    f = os.path.join(D, f'answers_{period}_{check}.csv')
    if not os.path.exists(f): print(f'\n== {year}: no answers file yet ({f})'); continue
    a = pd.read_csv(f); a['code'] = a.rating.map(CODE)
    a['party_s'] = a.fraction.fillna(a.party).map(lambda s: next((v for k, v in PARTY_MAP.items() if isinstance(s, str) and k.lower() in s.lower()), s))
    M = a.pivot_table(index='aw_id', columns='n', values='code', aggfunc='first')
    meta = a.drop_duplicates('aw_id').set_index('aw_id')[['name', 'party_s', 'constituency']]
    print(f'\n== {year}: {len(M)} MdBs answered, {M.shape[1]} theses; by party', meta.party_s.value_counts().to_dict())
    afd = M.loc[meta.index[meta.party_s == 'AfD']]
    if len(afd) < 5: print('   too few AfD respondents for a modal answer'); continue
    modal = afd.mode().iloc[0]                                           # AfD modal answer per thesis
    agree_share = afd.eq(modal).mean()                                   # AfD cohesion per thesis
    dist = (M.sub(modal, axis=1).abs() / 2.0)                            # 0 same, 0.5 neutral vs stance, 1 opposite
    meta['issue_dist'] = dist.mean(axis=1)                               # share of theses answered unlike the AfD
    meta['n_answered'] = M.notna().sum(axis=1)
    # principal component of the answer matrix as a second scale
    Z = M.fillna(0.0).values; Z = Z - Z.mean(0); u, s, vt = np.linalg.svd(Z, full_matrices=False); pc = Z @ vt[0]
    if np.corrcoef(pc, meta.issue_dist.values)[0, 1] < 0: pc = -pc
    meta['pc1'] = pc
    pd.DataFrame({'thesis': a.drop_duplicates('n').set_index('n').thesis, 'afd_modal': modal, 'afd_cohesion': agree_share}).to_csv(os.path.join(OUT, f'theses_{year}.csv'))
    # match to politicianId
    meta['key'] = meta.name.map(norm); st = stamm_names(wp)
    mm = meta.reset_index().merge(st, on='key', how='left').drop_duplicates('aw_id')
    last = st.assign(last=st.key.str.split().str[-1]).drop_duplicates(['politicianId', 'last']); cnt = last.groupby('last').politicianId.nunique(); uniq = last[last['last'].map(cnt) == 1].set_index('last').politicianId
    fx = mm.politicianId.isna(); mm.loc[fx, 'politicianId'] = mm.loc[fx, 'key'].str.split().str[-1].map(uniq)
    print(f'   matched to politicianId: {mm.politicianId.notna().sum()} of {len(mm)}')
    mm = mm.dropna(subset=['politicianId']).drop_duplicates('politicianId')
    mm.to_csv(os.path.join(OUT, f'issue_distance_{year}.csv'), index=False)
    non = mm[mm.party_s != 'AfD'].copy()
    non['issue_wz'] = non.groupby('party_s').issue_dist.transform(lambda s: (s - s.mean()) / s.std()); non['pc1_wz'] = non.groupby('party_s').pc1.transform(lambda s: (s - s.mean()) / s.std())
    print('   issue distance by party:', non.groupby('party_s').issue_dist.mean().round(3).to_dict())
    # validation against the expert positions (19th Bundestag)
    v = non.merge(pos[['politicianId', 'dist_afd', 'position']], on='politicianId')
    if len(v) > 30:
        print(f'   validation vs Breunig ({len(v)}): pooled r(issue_dist, dist_afd) = {v.issue_dist.corr(v.dist_afd):.3f}; within party:',
              v.groupby('party_s').apply(lambda g: g.issue_dist.corr(g.dist_afd) if len(g) > 10 else np.nan).round(2).to_dict())
    # tests
    (p0, p1), (q0, q1) = WINDOWS[year]
    pre = par[(par.q >= pd.Period(p0)) & (par.q <= pd.Period(p1))].groupby('politicianId')[EMI].agg(['mean', 'size'])
    post = par[(par.q >= pd.Period(q0)) & (par.q <= pd.Period(q1))].groupby('politicianId')[EMI].agg(['mean', 'size'])
    ch = pre.join(post, lsuffix='_pre', rsuffix='_post', how='inner'); ch = ch[(ch.size_pre >= 5) & (ch.size_post >= 5)]; ch['change'] = ch.mean_post - ch.mean_pre
    d = ch.reset_index().merge(non, on='politicianId').merge(exp[['politicianId', 'east', 'direct', 'land']], on='politicianId', how='left')
    d['land'] = d.land.fillna('NA'); d['east'] = d.east.fillna(0); d['direct'] = d.direct.fillna(0)
    if len(d) < 40: print(f'   change panel too small ({len(d)})'); continue
    print(f'   change panel {len(d)}:', d.groupby('party_s').size().to_dict())
    for pred in ['issue_wz', 'pc1_wz']:
        r = ols_cl(f'change ~ {pred} + C(party_s) + east + direct', d, 'land'); add(f'KC{year}', f'change across {year}', pred, r, pred)
    r = ols_cl('change ~ issue_wz * C(party_s) + east + direct', d, 'land')
    for t in r.params.index:
        if t.startswith('issue_wz:'): add(f'KC{year}', f'change across {year}', 'issue_wz x ' + t.split('T.')[1].rstrip(']'), r, t, note='vs reference party')
    # event study
    ev = par[par.politicianId.isin(d.politicianId) & (par.q >= pd.Period(p0) - 4) & (par.q <= pd.Period(q1) + 2)]
    ev = ev.groupby(['politicianId', 'q'])[EMI].agg(['mean', 'size']).reset_index().rename(columns={'mean': 'y'}); ev = ev[ev['size'] >= 2]
    ev = ev.merge(d[['politicianId', 'party_s', 'issue_wz']], on='politicianId'); ev['qs'] = ev.q.astype(str); ev['post'] = (ev.q > pd.Period(p1) + 1).astype(int); ev['x_post'] = ev.issue_wz * ev.post
    r = ols_cl('y ~ C(politicianId) + C(party_s):C(qs) + x_post', ev, 'politicianId'); add(f'KC{year}', f'event study {year}, post x issue', 'issue_wz', r, 'x_post', n=ev.politicianId.nunique())
    d.to_csv(os.path.join(OUT, f'panel_{year}.csv'), index=False)

summ = pd.DataFrame(SUMMARY)
if len(summ):
    summ['sig05'] = summ.p < 0.05; summ.to_csv(os.path.join(OUT, 'tests_summary.csv'), index=False)
    pd.set_option('display.width', 220); print('\n==== SUMMARY ===='); print(summ.round(4).to_string(index=False))
json.dump(SUMMARY, open(os.path.join(OUT, 'numbers.json'), 'w'), indent=1, default=float)
