"""Directional within-speaker reply levels: does party Y speak differently right after party X?

For every regular speech j (party Y) since the AfD's entry, X = party of the nearest preceding speech by another
party in the same sitting. Outcome = EMI_j minus the speaker's own mean EMI (speaker fixed effects), additionally
residualised on the position within the agenda item (position fixed effects). Cells = mean within-speaker deviation
of Y's speeches that follow X; s.e. clustered by sitting. The AfD row answers "does the AfD become more intuitive
after being addressed by X?", the AfD column is the known reply reaction.
Output: results/parliament/tables/reply_direction_matrix.csv (+ _n, _se), printed matrices.
"""
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))   # German-EMI repository root: data in data/, outputs in results/
import os, numpy as np, pandas as pd, statsmodels.api as sm
os.chdir(REPO_ROOT)
src = open('code/04_analysis/contagion/0411_contagion_sequence.py', encoding='utf-8').read().split('def mdb_mention_network')[0].replace('from matplotlib.patches', 'import matplotlib; matplotlib.use("Agg")\nfrom matplotlib.patches')
party_colors = {}; exec(src)
EMI = 'emi_w2vparliament'
raw = pd.read_csv('data/revision/bundestag_speeches_2000_2025_raw.csv', usecols=['Source_ID', 'Date', 'Sitting', 'Agenda', 'Party', 'Role'], dtype=str)
raw = raw[(raw.Role == 'Regular') & (raw.Date >= '2017-10-24')]
sc = pd.read_csv('data/parliament_nhb.csv', usecols=['id', EMI, 'politicianId'], dtype={'id': str})
d = raw.merge(sc, left_on='Source_ID', right_on='id').dropna(subset=[EMI, 'politicianId'])
d['party'] = d.Party.map(_SEQ_RAW_TO_NHB); d['order'] = pd.to_numeric(d.Source_ID.str.extract(r'ID(\d+)')[0]); d = d.sort_values(['Sitting', 'order']).reset_index(drop=True)
d['item'] = d.Sitting + '|' + d.Agenda.fillna(''); d['pos'] = d.groupby('item').cumcount().clip(upper=6)
rows = []
for sit, g in d.groupby('Sitting', sort=False):
    p, e, pos, pid = g.party.values, g[EMI].values, g.pos.values, g.politicianId.values
    for j in range(len(g)):
        Y = p[j]
        if Y not in _SEQ_PARTIES: continue
        i = next((k for k in range(j - 1, -1, -1) if p[k] != Y), None)
        if i is None or p[i] not in _SEQ_PARTIES: continue
        rows.append(dict(sitting=sit, pid=pid[j], Y=Y, X=p[i], pos=pos[j], y=e[j]))
W = pd.DataFrame(rows); W = W[W.groupby('pid').pid.transform('size') >= 5].copy()
W['dev'] = W.y - W.groupby('pid').y.transform('mean')                                  # speaker FE
W['dev_pos'] = W.dev - W.groupby(['Y', 'pos']).dev.transform('mean')                   # + position FE within replying party
print(f'{len(W):,} replies by {W.pid.nunique():,} speakers (>= 5 replies) since 24 Oct 2017')


def cell(sub, col):
    if len(sub) < 30: return np.nan, np.nan
    r = sm.OLS(sub[col].values, np.ones(len(sub))).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(sub.sitting)[0]})
    return float(r.params[0]), float(r.bse[0])


order = ['AFD', 'CDU/CSU', 'SPD', 'FDP', 'Alliance 90/The Greens', 'The Left Party']; short = {k: v for k, v in _SEQ_SHORT.items()}
for col, lab in [('dev', 'speaker FE'), ('dev_pos', 'speaker + position FE')]:
    M = pd.DataFrame(index=order, columns=order, dtype=float); S = M.copy(); N = M.copy()
    for X in order:
        for Y in order:
            if X == Y: continue
            sub = W[(W.X == X) & (W.Y == Y)]; m, s = cell(sub, col); M.loc[X, Y] = m; S.loc[X, Y] = s; N.loc[X, Y] = len(sub)
    M.index = [short.get(i, i) for i in M.index]; M.columns = [short.get(c, c) for c in M.columns]; S.index, S.columns, N.index, N.columns = M.index, M.columns, M.index, M.columns
    print(f'\n{lab}: mean within-speaker EMI deviation of the REPLYING party (columns) after the PRECEDING party (rows)')
    print(M.round(3).to_string()); print('\ns.e.'); print(S.round(3).to_string())
    if col == 'dev_pos':
        M.to_csv('results/parliament/tables/reply_direction_matrix.csv'); S.to_csv('results/parliament/tables/reply_direction_matrix_se.csv'); N.to_csv('results/parliament/tables/reply_direction_matrix_n.csv')
# AfD as replying party: after any non-AfD party vs after ... there is no AfD-after-AfD; compare after coalition vs opposition
afd = W[W.Y == 'AFD'].copy(); coal = afd.X.isin(['CDU/CSU', 'SPD'])
m1, s1 = cell(afd[coal], 'dev_pos'); m2, s2 = cell(afd[~coal], 'dev_pos')
print(f'\nAfD speeches after a governing party (CDU/CSU, SPD): {m1:+.3f} (se {s1:.3f}, n={int(coal.sum()):,}); after an opposition party: {m2:+.3f} (se {s2:.3f}, n={int((~coal).sum()):,}); difference {m1 - m2:+.3f} (se {np.hypot(s1, s2):.3f})')
# symmetric summary: for each party as replier, after AfD vs after others (within speaker + position)
print('\nreplying party: within-speaker deviation after AfD minus after all other parties')
for Y in order[1:]:
    a = W[(W.Y == Y) & (W.X == 'AFD')]; o = W[(W.Y == Y) & (W.X != 'AFD')]; ma, sa = cell(a, 'dev_pos'); mo, so = cell(o, 'dev_pos')
    print(f'   {short.get(Y, Y):10s} after AfD {ma:+.3f} (se {sa:.3f}, n={len(a):,}) | after others {mo:+.3f} | diff {ma - mo:+.3f} (se {np.hypot(sa, so):.3f})')
