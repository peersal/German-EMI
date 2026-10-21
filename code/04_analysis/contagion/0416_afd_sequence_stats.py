"""All statistics of the within-sitting before/after analysis in one place, with SE, 95% CI and p.
Also builds the speaker-transition network (party -> party, consecutive speeches in a sitting) with the mean
within-speaker EMI deviation of the following speech, and picks an illustrative sitting for the figure.
Output: results/revision/12_afd_sequence/sequence_stats.json, transitions.csv, example_sitting.csv
"""
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))   # German-EMI repository root: data in data/, outputs in results/
import os, json, numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = REPO_ROOT; OUT = os.path.join(REPO_ROOT, 'results', 'revision', '12_afd_sequence')
EMI = 'emi_w2vparliament'
raw = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'revision', 'bundestag_speeches_2000_2025_raw.csv'), usecols=['Source_ID', 'Date', 'Sitting', 'Agenda', 'Party', 'Role'], dtype=str)
raw = raw[(raw.Date >= '2017-10-24') & (raw.Role == 'Regular')]
sc = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'parliament_nhb.csv'), usecols=['id', EMI, 'party', 'politicianId', 'text_clean', 'firstName', 'lastName'], dtype={'id': str})
d = raw.merge(sc, left_on='Source_ID', right_on='id'); d['order'] = pd.to_numeric(d.Source_ID.str.extract(r'ID(\d+)')[0], errors='coerce')
d['item'] = d.Sitting + '|' + d.Agenda.fillna('(no agenda)'); d = d.sort_values(['Sitting', 'order']).reset_index(drop=True)
d['Party'] = d.Party.fillna('none'); d['politicianId'] = d.politicianId.astype(str)
d['mentions_afd'] = d.text_clean.fillna('').str.lower().str.contains(r'\bafd\b|alternative für deutschland').astype(int)
d['spk_mean'] = d.groupby('politicianId')[EMI].transform('mean'); d['dev'] = d[EMI] - d.spk_mean          # within-speaker deviation
d['pos_item'] = d.groupby('item').cumcount().clip(upper=8)
d['low_q'] = d.groupby('Party')[EMI].transform(lambda s: (s <= s.quantile(0.25)).astype(float)).fillna(0).astype(int)
S = {}
def z2p(z): return float(2 * (1 - stats.norm.cdf(abs(z))))
def cl_mean(x, g):
    """mean with SE clustered by g"""
    x = np.asarray(x, float); g = np.asarray(g); m = x.mean(); df_ = pd.DataFrame({'x': x - m, 'g': g}); u = df_.groupby('g').x.sum(); se = float(np.sqrt((u ** 2).sum()) / len(x))
    return dict(est=float(m), se=se, lo=float(m - 1.96 * se), hi=float(m + 1.96 * se), p=z2p(m / se), n=int(len(x)), n_clusters=int(len(u)))
def cl_ols(y, X, groups):
    r = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(groups)[0]}); return r
def term(r, t): ci = r.conf_int(); return dict(est=float(r.params[t]), se=float(r.bse[t]), lo=float(ci.loc[t, 0]), hi=float(ci.loc[t, 1]), p=float(r.pvalues[t]), n=int(r.nobs))

# --- descriptives ----------------------------------------------------------------------------------
S['descriptives'] = dict(speeches=int(len(d)), non_afd=int((d.Party != 'AfD').sum()), afd=int((d.Party == 'AfD').sum()), sittings=int(d.Sitting.nunique()), items=int(d.item.nunique()),
                         sittings_with_afd=float(d.groupby('Sitting').Party.apply(lambda s: (s == 'AfD').any()).mean()), items_with_afd=float(d.groupby('item').Party.apply(lambda s: (s == 'AfD').any()).mean()))

# --- local windows: every speech, k=1 and k=2 ------------------------------------------------------
rows = []
for sit, g in d.groupby('Sitting', sort=False):
    party = g.Party.values; emi = g[EMI].values; ment = g.mentions_afd.values; pos = g.pos_item.values; item = g.item.values; lowq = g.low_q.values; ids = g.Source_ID.values; n = len(g)
    for i in range(n):
        X = party[i]; before = [j for j in range(i - 1, -1, -1) if party[j] != X][:5]; after = [j for j in range(i + 1, n) if party[j] != X][:5]
        rec = dict(id=ids[i], sitting=sit, item=item[i], party=X, pos=pos[i], low_q=lowq[i], emi=emi[i])
        for k in (1, 2, 3, 5):
            rec[f'diff{k}'] = (emi[after[:k]].mean() - emi[before[:k]].mean()) if len(before) >= k and len(after) >= k else np.nan
        rec['after1_mentions_afd'] = ment[after[0]] if after else np.nan; rec['after1_emi'] = emi[after[0]] if after else np.nan; rec['before1_emi'] = emi[before[0]] if before else np.nan
        rows.append(rec)
L = pd.DataFrame(rows); L['is_afd'] = (L.party == 'AfD').astype(int)
PARTIES = ['AfD', 'FDP', 'SPD', 'CDU/CSU', 'BÜNDNIS 90/DIE GRÜNEN', 'DIE LINKE']
S['local_window'] = {}
for k in (1, 2, 3, 5):
    S['local_window'][f'k{k}'] = {}
    for X in PARTIES:
        g = L[(L.party == X)].dropna(subset=[f'diff{k}']); S['local_window'][f'k{k}'][X] = cl_mean(g[f'diff{k}'], g.sitting)
    afd = S['local_window'][f'k{k}']['AfD']; pl = [S['local_window'][f'k{k}'][X] for X in PARTIES[1:]]
    pm = np.mean([p['est'] for p in pl]); pse = np.sqrt(sum(p['se'] ** 2 for p in pl)) / len(pl); dse = np.sqrt(afd['se'] ** 2 + pse ** 2)
    S['local_window'][f'k{k}']['AfD_minus_placebo'] = dict(est=float(afd['est'] - pm), se=float(dse), lo=float(afd['est'] - pm - 1.96 * dse), hi=float(afd['est'] - pm + 1.96 * dse), p=z2p((afd['est'] - pm) / dse), placebo_mean=float(pm))
# position-matched with item FE
def item_fe(sub, extra_cols, ycol='diff1'):
    sub = sub.dropna(subset=[ycol]).copy(); X = pd.get_dummies(sub.pos.astype(int), prefix='pos', drop_first=True).astype(float)
    for c in extra_cols: X[c] = sub[c].astype(float)
    y = sub[ycol] - sub.groupby('item')[ycol].transform('mean'); X = X - X.groupby(sub.item).transform('mean')
    return cl_ols(y, X, sub.sitting)
r = item_fe(L, ['is_afd']); S['position_matched_item_fe_k1'] = term(r, 'is_afd')
r = item_fe(L, ['is_afd'], 'diff2'); S['position_matched_item_fe_k2'] = term(r, 'is_afd')
L['afd_x_low'] = L.is_afd * L.low_q
r = item_fe(L, ['is_afd', 'low_q', 'afd_x_low']); S['low_emi_interaction_item_fe'] = {t: term(r, t) for t in ['is_afd', 'low_q', 'afd_x_low']}
S['low_emi_speech_followed_by'] = {X: cl_mean(L[(L.party == X) & (L.low_q == 1)].dropna(subset=['diff1']).diff1, L[(L.party == X) & (L.low_q == 1)].dropna(subset=['diff1']).sitting) for X in PARTIES}
# dose
A = L[L.is_afd == 1].dropna(subset=['diff1']).copy(); A['qt'] = pd.qcut(A.emi, 4, labels=['Q1 most intuition-based', 'Q2', 'Q3', 'Q4 most evidence-based'])
S['dose_quartiles'] = {str(q): dict(**cl_mean(g.diff1, g.sitting), share_reply_mentions_afd=float(g.after1_mentions_afd.mean())) for q, g in A.groupby('qt', observed=True)}
r = cl_ols(A.diff1, sm.add_constant(A[['emi']]), A.sitting); S['dose_slope_per_emi_unit'] = term(r, 'emi')
# mentions
S['reply_mentions'] = {('mentions' if m else 'no_mention'): dict(**cl_mean(g.diff1, g.sitting), reply_emi_mean=float(g.after1_emi.mean())) for m, g in A.groupby('after1_mentions_afd')}
S['share_replies_mentioning_afd'] = float(A.after1_mentions_afd.mean())
nn = d[d.Party != 'AfD'].copy(); y = nn[EMI] - nn.groupby('politicianId')[EMI].transform('mean'); x = nn.mentions_afd - nn.groupby('politicianId').mentions_afd.transform('mean')
S['mention_afd_effect_speaker_fe'] = dict(**term(cl_ols(y, pd.DataFrame({'m': x}), nn.Sitting), 'm'), share=float(nn.mentions_afd.mean()))
# profile by offset
def profile(X, K=6):
    acc = {o: [] for o in range(-K, K + 1) if o != 0}
    for sit, g in d.groupby('Sitting', sort=False):
        party = g.Party.values; emi = g[EMI].values; isx = party == X; nonx = np.where(~isx)[0]
        for i in np.where(isx)[0]:
            p = np.searchsorted(nonx, i)
            for o in acc:
                jj = p + o - 1 if o > 0 else p + o
                if 0 <= jj < len(nonx): acc[o].append(emi[nonx[jj]])
    return {o: dict(mean=float(np.mean(v)), se=float(np.std(v, ddof=1) / np.sqrt(len(v))), n=len(v)) for o, v in acc.items()}
S['profile'] = {X: profile(X) for X in ['AfD', 'FDP', 'SPD', 'CDU/CSU']}
# --- transition network: previous party -> this party, within sitting, consecutive speeches ------------------
d['prev_party'] = d.groupby('Sitting').Party.shift(1); d['prev_emi'] = d.groupby('Sitting')[EMI].shift(1)
T = d.dropna(subset=['prev_party']); T = T[(T.prev_party != T.Party) & T.Party.isin(PARTIES) & T.prev_party.isin(PARTIES)]
net = T.groupby(['prev_party', 'Party']).agg(n=('dev', 'size'), mean_dev=('dev', 'mean'), se_dev=('dev', lambda s: s.std(ddof=1) / np.sqrt(len(s))), mean_emi=(EMI, 'mean')).reset_index()
net.to_csv(os.path.join(OUT, 'transitions.csv'), index=False)
S['network_from_afd_mean_dev'] = {r.Party: dict(est=float(r.mean_dev), se=float(r.se_dev), n=int(r.n)) for r in net[net.prev_party == 'AfD'].itertuples()}
S['network_from_others_mean_dev'] = {r.Party: dict(est=float(r.mean_dev), n=int(r.n)) for r in net[net.prev_party != 'AfD'].groupby('Party').apply(lambda g: pd.Series(dict(mean_dev=np.average(g.mean_dev, weights=g.n), n=g.n.sum()))).reset_index().itertuples()}
node = d[d.Party.isin(PARTIES)].groupby('Party').agg(n=(EMI, 'size'), mean_emi=(EMI, 'mean')).reset_index(); node.to_csv(os.path.join(OUT, 'nodes.csv'), index=False)
# --- illustrative sitting: >= 40 speeches, >= 5 AfD speeches, mean local diff of AfD speeches closest to the overall AfD estimate
cand = L[L.is_afd == 1].dropna(subset=['diff1']).groupby('sitting').agg(m=('diff1', 'mean'), k=('diff1', 'size'))
sizes = d.groupby('Sitting').size(); cand = cand[(cand.k >= 5) & (sizes.reindex(cand.index) >= 40)]
target = S['local_window']['k1']['AfD']['est']; best = (cand.m - target).abs().idxmin(); S['example_sitting'] = dict(sitting=best, mean_afd_local_diff=float(cand.loc[best, 'm']), n_afd=int(cand.loc[best, 'k']), n_speeches=int(sizes[best]))
ex = d[d.Sitting == best][['Source_ID', 'Date', 'order', 'Party', 'firstName', 'lastName', EMI, 'mentions_afd', 'Agenda']].copy(); ex['seq'] = range(1, len(ex) + 1); ex.to_csv(os.path.join(OUT, 'example_sitting.csv'), index=False)
json.dump(S, open(os.path.join(OUT, 'sequence_stats.json'), 'w'), indent=1, default=float)
L.to_csv(os.path.join(OUT, 'local_window_all_speeches_k.csv'), index=False)
# --- print the headline numbers ----------------------------------------------------------------------
f = lambda e: f"{e['est']:+.3f} (SE {e['se']:.3f}, 95% CI {e['lo']:+.3f} to {e['hi']:+.3f}, p {'< 0.001' if e['p'] < 0.001 else f'= {e['p']:.3f}'})"
print('descriptives', S['descriptives'])
for k in (1, 2, 3, 5): print(f'k={k}: AfD {f(S["local_window"][f"k{k}"]["AfD"])} | AfD minus placebo {f(S["local_window"][f"k{k}"]["AfD_minus_placebo"])}')
for X in PARTIES[1:]: print(f'   placebo k=1 {X}: {f(S["local_window"]["k1"][X])}')
print('position-matched item FE k1:', f(S['position_matched_item_fe_k1']), '| k2:', f(S['position_matched_item_fe_k2']))
print('low-EMI interaction:', {t: f(v) for t, v in S['low_emi_interaction_item_fe'].items()})
print('dose quartiles:', {q: f(v) for q, v in S['dose_quartiles'].items()}); print('dose slope:', f(S['dose_slope_per_emi_unit']))
print('reply mentions:', {m: f(v) for m, v in S['reply_mentions'].items()}, '| share', round(S['share_replies_mentioning_afd'], 3))
print('mention effect speaker FE:', f(S['mention_afd_effect_speaker_fe']))
print('network from AfD (within-speaker deviation of the following speech):', {k: round(v['est'], 3) for k, v in S['network_from_afd_mean_dev'].items()})
print('network from others:', {k: round(v['est'], 3) for k, v in S['network_from_others_mean_dev'].items()})
print('example sitting:', S['example_sitting'])
