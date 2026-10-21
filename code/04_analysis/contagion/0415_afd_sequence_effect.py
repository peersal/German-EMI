"""Does EMI of other speakers fall after an AfD contribution within the same sitting / agenda item?

Data: official plenary protocols of the 19th-21st Bundestag (data/revision/bundestag_speeches_2000_2025_raw.csv has
Sitting, Agenda, Party and the speech order via the numeric Source_ID) joined to the paper's EMI scores by id.
Unit: non-AfD, non-procedural speech. Within an agenda item, speeches are ordered; for each speech we know whether an
AfD speaker has already spoken in that item, how many AfD speeches preceded it, and whether the immediately preceding
speech was by the AfD. Models (two-way demeaned OLS, speaker and item fixed effects, errors clustered by sitting):
  EMI = a_speaker + g_item + b1 * after_first_afd + e
  EMI = a_speaker + g_item + b1 * after_first_afd + b2 * directly_after_afd + e
Also: items with vs without any AfD speech (item FE replaced by sitting FE), and the same by party of the speaker.
Output: results/revision/12_afd_sequence/
"""
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))   # German-EMI repository root: data in data/, outputs in results/
import os, numpy as np, pandas as pd, statsmodels.api as sm
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = REPO_ROOT; OUT = os.path.join(REPO_ROOT, 'results', 'revision', '12_afd_sequence'); os.makedirs(OUT, exist_ok=True)
EMI = 'emi_w2vparliament'
raw = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'revision', 'bundestag_speeches_2000_2025_raw.csv'), usecols=['Source_ID', 'Date', 'Term', 'Sitting', 'Agenda', 'Party', 'Speaker_ID', 'Role', 'Procedural'], dtype=str)
raw = raw[(raw.Date >= '2017-10-24') & (raw.Role == 'Regular')].copy()
sc = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'parliament_nhb.csv'), usecols=['id', EMI, 'party', 'politicianId'], dtype={'id': str})
d = raw.merge(sc, left_on='Source_ID', right_on='id', how='inner')          # speeches with EMI (the paper's corpus)
d['order'] = pd.to_numeric(d.Source_ID.str.extract(r'ID(\d+)')[0], errors='coerce')
d['item'] = d.Sitting + '|' + d.Agenda.fillna('(no agenda)')
d = d.sort_values(['Sitting', 'order']).reset_index(drop=True)
d['is_afd'] = (d.Party == 'AfD').astype(int)
# position relative to AfD within the item
d['afd_before'] = d.groupby('item').is_afd.cumsum() - d.is_afd              # AfD speeches earlier in the item
d['after_first_afd'] = (d.afd_before > 0).astype(int)
d['prev_afd'] = d.groupby('Sitting').is_afd.shift(1).fillna(0).astype(int)   # immediately preceding speech in the sitting is AfD
d['item_has_afd'] = d.groupby('item').is_afd.transform('max')
d['pos_in_item'] = d.groupby('item').cumcount(); d['item_len'] = d.groupby('item').is_afd.transform('size')
d['rel_pos'] = d.pos_in_item / d.item_len.clip(lower=1)
non = d[d.is_afd == 0].copy()
print(f'speeches with EMI since 24 Oct 2017: {len(d):,} | non-AfD: {len(non):,} | sittings: {d.Sitting.nunique()} | agenda items: {d.item.nunique()}')
sit = d.groupby('Sitting').is_afd.max(); itm = d.groupby('item').is_afd.max()
print(f'sittings with an AfD speech: {sit.mean():.1%} | agenda items with an AfD speech: {itm.mean():.1%} | non-AfD speeches after the first AfD speech in their item: {non.after_first_afd.mean():.1%} | directly after an AfD speech: {non.prev_afd.mean():.1%}')
print('\nraw means of EMI (non-AfD speeches):')
print(non.groupby(['item_has_afd', 'after_first_afd'])[EMI].agg(['mean', 'size']).round(3).to_string())
print(non.groupby('prev_afd')[EMI].agg(['mean', 'size']).round(3).to_string())

def twoway(df, cols, f1, f2, it=40):
    w = df[cols].copy().astype(float)
    for _ in range(it):
        for c in cols: w[c] = w[c] - w[c].groupby(df[f1]).transform('mean'); w[c] = w[c] - w[c].groupby(df[f2]).transform('mean')
    return w
def fe_ols(df, xs, f1, f2, tag):
    w = twoway(df, [EMI] + xs, f1, f2)
    r = sm.OLS(w[EMI], w[xs]).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(df.Sitting)[0]})
    out = {x: (r.params[x], r.bse[x], r.pvalues[x]) for x in xs}
    print(f'{tag}: ' + ' | '.join(f'{x} = {v[0]:+.4f} (se {v[1]:.4f}, p={v[2]:.3f})' for x, v in out.items()) + f'  n={int(r.nobs):,}')
    return out
rows = []
non['politicianId'] = non.politicianId.astype(str)
# 1. within item: before vs after the first AfD speech (items with AfD only; item FE absorb item-level EMI)
a = non[non.item_has_afd == 1]
r1 = fe_ols(a, ['after_first_afd'], 'politicianId', 'item', 'within-item, after first AfD speech')
r2 = fe_ols(a, ['after_first_afd', 'prev_afd'], 'politicianId', 'item', 'within-item, + directly after AfD')
r3 = fe_ols(a, ['after_first_afd', 'prev_afd', 'rel_pos'], 'politicianId', 'item', 'within-item, + position in item')
r4 = fe_ols(a, ['afd_before'], 'politicianId', 'item', 'within-item, number of AfD speeches before')
# 2. items with vs without AfD (sitting FE + speaker FE)
r5 = fe_ols(non, ['item_has_afd'], 'politicianId', 'Sitting', 'within-sitting, item has an AfD speech')
# 3. by party of the responding speaker
print('\nby party of the speaker (within-item, after first AfD speech):')
for pty, g in a.groupby('party'):
    if len(g) < 500: continue
    o = fe_ols(g, ['after_first_afd', 'prev_afd'], 'politicianId', 'item', f'   {pty:24s}')
    rows.append(dict(party=pty, n=len(g), after_coef=o['after_first_afd'][0], after_p=o['after_first_afd'][2], prev_coef=o['prev_afd'][0], prev_p=o['prev_afd'][2]))
# 4. by term
print('\nby electoral term:')
for t, g in a.groupby('Term'):
    o = fe_ols(g, ['after_first_afd', 'prev_afd'], 'politicianId', 'item', f'   WP{t}')
pd.DataFrame(rows).to_csv(os.path.join(OUT, 'by_party.csv'), index=False)
pd.DataFrame([dict(model=k, **{f'{x}_{s}': v[i] for x, v in o.items() for i, s in enumerate(['coef', 'se', 'p'])}) for k, o in
              [('after_first_afd', r1), ('+prev_afd', r2), ('+rel_pos', r3), ('n_afd_before', r4), ('item_has_afd', r5)]]).to_csv(os.path.join(OUT, 'models.csv'), index=False)
non[['Source_ID', 'Date', 'Term', 'Sitting', 'item', 'party', 'politicianId', EMI, 'after_first_afd', 'afd_before', 'prev_afd', 'item_has_afd', 'rel_pos']].to_csv(os.path.join(OUT, 'speeches_sequence.csv'), index=False)
# 5. placebo: same construction with the FDP (re-entered in 2017) and the Greens instead of the AfD
for pl in ['FDP', 'BÜNDNIS 90/DIE GRÜNEN', 'DIE LINKE']:
    dd = d.copy(); dd['is_x'] = (dd.Party == pl).astype(int); dd['x_before'] = dd.groupby('item').is_x.cumsum() - dd.is_x; dd['after_first_x'] = (dd.x_before > 0).astype(int)
    dd['prev_x'] = dd.groupby('Sitting').is_x.shift(1).fillna(0).astype(int); dd['item_has_x'] = dd.groupby('item').is_x.transform('max')
    nn = dd[(dd.is_x == 0) & (dd.is_afd == 0) & (dd.item_has_x == 1)].copy(); nn['politicianId'] = nn.politicianId.astype(str)
    fe_ols(nn, ['after_first_x', 'prev_x'], 'politicianId', 'item', f'placebo {pl:24s}')
