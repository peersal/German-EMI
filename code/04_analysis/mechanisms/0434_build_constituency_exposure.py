"""Constituency exposure to the AfD for every MdB of the 19th Bundestag.

Sources: Bundeswahlleiter btw17_kerg.csv (final 2017 results per Wahlkreis with the 2013 results as 'Vorperiode',
re-expressed on 2017 boundaries) and the Bundestag master data (WKR_NUMMER, WKR_LAND, MANDATSART, LISTE per term).
Output: data/revision/mdb_exposure_wp19.csv with, per politicianId: AfD second-vote share 2017 and 2013 in the MdB's
own constituency (list MdBs: the constituency they ran in), its change, the MdB's own party's share, the Land-level AfD
share, mandate type and East/West.
"""
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))   # German-EMI repository root: data in data/, outputs in results/
import os, csv, io, numpy as np, pandas as pd, xml.etree.ElementTree as ET
HERE = os.path.dirname(os.path.abspath(__file__))
KERG = os.path.join(REPO_ROOT, 'data', 'revision', 'wahlkreise', 'btw17_kerg.csv')
STAMM = os.path.join(REPO_ROOT, 'data', 'revision', 'wahlkreise', 'MDB_STAMMDATEN.XML')
if not os.path.exists(STAMM):
    raise FileNotFoundError(f'{STAMM} missing: download MDB_STAMMDATEN.XML (Stammdaten aller Abgeordneten, bundestag.de -> Abgeordnete -> Biografien -> Open Data) into data/revision/wahlkreise/')

raw = open(KERG, encoding='utf-8-sig').read().splitlines()
lines = [l for l in raw if not l.startswith('#')]
hdr1, hdr2, hdr3 = lines[0].split(';'), lines[1].split(';'), lines[2].split(';')
# forward-fill the party/block names (row 1) and vote type (row 2); row 3 = Endgültig / Vorperiode
blocks, cur = [], None
for i, h in enumerate(hdr1):
    if h: cur = h
    blocks.append(cur)
vtype, cur = [], None
for i, h in enumerate(hdr2):
    if h: cur = h
    vtype.append(cur)
cols = []
for i in range(len(hdr1)):
    if i < 3: cols.append(hdr1[i]); continue
    cols.append(f'{blocks[i]}|{vtype[i]}|{hdr3[i] if i < len(hdr3) else ""}')
rows = [l.split(';') for l in lines[3:] if l.strip()]
df = pd.DataFrame(rows, columns=cols[:len(rows[0])] if len(rows[0]) <= len(cols) else cols + [''] * (len(rows[0]) - len(cols)))
df = df.rename(columns={'Nr': 'nr', 'Gebiet': 'gebiet', 'gehört zu': 'gehoert_zu'})
num = lambda s: pd.to_numeric(s.replace('', np.nan), errors='coerce')
def share(party, period):
    return num(df[f'{party}|Zweitstimmen|{period}']) / num(df[f'Gültige|Zweitstimmen|{period}'])
P = {'AfD': 'Alternative für Deutschland', 'CDU': 'Christlich Demokratische Union Deutschlands', 'CSU': 'Christlich-Soziale Union in Bayern e.V.',
     'SPD': 'Sozialdemokratische Partei Deutschlands', 'LINKE': 'DIE LINKE', 'GRUENE': 'BÜNDNIS 90/DIE GRÜNEN', 'FDP': 'Freie Demokratische Partei'}
out = pd.DataFrame({'nr': num(df.nr).astype('Int64'), 'gebiet': df.gebiet, 'gehoert_zu': num(df.gehoert_zu).astype('Int64')})
for k, v in P.items():
    out[f'{k}_2017'] = share(v, 'Endgültig'); out[f'{k}_2013'] = share(v, 'Vorperiode')
wk = out[(out.gehoert_zu.notna()) & (out.gehoert_zu != 99)].copy()           # 299 constituencies
laender = out[out.gehoert_zu == 99].copy()                                     # 16 Länder (nr = Land number)
LAND_CODE = {1: 'SH', 2: 'HH', 3: 'NI', 4: 'HB', 5: 'NW', 6: 'HE', 7: 'RP', 8: 'BW', 9: 'BY', 10: 'SL', 11: 'BE', 12: 'BB', 13: 'MV', 14: 'SN', 15: 'ST', 16: 'TH'}
laender['land'] = laender.nr.map(LAND_CODE)
print('constituencies', len(wk), '| AfD 2017 mean share %.3f, 2013 %.3f' % (wk.AfD_2017.mean(), wk.AfD_2013.mean()))
wk.to_csv(os.path.join(REPO_ROOT, 'data', 'revision', 'wahlkreise', 'wahlkreise_2017_shares.csv'), index=False)

# --- MdBs of the 19th term from the master data ---------------------------------------------------
t = ET.parse(STAMM).getroot(); rows = []
for m in t.findall('MDB'):
    mid = m.findtext('ID'); bio = m.find('BIOGRAFISCHE_ANGABEN')
    for wp in m.find('WAHLPERIODEN').findall('WAHLPERIODE'):
        if wp.findtext('WP') != '19': continue
        fr = [i.findtext('INS_LANG') for i in wp.find('INSTITUTIONEN').findall('INSTITUTION') if i.findtext('INSART_LANG') == 'Fraktion/Gruppe']
        rows.append(dict(politicianId=mid, wkr=pd.to_numeric(wp.findtext('WKR_NUMMER'), errors='coerce'), wkr_land=wp.findtext('WKR_LAND'),
                         mandatsart=wp.findtext('MANDATSART'), liste=wp.findtext('LISTE'), partei=bio.findtext('PARTEI_KURZ'), fraktion=fr[0] if fr else None))
mdb = pd.DataFrame(rows).drop_duplicates('politicianId')
mdb['land'] = mdb.wkr_land.fillna(mdb.liste)
EAST = {'BB', 'MV', 'SN', 'ST', 'TH', 'BE'}
mdb['east'] = mdb.land.isin(EAST).astype(int); mdb['direct'] = (mdb.mandatsart == 'Direktwahl').astype(int)
own = {'CDU': 'CDU', 'CSU': 'CSU', 'SPD': 'SPD', 'DIE LINKE.': 'LINKE', 'BÜNDNIS 90/DIE GRÜNEN': 'GRUENE', 'FDP': 'FDP', 'AfD': 'AfD'}
mdb = mdb.merge(wk[['nr'] + [c for c in wk.columns if c.endswith('_2017') or c.endswith('_2013')]], left_on='wkr', right_on='nr', how='left')
mdb = mdb.merge(laender[['land', 'AfD_2017', 'AfD_2013']].rename(columns={'AfD_2017': 'AfD_land_2017', 'AfD_2013': 'AfD_land_2013'}), on='land', how='left')
mdb['own_party_key'] = mdb.partei.map(own)
mdb['own_share_2017'] = [r[f'{k}_2017'] if isinstance(k, str) else np.nan for k, r in zip(mdb.own_party_key, mdb.to_dict('records'))]
mdb['afd_share_2017'] = mdb.AfD_2017; mdb['afd_share_2013'] = mdb.AfD_2013; mdb['afd_change'] = mdb.AfD_2017 - mdb.AfD_2013
mdb['afd_minus_own'] = mdb.afd_share_2017 - mdb.own_share_2017
keep = ['politicianId', 'partei', 'fraktion', 'land', 'east', 'direct', 'wkr', 'afd_share_2017', 'afd_share_2013', 'afd_change', 'own_share_2017', 'afd_minus_own', 'AfD_land_2017', 'AfD_land_2013']
mdb[keep].to_csv(os.path.join(REPO_ROOT, 'data', 'revision', 'mdb_exposure_wp19.csv'), index=False)
print('MdBs WP19', len(mdb), '| with constituency share', mdb.afd_share_2017.notna().sum(), '| direct', mdb.direct.sum(), '| east', mdb.east.sum())
print(mdb.groupby('partei')[['afd_share_2017', 'afd_change']].mean().round(3).to_string())
