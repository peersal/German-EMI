"""Resolve the Open Discourse speech ids of the validation chunks to party, date and term by streaming
data/full_parliament/speeches_table_Germany.json (Source_ID = 'OD<id>').
Output: data/revision/validation_chunk_party.csv
"""
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))   # German-EMI repository root: data in data/, outputs in results/
import json, os, csv, time, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = REPO_ROOT
SRC = os.path.join(REPO_ROOT, 'data', 'full_parliament', 'speeches_table_Germany.json')
ids = set('OD' + s for s in pd.read_csv(os.path.join(REPO_ROOT, 'data', 'validation', 'validation_instances_labels.csv'), dtype={'id': str}).id)
print('looking up', len(ids), 'ids')
rows = []; t0 = time.time(); n = 0
with open(SRC, encoding='utf-8') as f:
    for line in f:
        n += 1
        s = line.strip().rstrip(',')
        if not s.startswith('{') or '"OD' not in s:
            continue
        r = json.loads(s)
        if r.get('Source_ID') in ids:
            rows.append(dict(id=r['Source_ID'][2:], od_date=r.get('Date'), od_term=r.get('Term'), party=r.get('Party'),
                             party_orientation=r.get('Party_Orientation'), role=r.get('Role'), speaker=r.get('Speaker_ID')))
            if len(rows) == len(ids):
                break
print('found', len(rows), 'of', len(ids), 'in', round(time.time() - t0), 's')
pd.DataFrame(rows).to_csv(os.path.join(REPO_ROOT, 'data', 'revision', 'validation_chunk_party.csv'), index=False)
