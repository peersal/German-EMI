"""Stream data/full_parliament/speeches_table_Germany.json (14.7 GB, one JSON record per line)
and keep Bundestag plenary speeches from START onwards: Role == Regular, not procedural.
Writes data/revision/bundestag_speeches_2000_2025_raw.csv (text untouched) plus a build report.
Run: .venv/Scripts/python.exe code/01_data/parliament/0114_extract_bundestag_2000.py
"""
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))   # German-EMI repository root: data in data/, outputs in results/
import json, csv, os, sys, time, collections

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(REPO_ROOT, 'data', 'full_parliament', 'speeches_table_Germany.json')
OUT = os.path.join(REPO_ROOT, 'data', 'revision', 'bundestag_speeches_2000_2025_raw.csv')
START = '2000-01-01'
KEEP = ['ID', 'Source_ID', 'Date', 'Term', 'Sitting', 'Agenda', 'Speaker_ID', 'Role', 'Procedural', 'MP', 'Minister',
        'Party', 'Party_Orientation', 'Party_Status', 'Source', 'Source_File', 'Text']

os.makedirs(os.path.dirname(OUT), exist_ok=True)
stats = collections.Counter(); by_term = collections.Counter(); by_source = collections.Counter()
t0 = time.time(); n = 0
with open(SRC, 'r', encoding='utf-8') as f, open(OUT, 'w', encoding='utf-8', newline='') as g:
    w = csv.DictWriter(g, fieldnames=KEEP); w.writeheader()
    for line in f:
        n += 1
        s = line.strip()
        if s in ('[', ']', ''):
            continue
        if s.endswith(','):
            s = s[:-1]
        try:
            r = json.loads(s)
        except json.JSONDecodeError:
            stats['json_error'] += 1; continue
        if r.get('Body') != 'Bundestag':
            continue
        stats['bundestag'] += 1
        d = r.get('Date') or ''
        if d < START:
            continue
        stats['since_start'] += 1
        if r.get('Role') != 'Regular' or r.get('Procedural'):
            stats['dropped_role_or_procedural'] += 1; continue
        if not r.get('Text'):
            stats['dropped_empty'] += 1; continue
        stats['kept'] += 1; by_term[r.get('Term')] += 1; by_source[r.get('Source')] += 1
        w.writerow({k: r.get(k) for k in KEEP})
        if n % 500000 == 0:
            print(f'{n:,} lines, kept {stats["kept"]:,}, {time.time()-t0:.0f}s', flush=True)
report = dict(lines=n, seconds=round(time.time() - t0), stats=dict(stats), by_term=dict(by_term), by_source=dict(by_source))
json.dump(report, open(os.path.join(REPO_ROOT, 'data', 'revision', 'extract_report.json'), 'w'), indent=1)
print(json.dumps(report, indent=1))
