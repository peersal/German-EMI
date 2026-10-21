"""Build the scoring input for the pre-2015 extension.

2015-2025: EXACTLY the paper's corpus rows (data/parliament_nhb.csv: id, metadata, text_clean), untouched.
2000-2014: Open Discourse speeches from data/revision/bundestag_speeches_2000_2025_raw.csv, cleaned with the
           paper's preprocessing (score_emi.preprocess), faction speeches only, > 10 cleaned tokens.
Output: data/revision/bundestag_2000_2025_input.csv  (one row per speech, column text_clean for all rows)
"""
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))   # German-EMI repository root: data in data/, outputs in results/
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = REPO_ROOT
sys.path.insert(0, os.path.join(REPO_ROOT, 'code', '03_scoring')); import importlib; S = importlib.import_module('0304_score_emi')

PARTY_MAP = {'CDU/CSU': 'CDU/CSU', 'SPD': 'SPD', 'FDP': 'FDP', 'BÜNDNIS 90/DIE GRÜNEN': 'Alliance 90/The Greens',
             'DIE LINKE': 'The Left Party', 'PDS': 'The Left Party', 'AfD': 'AFD', 'BSW': 'BSW'}
LEANING = {'CDU/CSU': 'middle', 'SPD': 'middle', 'FDP': 'middle', 'Alliance 90/The Greens': 'middle',
           'The Left Party': 'left', 'BSW': 'left', 'AFD': 'right'}

old = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'parliament_nhb.csv'),
                  usecols=['id', 'date', 'electoralTerm', 'politicianId', 'party', 'leaning', 'text_clean', 'length',
                           'evidence_score_w2vparliament', 'intuition_score_w2vparliament', 'emi_w2vparliament'],
                  dtype={'id': str})
old = old.rename(columns={'electoralTerm': 'term', 'evidence_score_w2vparliament': 'evidence_score_paper',
                          'intuition_score_w2vparliament': 'intuition_score_paper', 'emi_w2vparliament': 'emi_paper'})
old['source'] = 'paper_corpus'; old['politicianId'] = old['politicianId'].astype('Int64').astype(str)
old['date'] = pd.to_datetime(old['date'], utc=True)
print('paper corpus rows', len(old))

new = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'revision', 'bundestag_speeches_2000_2025_raw.csv'),
                  usecols=['Source_ID', 'Date', 'Term', 'Speaker_ID', 'Party', 'Source', 'Text'], dtype=str)
new['Date'] = pd.to_datetime(new['Date'], utc=True)
new = new[(new.Date >= '2000-01-01') & (new.Date < '2015-01-01') & new.Party.isin(PARTY_MAP)].copy()
print('open discourse 2000-2014 faction speeches', len(new), new.Source.value_counts().to_dict())
clean, lens = S.preprocess(new.Text.tolist(), S.nltk_german_stopwords() | S.CUSTOM_STOP)
new['text_clean'] = clean; new['length'] = lens
new = new[new.length > 10]
new = pd.DataFrame({'id': new.Source_ID, 'date': new.Date, 'term': new.Term.astype(int),
                    'politicianId': new.Speaker_ID.fillna('unknown_' + new.Source_ID),
                    'party': new.Party.map(PARTY_MAP), 'text_clean': new.text_clean, 'length': new.length,
                    'source': 'open_discourse'})
new['leaning'] = new.party.map(LEANING)
print('after > 10 tokens', len(new))

out = pd.concat([new, old], ignore_index=True).sort_values('date')
out.to_csv(os.path.join(REPO_ROOT, 'data', 'revision', 'bundestag_2000_2025_input.csv'), index=False)
print('written', len(out), 'rows;', out.groupby('source').size().to_dict())
print(out.groupby(out.date.dt.year).size().to_dict())
