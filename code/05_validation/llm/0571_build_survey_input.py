"""Build the LLM input for the 2026 validation surveys: the 300 Bundestag chunks and the 300 tweets rated on Prolific,
with the human ratings aggregated per item. Writes inputs/survey_input.csv (id, text) for the cluster job and
inputs/survey_meta.csv (id, corpus, party, leaning, date, emi, human means, n raters) for analyze_survey.py.

  python code/05_validation/llm/0571_build_survey_input.py
"""
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))   # German-EMI repository root: data in data/, outputs in results/
import os, pandas as pd, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); SURVEY = os.path.join(os.path.dirname(HERE), 'survey'); INP = os.path.join(REPO_ROOT, 'data', 'llm', 'inputs'); os.makedirs(INP, exist_ok=True)
LIK = {'Stimme überhaupt nicht zu': -2, 'Stimme nicht zu': -1, 'Neutral': 0, 'Stimme zu': 1, 'Stimme voll und ganz zu': 2}
EV, IU = 'Rating.Evidenz basiert', 'Rating.Intuition basiert'
SETS = {'parliament': (os.path.join(SURVEY, 'parliament_results', 'pilot'), 'text'),
        'twitter': (os.path.join(SURVEY, 'twitter_results', 'twitter', 'pilot'), 'text_shown')}
rows, meta = [], []
for corpus, (P, textcol) in SETS.items():
    key = pd.read_csv(os.path.join(SURVEY, corpus, 'data_files', 'survey_sample_key.csv'), dtype={'id': str})
    if textcol not in key.columns: textcol = 'text'
    a = pd.read_csv(os.path.join(P, 'exports', 'csv', 'annotations.csv'), encoding='utf-8'); a = a[~a.instance_id.str.startswith('gold')].copy()
    a['ev'] = a[EV].map(LIK); a['iu'] = a[IU].map(LIK)
    h = a.groupby('instance_id').agg(ev_mean=('ev', 'mean'), iu_mean=('iu', 'mean'), n_raters=('ev', 'size')).reset_index()
    k = key.merge(h, left_on='id', right_on='instance_id', how='left')
    for _, r in k.iterrows():
        rows.append(dict(id=r['id'], text=str(r[textcol]).replace('\r', ' ').replace('\n', ' ').strip()))
        meta.append(dict(id=r['id'], corpus=corpus, party=r.get('party'), leaning=r.get('leaning'), date=r.get('date'), emi=r.get('emi'), speech_emi=r.get('speech_emi', np.nan),
                         n_words=r.get('n_words'), ev_mean=r.get('ev_mean'), iu_mean=r.get('iu_mean'), n_raters=r.get('n_raters')))
    print(corpus, len(k), 'items;', int(k.ev_mean.notna().sum()), 'with human ratings; words median', k[textcol].astype(str).str.split().str.len().median())
pd.DataFrame(rows).to_csv(os.path.join(INP, 'survey_input.csv'), index=False)
pd.DataFrame(meta).to_csv(os.path.join(INP, 'survey_meta.csv'), index=False)
print('written', os.path.join(INP, 'survey_input.csv'), len(rows), 'rows')
