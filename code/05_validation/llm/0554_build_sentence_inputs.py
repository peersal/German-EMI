"""Sentence-level inputs for the four-category coding (own intuitive justification / criticism of others' reasoning /
credibility or lying accusation / neutral).

 1. sentences_input.csv  : EVERY sentence of every speech 2015-2025 (id = <speech id>_s<n>), speaker header stripped,
                           sentences with >= 4 words. Coded by one model (Llama) so each speech gets its share of
                           other-directed sentences and can be re-scored without them.
 2. sentsample_input.csv : 10,000 of those sentences, stratified by year x leaning (equal cells), coded by both models.
Run from code/:  .venv/Scripts/python.exe code/05_validation/llm/0554_build_sentence_inputs.py
"""
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))   # German-EMI repository root: data in data/, outputs in results/
import os, re, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = REPO_ROOT; INP = os.path.join(REPO_ROOT, 'data', 'llm', 'inputs')
SEED = 20261015; N_SAMPLE = 10_000; MIN_WORDS = 4
cols = ['id', 'date', 'politicianId', 'lastName', 'party', 'leaning', 'text']
sc = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'parliament_nhb.csv'), usecols=cols, dtype={'id': str}).dropna(subset=['text'])
sc['date'] = pd.to_datetime(sc.date, utc=True).dt.tz_localize(None)
def strip_header(text, last):
    m = re.match(r'^([^:]{0,220}?):\s', text)
    if m and isinstance(last, str) and last.split()[-1] in m.group(1): return text[m.end():]
    return text
SENT = re.compile(r'(?<=[.!?])\s+(?=[A-ZÄÖÜ„"(])')
rows = []
for sid, text, last in zip(sc.id, sc.text, sc.lastName):
    for n, s in enumerate(SENT.split(strip_header(text, last).strip())):
        s = s.strip()
        if len(s.split()) >= MIN_WORDS: rows.append((f'{sid}_s{n}', sid, n, s))
S = pd.DataFrame(rows, columns=['id', 'speech_id', 'sent_idx', 'text'])
S = S.merge(sc[['id', 'date', 'politicianId', 'party', 'leaning']].rename(columns={'id': 'speech_id'}), on='speech_id')
S['year'] = S.date.dt.year
print(f'sentences: {len(S):,} from {S.speech_id.nunique():,} speeches; words median {S.text.str.split().str.len().median():.0f}; per speech median {S.groupby("speech_id").size().median():.0f}')
S[['id', 'text']].to_csv(os.path.join(INP, 'sentences_input.csv'), index=False)
S.drop(columns=['text']).to_csv(os.path.join(INP, 'sentences_meta.csv'), index=False)
# stratified sample: equal cells year x leaning
cells = S.groupby(['year', 'leaning']).size(); k = N_SAMPLE // len(cells)
samp = pd.concat([g.sample(min(k, len(g)), random_state=SEED) for _, g in S.groupby(['year', 'leaning'])])
rest = S.drop(samp.index); samp = pd.concat([samp, rest.sample(N_SAMPLE - len(samp), random_state=SEED)]) if len(samp) < N_SAMPLE else samp
print(f'sample: {len(samp):,}; cells {len(cells)} x {k}; by leaning {samp.leaning.value_counts().to_dict()}')
samp[['id', 'text']].to_csv(os.path.join(INP, 'sentsample_input.csv'), index=False)
samp.drop(columns=['text']).to_csv(os.path.join(INP, 'sentsample_meta.csv'), index=False)
