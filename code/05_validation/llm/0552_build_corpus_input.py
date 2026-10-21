"""Build the full-corpus LLM input: every speech in data/parliament_nhb.csv (2015-2025), whole speech text,
speaker header stripped, truncated to MAX_WORDS words. Output: inputs/corpus_input.csv (id,text) + inputs/corpus_meta.csv.
Run from code/:  .venv/Scripts/python.exe code/05_validation/llm/0552_build_corpus_input.py
"""
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))   # German-EMI repository root: data in data/, outputs in results/
import os, re, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = REPO_ROOT; INP = os.path.join(REPO_ROOT, 'data', 'llm', 'inputs')
MAX_WORDS = 1500
cols = ['id', 'date', 'electoralTerm', 'politicianId', 'lastName', 'party', 'leaning', 'text', 'length', 'emi_w2vparliament', 'evidence_z_w2vparliament', 'intuition_z_w2vparliament']
sc = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'parliament_nhb.csv'), usecols=lambda c: c in cols, dtype={'id': str}).dropna(subset=['text'])
def strip_header(text, last):
    m = re.match(r'^([^:]{0,220}?):\s', text)
    if m and isinstance(last, str) and last.split()[-1] in m.group(1): return text[m.end():]
    return text
sc['text_body'] = [strip_header(t, l) for t, l in zip(sc.text, sc.lastName)]
w = sc.text_body.str.split(); sc['n_words'] = w.str.len(); sc['text_body'] = w.str[:MAX_WORDS].str.join(' ')
stripped = (sc.text_body.str.len() < sc.text.str.len()).mean(); print(f'speeches {len(sc):,}; header stripped in {stripped:.1%}; words median {sc.n_words.median():.0f}, p95 {sc.n_words.quantile(.95):.0f}, truncated {int((sc.n_words > MAX_WORDS).sum()):,}')
sc = sc[sc.n_words >= 10]
sc[['id', 'text_body']].rename(columns={'text_body': 'text'}).to_csv(os.path.join(INP, 'corpus_input.csv'), index=False)
sc.drop(columns=['text', 'text_body', 'lastName']).to_csv(os.path.join(INP, 'corpus_meta.csv'), index=False)
print('example:', sc.text_body.iloc[0][:250]); print(f"written {len(sc):,} rows, {os.path.getsize(os.path.join(INP, 'corpus_input.csv'))/1e6:.0f} MB")
