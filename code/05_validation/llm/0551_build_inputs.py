"""Build the LLM annotation inputs.

 1. ground truth: the 1,142 human-rated chunks of the validation survey (data/validation/unzipped/annotated_instances*.csv),
    joined with the human means / labels and the stored EMI (results/revision/07_validation_auc/validation_instances.csv)
 2. sample: 10,000 random speeches 2015-2025 from data/parliament_nhb.csv, one random chunk of ~45-90 words per speech,
    cut at sentence boundaries; speaker header stripped; chunk-level EMI scored afterwards with code/03_scoring/0304_score_emi.py
 3. annotate_input.csv = both, columns id,text  (id prefix gt_ / s_)

Run from code/:  .venv/Scripts/python.exe code/05_validation/llm/0551_build_inputs.py
"""
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))   # German-EMI repository root: data in data/, outputs in results/
import os, re, glob, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = REPO_ROOT
INP = os.path.join(REPO_ROOT, 'data', 'llm', 'inputs'); os.makedirs(INP, exist_ok=True)
rng = np.random.default_rng(20261015)
N_SAMPLE, MIN_WORDS, TARGET, MAX_WORDS = 10_000, 45, 60, 90

# ---------------- 1. ground truth ----------------------------------------------------------------------------
fs = sorted(glob.glob(os.path.join(REPO_ROOT, 'data', 'validation', 'unzipped', 'annotated_instances*.csv')))
ann = pd.concat([pd.read_csv(f, encoding='utf-8', dtype={'instance_id': str}) for f in fs])
ann = ann[~ann.instance_id.str.contains('testing', na=False)]
txt = ann.drop_duplicates('instance_id').set_index('instance_id').displayed_text.str.strip()
gt = pd.read_csv(os.path.join(REPO_ROOT, 'results', 'revision', '07_validation_auc', 'validation_instances.csv'), dtype={'instance_id': str})
gt['text'] = gt.instance_id.map(txt); miss = gt.text.isna().sum(); gt = gt.dropna(subset=['text'])
gt['id'] = 'gt_' + gt.instance_id
gt.to_csv(os.path.join(INP, 'groundtruth_meta.csv'), index=False)
gt[['id', 'text']].to_csv(os.path.join(INP, 'groundtruth_input.csv'), index=False)
print(f'ground truth: {len(gt)} chunks with text ({miss} without), words median {gt.text.str.split().str.len().median():.0f}')

# ---------------- 2. sample ----------------------------------------------------------------------------------
cols = ['id', 'date', 'electoralTerm', 'politicianId', 'firstName', 'lastName', 'party', 'leaning', 'text', 'length',
        'emi_w2vparliament', 'evidence_z_w2vparliament', 'intuition_z_w2vparliament']
sc = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'parliament_nhb.csv'), usecols=lambda c: c in cols, dtype={'id': str})
sc = sc.dropna(subset=['text', 'politicianId', 'emi_w2vparliament'])
sc['date'] = pd.to_datetime(sc.date, utc=True).dt.tz_localize(None)
sc['n_words'] = sc.text.str.split().str.len()
pool = sc[sc.n_words >= MIN_WORDS]
print(f'speeches with text: {len(sc):,}; with >= {MIN_WORDS} words: {len(pool):,}')
samp = pool.sample(N_SAMPLE, random_state=20261015).copy()

def strip_header(text, last):
    """speaker header 'Name (Party), Function:' at the very start -> drop up to the first colon if the surname is in it"""
    m = re.match(r'^([^:]{0,220}?):\s', text)
    if m and isinstance(last, str) and last.split()[-1] in m.group(1): return text[m.end():]
    return text
SENT = re.compile(r'(?<=[.!?])\s+(?=[A-ZÄÖÜ„"(])')
def one_chunk(text):
    sents = SENT.split(text.strip()); chunks, cur, n = [], [], 0
    for s in sents:
        w = len(s.split())
        if n and n + w > MAX_WORDS: chunks.append(' '.join(cur)); cur, n = [], 0
        cur.append(s); n += w
        if n >= TARGET: chunks.append(' '.join(cur)); cur, n = [], 0
    if cur: chunks.append(' '.join(cur))
    ok = [c for c in chunks if MIN_WORDS <= len(c.split()) <= MAX_WORDS] or [c for c in chunks if len(c.split()) >= 25]
    if not ok: return None, None
    k = int(rng.integers(len(ok))); return ok[k], chunks.index(ok[k])
samp['text_body'] = [strip_header(t, l) for t, l in zip(samp.text, samp.lastName)]
samp[['chunk', 'chunk_idx']] = pd.DataFrame([one_chunk(t) for t in samp.text_body], index=samp.index)
samp = samp.dropna(subset=['chunk']); samp['chunk_words'] = samp.chunk.str.split().str.len()
samp['id_chunk'] = 's_' + samp.id + '_' + samp.chunk_idx.astype(int).astype(str)
meta = samp[['id_chunk', 'id', 'date', 'electoralTerm', 'politicianId', 'party', 'leaning', 'n_words', 'chunk_idx', 'chunk_words',
             'emi_w2vparliament', 'evidence_z_w2vparliament', 'intuition_z_w2vparliament']].rename(columns={'id': 'speech_id', 'id_chunk': 'id'})
meta.to_csv(os.path.join(INP, 'sample10k_meta.csv'), index=False)
samp[['id_chunk', 'chunk']].rename(columns={'id_chunk': 'id', 'chunk': 'text'}).to_csv(os.path.join(INP, 'sample10k_input.csv'), index=False)
print(f'sample: {len(samp):,} chunks; words median {samp.chunk_words.median():.0f} (IQR {samp.chunk_words.quantile(.25):.0f}-{samp.chunk_words.quantile(.75):.0f})')
print('by party:', samp.party.value_counts().to_dict()); print('by year:', samp.date.dt.year.value_counts().sort_index().to_dict())
print('example:', samp.chunk.iloc[0][:300])

# ---------------- 3. combined --------------------------------------------------------------------------------
both = pd.concat([gt[['id', 'text']], samp[['id_chunk', 'chunk']].rename(columns={'id_chunk': 'id', 'chunk': 'text'})])
both.to_csv(os.path.join(INP, 'annotate_input.csv'), index=False)
print(f'annotate_input.csv: {len(both):,} rows')
