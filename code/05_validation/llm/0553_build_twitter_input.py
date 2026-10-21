"""Sample 60,000 tweets spread evenly over time from data/twitter_nhb.csv for LLM annotation.
Rules: original tweets only (no retweets), a known leaning (the breakpoint model needs it), an EMI score, >= 5 words;
equal number per quarter (remainder spread over the quarters with most tweets). Raw tweet text, URLs removed.
Output: inputs/twitter_input.csv (id,text), inputs/twitter_meta.csv
Run from code/:  .venv/Scripts/python.exe code/05_validation/llm/0553_build_twitter_input.py
"""
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))   # German-EMI repository root: data in data/, outputs in results/
import os, re, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = REPO_ROOT; INP = os.path.join(REPO_ROOT, 'data', 'llm', 'inputs')
N = 60_000; SEED = 20261015
cols = ['id', 'author_id', 'created_at', 'party', 'leaning', 'retweeted', 'reply', 'text', 'length', 'emi_w2vparliament', 'evidence_z_w2vparliament', 'intuition_z_w2vparliament']
parts = []
for ch in pd.read_csv(os.path.join(REPO_ROOT, 'data', 'twitter_nhb.csv'), usecols=cols, dtype={'id': str, 'author_id': str}, chunksize=500_000):
    ch = ch[(ch.retweeted.astype(str) != 'True') & ch.leaning.notna() & ch.emi_w2vparliament.notna() & ch.text.notna()]
    ch = ch[ch.text.str.split().str.len() >= 5]; parts.append(ch)
T = pd.concat(parts); T['created_at'] = pd.to_datetime(T.created_at, utc=True).dt.tz_localize(None); T['q'] = T.created_at.dt.to_period('Q')
print(f'eligible tweets: {len(T):,}; {T.created_at.min().date()} to {T.created_at.max().date()}; quarters {T.q.nunique()}')
per_q = T.q.value_counts().sort_index(); k = N // len(per_q)
take = per_q.clip(upper=k); short = N - take.sum()
if short > 0:  # give the remainder to quarters that have room, proportionally
    room = (per_q - take); extra = (room / room.sum() * short).round().astype(int); take = (take + extra).clip(upper=per_q)
S = pd.concat([g.sample(min(int(take[q]), len(g)), random_state=SEED) for q, g in T.groupby('q')])
S['text'] = S.text.str.replace(r'https?://\S+', '', regex=True).str.replace(r'\s+', ' ', regex=True).str.strip()
S = S[S.text.str.split().str.len() >= 5]
print(f'sampled {len(S):,}; per quarter min {S.q.value_counts().min()} max {S.q.value_counts().max()}; by leaning {S.leaning.value_counts().to_dict()}')
print('by year:', S.created_at.dt.year.value_counts().sort_index().to_dict()); print('example:', S.text.iloc[0][:200])
S[['id', 'text']].to_csv(os.path.join(INP, 'twitter_input.csv'), index=False)
S.drop(columns=['text', 'q']).rename(columns={'created_at': 'date'}).to_csv(os.path.join(INP, 'twitter_meta.csv'), index=False)
print('written inputs/twitter_input.csv, inputs/twitter_meta.csv')
