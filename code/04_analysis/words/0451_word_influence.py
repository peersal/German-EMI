"""Word-level contributions to the DDR EMI, per party and period, for the Bundestag and Twitter corpora.

For every vocabulary word w of the paper's parliament word2vec model the word score is
    s_w = cos(v_w, E) - cos(v_w, I)
with E, I the evidence / intuition dictionary centroids used by the DDR (see score_emi.dictionary_vectors). The
document vector of the DDR is the mean of its word vectors, so the group-level score is (to first order) the
frequency-weighted mean of s_w. The contribution of word w to the EMI of group g (party x period) is
    delta_{w,g} = p_{w,g} * (s_w - s_bar)         p = share of the group's in-vocabulary tokens, s_bar = corpus mean
and the word-shift contribution against the rest of the corpus is
    shift_{w,g} = (p_{w,g} - p_{w,ref}) * (s_w - s_bar).
Writes results/word_influence/<corpus>_words.csv (one row per word x party x period with freq, p, s_w, delta,
shift) and <corpus>_quarterly.csv (quarterly mean EMI per party for the line panels).

  python code/04_analysis/words/0451_word_influence.py parliament
  python code/04_analysis/words/0451_word_influence.py twitter          (reads the 6.2M tweets in chunks, ~10 min)
"""
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))   # German-EMI repository root: data in data/, outputs in results/
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = REPO_ROOT
sys.path.insert(0, os.path.join(REPO_ROOT, 'code', '03_scoring')); import importlib; S = importlib.import_module('0304_score_emi')
OUT = os.path.join(ROOT, 'results', 'word_influence'); os.makedirs(OUT, exist_ok=True)
EMI = 'emi_w2vparliament'
PERIODS = [('2015-2019', '2015-01-01', '2020-01-01'), ('2020-2025', '2020-01-01', '2026-01-01')]   # early / late period as in the paper
GROUP = os.environ.get('WI_GROUP', 'party')                 # 'party' or 'leaning'
if os.environ.get('WI_WINDOWS') == 'biennial':              # timeline figure: two-year windows
    PERIODS = [('2015-2016', '2015-01-01', '2017-01-01'), ('2017-2018', '2017-01-01', '2019-01-01'), ('2019-2020', '2019-01-01', '2021-01-01'),
               ('2021-2022', '2021-01-01', '2023-01-01'), ('2023-2025', '2023-01-01', '2026-01-01')]
SUFFIX = '' if GROUP == 'party' and not os.environ.get('WI_WINDOWS') else f"_{GROUP}_{os.environ.get('WI_WINDOWS', 'paper')}"
PARTIES = ['CDU/CSU', 'SPD', 'FDP', 'Alliance 90/The Greens', 'The Left Party', 'AFD'] if GROUP == 'party' else ['left', 'middle', 'right']
MIN_FREQ = 20                                             # a word needs this many tokens in a group to be ranked

corpus = sys.argv[1] if len(sys.argv) > 1 else 'parliament'
kv = S.load_model(os.path.join(REPO_ROOT, 'data', 'parliament_model')); tok = kv.tokenizer
W = kv.model[0].emb_layer.weight.detach().numpy(); vocab = np.array(tok.vocab, dtype=object); w2i = tok.word2idx
stop = set(tok.stop_words) | S.nltk_german_stopwords()
E, I, _, _ = S.dictionary_vectors(kv, S.nltk_german_stopwords())
print('vocab', len(vocab), 'dim', W.shape[1], flush=True)
s_w = np.zeros(len(vocab), dtype=np.float32)
for a in range(0, len(vocab), 200000):
    V = W[a:a + 200000]; s_w[a:a + 200000] = S.cosine(V, E) - S.cosine(V, I)
s_w = np.nan_to_num(s_w)
stop_idx = np.array([w2i[w] for w in stop if w in w2i], dtype=np.int64); stop_mask = np.zeros(len(vocab), bool); stop_mask[stop_idx] = True; stop_mask[0] = True

keys = [(p, per[0]) for p in PARTIES for per in PERIODS]; kidx = {k: i for i, k in enumerate(keys)}
counts = np.zeros((len(keys), len(vocab)), dtype=np.int32)
qsum, qn = {}, {}


def period_of(dates):
    out = pd.Series(pd.NA, index=dates.index, dtype=object)
    for lab, a, b in PERIODS:
        out[(dates >= pd.Timestamp(a, tz='UTC')) & (dates < pd.Timestamp(b, tz='UTC'))] = lab
    return out


def consume(df, datecol):
    df = df.dropna(subset=['party', 'text_clean', EMI]); df = df[df.party.isin(PARTIES)]
    d = pd.to_datetime(df[datecol], utc=True, errors='coerce'); df = df.assign(period=period_of(d), q=d.dt.to_period('Q').astype(str)).dropna(subset=['period'])
    g = df.groupby(['party', 'q'])[EMI].agg(['sum', 'count'])
    for (p, q), r in g.iterrows():
        qsum[(p, q)] = qsum.get((p, q), 0.0) + r['sum']; qn[(p, q)] = qn.get((p, q), 0) + int(r['count'])
    for (p, per), sub in df.groupby(['party', 'period']):
        toks = sub['text_clean'].astype(str).str.split().explode()
        idx = toks.map(w2i).dropna().astype(np.int64).to_numpy()
        counts[kidx[(p, per)]] += np.bincount(idx, minlength=len(vocab)).astype(np.int32)


if corpus == 'parliament':
    df = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'parliament_nhb.csv'), usecols=['date', 'party', 'leaning', 'text_clean', EMI]); df['party'] = df[GROUP]
    consume(df, 'date'); print('speeches', len(df), flush=True)
else:
    n = 0
    for ch in pd.read_csv(os.path.join(REPO_ROOT, 'data', 'twitter_nhb.csv'), usecols=['created_at', 'party', 'leaning', 'text_clean', EMI], chunksize=250000):
        ch['party'] = ch[GROUP]; consume(ch, 'created_at'); n += len(ch); print(f'  {n:,} tweets', flush=True)

counts[:, stop_mask] = 0
tot = counts.sum(axis=1, keepdims=True).astype(np.float64)
P = counts / np.maximum(tot, 1)                                            # token shares per group
all_counts = counts.sum(axis=0).astype(np.float64); p_ref = all_counts / all_counts.sum()
s_bar = float((p_ref * s_w).sum())
print('corpus-mean word score s_bar =', round(s_bar, 4), flush=True)
rows = []
for k, (p, per) in enumerate(keys):
    idx = np.where(counts[k] >= MIN_FREQ)[0]
    if len(idx) == 0: continue
    delta = P[k, idx] * (s_w[idx] - s_bar); shift = (P[k, idx] - p_ref[idx]) * (s_w[idx] - s_bar)
    rows.append(pd.DataFrame({'party': p, 'period': per, 'word': vocab[idx], 'freq': counts[k, idx], 'p': P[k, idx], 'p_ref': p_ref[idx],
                              's_w': s_w[idx], 'delta': delta, 'shift': shift, 'group_tokens': int(tot[k, 0]),
                              'group_score': float((P[k] * s_w).sum())}))
R = pd.concat(rows); R.to_csv(os.path.join(OUT, f'{corpus}_words{SUFFIX}.csv'), index=False)
Q = pd.DataFrame([(p, q, qsum[(p, q)] / qn[(p, q)], qn[(p, q)]) for (p, q) in qsum], columns=['party', 'quarter', 'emi_mean', 'n']).sort_values(['party', 'quarter'])
Q.to_csv(os.path.join(OUT, f'{corpus}_quarterly{SUFFIX}.csv'), index=False)
G = R.groupby(['party', 'period'])[['group_tokens', 'group_score']].first().reset_index(); G['s_bar'] = s_bar
G.to_csv(os.path.join(OUT, f'{corpus}_groups{SUFFIX}.csv'), index=False)
print(G.to_string(index=False))
for (p, per), sub in R.groupby(['party', 'period']):
    top = sub.nlargest(6, 'delta').word.tolist(); bot = sub.nsmallest(6, 'delta').word.tolist()
    print(f'{p:24s} {per}  +{top}  -{bot}')
