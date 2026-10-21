"""Twitter: baseline linear mixed model (Eq. 1) with and without topic fixed effects on ONE identical sample,
with the BERTopic noise cluster (topic -1) kept as its own category; plus the pair on the sample without noise tweets.
Output: results/revision/08_topic_controls/twitter_same_sample_models.csv
Run:    .venv/Scripts/python.exe code/05_validation/topics/0542_topic_twitter_same_sample.py [--sample N]
"""
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))   # German-EMI repository root: data in data/, outputs in results/
import os, sys, time, argparse, numpy as np, pandas as pd
from statsmodels.regression.mixed_linear_model import MixedLM
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = REPO_ROOT
OUT = os.path.join(REPO_ROOT, 'results', 'revision', '08_topic_controls'); os.makedirs(OUT, exist_ok=True)
ap = argparse.ArgumentParser(); ap.add_argument('--sample', type=int, default=None); a = ap.parse_args()
EMI = 'emi_w2vparliament'
t0 = time.time()
df = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'twitter_topics_nhb.csv'),
                 usecols=['author_id', 'created_at', 'party', 'leaning', 'like_count', 'topic', EMI])
df = df.dropna(subset=['party', 'leaning', EMI, 'created_at', 'topic'])
df['created_at'] = pd.to_datetime(df['created_at'], utc=True, errors='coerce'); df = df.dropna(subset=['created_at'])
df['time_quarterly'] = (df.created_at.dt.year * 10 + df.created_at.dt.quarter).astype(int)
df['log_like_count'] = np.log1p(df.like_count.fillna(0).clip(lower=0))
df['topic'] = df.topic.astype(int)
if a.sample:
    df = df.sample(a.sample, random_state=0)
print('tweets', len(df), 'loaded in', round(time.time() - t0), 's', flush=True)

def prep(d):
    m = d[[EMI, 'time_quarterly', 'leaning', 'author_id', 'log_like_count', 'topic']].copy()
    q = m.groupby(['author_id', 'time_quarterly'])[EMI].mean().rename('_q').reset_index()
    q['emi_lag1'] = q.groupby('author_id')['_q'].shift(1)
    m = m.merge(q[['author_id', 'time_quarterly', 'emi_lag1']], on=['author_id', 'time_quarterly']).dropna()
    m['time_centered'] = m.time_quarterly - m.time_quarterly.mean()
    m['leaning'] = pd.Categorical(m.leaning, categories=['middle', 'left', 'right'])
    return m.reset_index(drop=True)

def topic_terms(m, keep_noise):
    counts = m.topic.value_counts(); ref = int(counts[counts.index != -1].idxmax()); cols = []
    for t in counts.index:
        t = int(t)
        if t == ref or (t == -1 and not keep_noise): continue
        name = f'topic_{"noise" if t == -1 else t}'; m[name] = (m.topic == t).astype(np.float32); cols.append(name)
    return cols

def fit(m, formula, tag):
    t1 = time.time()
    r = MixedLM.from_formula(formula, groups=m.author_id, data=m).fit(reml=False)
    ci = r.conf_int(); t = pd.DataFrame({'coef': r.params, 'se': r.bse, 'p': r.pvalues, 'lo': ci[0], 'hi': ci[1]})
    t['model'] = tag; t['n'] = len(m); t['n_authors'] = m.author_id.nunique(); t['group_var'] = r.cov_re.iloc[0, 0]; t['llf'] = r.llf; t['converged'] = bool(r.converged)
    print(f'{tag}: N={len(m):,} time={r.params["time_centered"]:.5f} right={r.params["leaning[T.right]"]:.4f} '
          f'converged={r.converged} ({round(time.time() - t1)}s)', flush=True)
    return t[~t.index.str.startswith('topic_')]

base = f'{EMI} ~ time_centered * leaning + log_like_count + emi_lag1'
m_all = prep(df); m_nn = prep(df[df.topic != -1])
res = [fit(m_all, base, 'all|baseline')]
mk = m_all.copy(); cols = topic_terms(mk, True); res.append(fit(mk, base + ' + ' + ' + '.join(cols), 'all|topicFE_noise_kept'))
res.append(fit(m_nn, base, 'no_noise|baseline'))
mn = m_nn.copy(); cols = topic_terms(mn, False); res.append(fit(mn, base + ' + ' + ' + '.join(cols), 'no_noise|topicFE'))
out = pd.concat(res); out.index.name = 'term'
out.to_csv(os.path.join(OUT, 'twitter_same_sample_models.csv' if not a.sample else f'twitter_same_sample_models_sample{a.sample}.csv'))
print('done in', round(time.time() - t0), 's')
