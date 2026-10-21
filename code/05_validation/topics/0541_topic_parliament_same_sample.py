"""Parliament: constrained (Eq. 2) and full (Eq. 3) breakpoint models with and without topic fixed effects on ONE
identical sample, BERTopic noise cluster (topic -1) kept as its own category; plus the pair on the sample without
noise speeches. Output: results/revision/08_topic_controls/parliament_same_sample_models.csv (+ key coefficients).
"""
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))   # German-EMI repository root: data in data/, outputs in results/
import os, time, traceback, numpy as np, pandas as pd
from statsmodels.regression.mixed_linear_model import MixedLM
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = REPO_ROOT
OUT = os.path.join(REPO_ROOT, 'results', 'revision', '08_topic_controls'); os.makedirs(OUT, exist_ok=True)
EMI = 'emi_w2vparliament'
BP = {'elec2017': pd.Timestamp('2017-07-01', tz='UTC'), 'elec2021': pd.Timestamp('2021-07-01', tz='UTC'), 'elec2025': pd.Timestamp('2025-01-01', tz='UTC')}

tp = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'parliament_topics_nhb.csv'), usecols=['id', 'topic'], dtype={'id': str})
par = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'parliament_nhb.csv'), usecols=['id', 'date', 'party', 'leaning', 'politicianId', EMI], dtype={'id': str})
par['date'] = pd.to_datetime(par.date, utc=True)
pt = par.merge(tp, on='id').dropna(subset=['politicianId', 'party', 'leaning', EMI])
pt['politicianId'] = pt.politicianId.astype(int); pt['topic'] = pt.topic.astype(int)
pt['time_quarterly'] = (pt.date.dt.year * 10 + pt.date.dt.quarter).astype(int)
print('speeches with topics', len(pt), '| noise share', round((pt.topic == -1).mean(), 3), flush=True)

def prep_bp(d):
    m = d[[EMI, 'time_quarterly', 'leaning', 'politicianId', 'topic']].copy()
    q = m.groupby(['politicianId', 'time_quarterly'])[EMI].mean().rename('_q').reset_index()
    q['emi_lag1'] = q.groupby('politicianId')['_q'].shift(1)
    m = m.merge(q[['politicianId', 'time_quarterly', 'emi_lag1']], on=['politicianId', 'time_quarterly']).dropna()
    tm = m.time_quarterly.mean(); m['time_centered'] = m.time_quarterly - tm
    for k, ts in BP.items():
        bq = int(f'{ts.year}{ts.quarter}'); m[f'slope_{k}'] = np.maximum(0, m.time_centered - (bq - tm)); m[f'level_{k}'] = (m.time_quarterly >= bq).astype(float)
    m['leaning'] = pd.Categorical(m.leaning, categories=['middle', 'left', 'right'])
    # explicit camp interactions as in the paper's full model (Table model3): the right camp (AfD) has no speeches
    # before the 2017 breakpoint, so its 2017 interactions are not identified and are omitted
    for k in BP:
        for camp in ('left', 'right'):
            if k == 'elec2017' and camp == 'right': continue
            ind = (m.leaning == camp).astype(float)
            m[f'slope_{k}_{camp}'] = m[f'slope_{k}'] * ind; m[f'level_{k}_{camp}'] = m[f'level_{k}'] * ind
    return m.reset_index(drop=True)

def topic_terms(m, keep_noise):
    counts = m.topic.value_counts(); ref = int(counts[counts.index != -1].idxmax()); cols = []
    for t in counts.index:
        t = int(t)
        if t == ref or (t == -1 and not keep_noise): continue
        name = f'topic_{"noise" if t == -1 else t}'; m[name] = (m.topic == t).astype(float); cols.append(name)
    return cols, ref

def bp_formula(full):
    t = ['time_centered', 'leaning', 'time_centered:leaning'] + [f'slope_{k}' for k in BP] + [f'level_{k}' for k in BP]
    if full: t += [f'{w}_{k}_{camp}' for k in BP for camp in ('left', 'right') for w in ('slope', 'level') if not (k == 'elec2017' and camp == 'right')]
    return f'{EMI} ~ ' + ' + '.join(t) + ' + emi_lag1'

def fit(m, formula, tag):
    t1 = time.time(); r = None; last = None
    for kw in ({}, {'method': 'powell'}, {'method': 'lbfgs', 'maxiter': 2000}):
        try:
            cand = MixedLM.from_formula(formula, groups=m.politicianId, data=m).fit(reml=False, **kw)
            if r is None: r = cand
            if cand.cov_re.iloc[0, 0] > 1e-6 and np.isfinite(cand.params).all():
                r = cand; break
            print(f'  {tag}: {kw or "default"} gave group_var={cand.cov_re.iloc[0,0]:.2e}, retrying', flush=True)
        except Exception as e:
            last = e; print(f'  {tag}: {kw or "default"} raised {type(e).__name__}: {str(e)[:120]}', flush=True)
    if r is None:
        raise RuntimeError(f'{tag}: all optimizers failed: {last}')
    ci = r.conf_int(); t = pd.DataFrame({'coef': r.params, 'se': r.bse, 'p': r.pvalues, 'lo': ci[0], 'hi': ci[1]})
    t['model'] = tag; t['n'] = len(m); t['n_pol'] = m.politicianId.nunique(); t['group_var'] = r.cov_re.iloc[0, 0]; t['llf'] = r.llf; t['converged'] = bool(r.converged)
    print(f'{tag}: N={len(m):,} level2017={r.params["level_elec2017"]:.3f} level2021={r.params["level_elec2021"]:.3f} '
          f'slope2021={r.params["slope_elec2021"]:.4f} group_var={r.cov_re.iloc[0,0]:.3f} converged={r.converged} ({round(time.time()-t1)}s)', flush=True)
    return t[~t.index.str.startswith('topic_')]

m_all = prep_bp(pt); m_nn = prep_bp(pt[pt.topic != -1])
fits = []
for full in (False, True):
    kind = 'full' if full else 'constrained'; f0 = bp_formula(full)
    fits.append(fit(m_all, f0, f'{kind}|all|baseline'))
    mk = m_all.copy(); cols, _ = topic_terms(mk, True); fits.append(fit(mk, f0 + ' + ' + ' + '.join(cols), f'{kind}|all|topicFE_noise_kept'))
    fits.append(fit(m_nn, f0, f'{kind}|no_noise|baseline'))
    mn = m_nn.copy(); cols, _ = topic_terms(mn, False); fits.append(fit(mn, f0 + ' + ' + ' + '.join(cols), f'{kind}|no_noise|topicFE'))
res = pd.concat(fits); res.index.name = 'term'
res.to_csv(os.path.join(OUT, 'parliament_same_sample_models.csv'))
key = ['level_elec2017', 'level_elec2021', 'level_elec2025', 'slope_elec2021', 'slope_elec2025', 'leaning[T.right]', 'leaning[T.left]', 'time_centered', 'emi_lag1']
res[res.index.isin(key)].reset_index().pivot(index='term', columns='model', values='coef').reindex(key).to_csv(os.path.join(OUT, 'parliament_key_coefficients.csv'))
print('done')
