"""Discriminant validity of EMI against moral and emotional language (revision, Reviewer 4).

Dictionaries (exact word-form lists, counts per 100 tokens of the lower-cased text):
  moral    multilingual Moral Political Dictionary, German file (Simonsen & Widmann 2025; data/dictionaries/mMPD_de.yml,
           MIT licence; MORALITY = union of all foundations, POSITIVE = virtue, NEGATIVE = vice)
  negemo   ed8 (Widmann & Wich 2023): union of ANGER, FEAR, DISGUST, SADNESS;  posemo: JOY, ENTHUSIASM, PRIDE, HOPE
           (ed8.yml is downloaded from https://github.com/tweedmann/3x8emotions on first use; free for academic research,
           cite the article)
Steps
  1  score the validation items (1,142 chunks of the 2025 survey, 300 chunks and 300 tweets of the 2026 surveys; the
     tweets are skipped unless data/twitter_nhb.csv with the tweet text is available)
  2  Spearman matrix (dictionary EMI, human EMI, moral, negemo, posemo); AUC of each score against the human label
  3  logistic model of the human label on EMI with and without the moral and emotion scores; AUC with and without
     controls; AUC of EMI residualised on the controls
  4  score all 59,170 speeches 2015-2025 (data/parliament_nhb.csv, raw text); constrained breakpoint model (Eq. 2) with
     and without the scores as document-level covariates; the same model with the moral and emotion scores as outcomes
Output: results/revision/15_discriminant_validity/{validation_scores.csv, correlations.csv, auc_logit.csv,
        speech_scores.csv, yearly_means.csv, breakpoint_models.csv, numbers.json, fig_discriminant.png/pdf}
"""
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))   # German-EMI repository root: data in data/, outputs in results/
import os, re, io, json, time, urllib.request, numpy as np, pandas as pd, yaml, statsmodels.api as sm
from scipy import stats
from sklearn.metrics import roc_auc_score
from statsmodels.regression.mixed_linear_model import MixedLM
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(REPO_ROOT, 'results', 'revision', '15_discriminant_validity'); os.makedirs(OUT, exist_ok=True)
VD = os.path.join(HERE, 'data_2026')
DICT_DIR = os.path.join(REPO_ROOT, 'data', 'dictionaries'); os.makedirs(DICT_DIR, exist_ok=True)
ED8_URL = 'https://raw.githubusercontent.com/tweedmann/3x8emotions/main/01_ed8/ed8.yml'
EMI = 'emi_w2vparliament'
BP = {'elec2017': pd.Timestamp('2017-07-01', tz='UTC'), 'elec2021': pd.Timestamp('2021-07-01', tz='UTC'), 'elec2025': pd.Timestamp('2025-01-01', tz='UTC')}
t0 = time.time(); log = lambda *a: print(f'[{time.time() - t0:6.0f}s]', *a, flush=True)

# ---------------------------------------------------------------- dictionaries
if not os.path.exists(os.path.join(DICT_DIR, 'ed8.yml')):
    log('downloading ed8.yml from', ED8_URL); urllib.request.urlretrieve(ED8_URL, os.path.join(DICT_DIR, 'ed8.yml'))
mm = yaml.safe_load(io.open(os.path.join(DICT_DIR, 'mMPD_de.yml'), encoding='utf-8'))['mMPD_de']
e8 = yaml.safe_load(io.open(os.path.join(DICT_DIR, 'ed8.yml'), encoding='utf-8'))['ed8']
S = lambda *cats, d: set(str(w).lower() for c in cats for w in d[c])
SETS = {'moral': S('MORALITY', d=mm), 'moral_virtue': S('POSITIVE', d=mm), 'moral_vice': S('NEGATIVE', d=mm),
        'negemo': S('ANGER', 'FEAR', 'DISGUST', 'SADNESS', d=e8), 'posemo': S('JOY', 'ENTHUSIASM', 'PRIDE', 'HOPE', d=e8),
        'anger': S('ANGER', d=e8), 'fear': S('FEAR', d=e8)}
log('dictionary sizes', {k: len(v) for k, v in SETS.items()})
TOK = re.compile(r'[a-zäöüß]+')


def score(text):
    toks = TOK.findall(str(text).lower()); n = len(toks)
    out = {'n_tokens': n}
    for k, s in SETS.items(): out[k] = 100.0 * sum(1 for t in toks if t in s) / n if n else np.nan
    return out


def score_frame(texts):
    return pd.DataFrame([score(t) for t in texts])


# ---------------------------------------------------------------- 1 validation items
g24 = pd.read_csv(os.path.join(VD, 'groundtruth_parliament_2024.csv'), dtype={'id': str}).drop_duplicates('id')
g26 = pd.read_csv(os.path.join(VD, 'groundtruth_parliament_2026.csv'), dtype={'id': str})
sets = [('Bundestag 2025', g24), ('Bundestag 2026', g26)]
gt = pd.read_csv(os.path.join(VD, 'groundtruth_twitter_2026.csv'), dtype={'tweet_id': str})
TW = os.path.join(REPO_ROOT, 'data', 'twitter_nhb.csv')
if os.path.exists(TW):   # tweet texts are not part of the public data (X Developer Agreement); looked up where the corpus is available
    ids = set(gt.tweet_id); tw = pd.concat([c[c.id.isin(ids)] for c in pd.read_csv(TW, usecols=['id', 'text'], dtype={'id': str}, chunksize=500000)]).drop_duplicates('id')
    gt = gt.merge(tw.rename(columns={'id': 'tweet_id'}), on='tweet_id', how='left'); sets.append(('Twitter 2026', gt))
else:
    log('data/twitter_nhb.csv not available: tweets skipped')
V = []
for name, g in sets:
    g = g.dropna(subset=['ev_mean', 'iu_mean', 'emi', 'text']).copy()
    sc = score_frame(g.text); sc.index = g.index
    v = pd.concat([g[['emi', 'ev_mean', 'iu_mean']], sc], axis=1); v['set'] = name; v['human_emi'] = v.ev_mean - v.iu_mean
    v['label'] = np.where(v.human_emi > 0, 1, np.where(v.human_emi < 0, 0, np.nan)); V.append(v)
V = pd.concat(V, ignore_index=True); V.to_csv(os.path.join(OUT, 'validation_scores.csv'), index=False)
log('validation items scored', V.groupby('set').size().to_dict())

# ---------------------------------------------------------------- 2 correlations and AUCs
VARS = ['emi', 'human_emi', 'moral', 'moral_virtue', 'moral_vice', 'negemo', 'posemo']
rows = []; NUM = {'dictionary_sizes': {k: len(v) for k, v in SETS.items()}}
groups = {'Bundestag pooled': V[V.set.str.startswith('Bundestag')], 'Bundestag 2025': V[V.set == 'Bundestag 2025'], 'Bundestag 2026': V[V.set == 'Bundestag 2026']}
if (V.set == 'Twitter 2026').any(): groups['Twitter 2026'] = V[V.set == 'Twitter 2026']
for gname, g in groups.items():
    for a in VARS:
        for b in VARS:
            if a < b:
                r, p = stats.spearmanr(g[a], g[b], nan_policy='omit'); rows.append(dict(set=gname, x=a, y=b, rho=r, p=p, n=int(g[[a, b]].dropna().shape[0])))
C = pd.DataFrame(rows); C.to_csv(os.path.join(OUT, 'correlations.csv'), index=False)
auc_rows = []
for gname, g in groups.items():
    gl = g.dropna(subset=['label']); y = gl.label.astype(int)
    for v in ['emi', 'moral', 'moral_virtue', 'moral_vice', 'negemo', 'posemo']:
        a = roc_auc_score(y, gl[v]); auc_rows.append(dict(set=gname, score=v, auc=a, auc_oriented=max(a, 1 - a), n=len(gl)))
A = pd.DataFrame(auc_rows)
log('AUCs\n' + A.pivot(index='score', columns='set', values='auc').round(3).to_string())

# ---------------------------------------------------------------- 3 logistic models with controls
def z(x): return (x - x.mean()) / x.std()


lg = []
for gname, g in groups.items():
    gl = g.dropna(subset=['label']).copy(); y = gl.label.astype(int)
    gl['log_tokens'] = np.log(gl.n_tokens.clip(lower=1))
    X0 = sm.add_constant(pd.DataFrame({'emi': z(gl.emi)}))
    ctrl = pd.DataFrame({'moral': z(gl.moral), 'negemo': z(gl.negemo), 'posemo': z(gl.posemo), 'log_tokens': z(gl.log_tokens)})
    X1 = sm.add_constant(pd.concat([X0[['emi']], ctrl], axis=1)); Xc = sm.add_constant(ctrl)
    m0 = sm.Logit(y, X0).fit(disp=0); m1 = sm.Logit(y, X1).fit(disp=0); mc = sm.Logit(y, Xc).fit(disp=0)
    resid = sm.OLS(z(gl.emi), sm.add_constant(ctrl)).fit().resid
    lg.append(dict(set=gname, n=len(gl), emi_coef_alone=m0.params['emi'], emi_se_alone=m0.bse['emi'], emi_coef_with_controls=m1.params['emi'], emi_se_with_controls=m1.bse['emi'],
                   moral_coef=m1.params['moral'], negemo_coef=m1.params['negemo'], posemo_coef=m1.params['posemo'],
                   auc_emi=roc_auc_score(y, m0.predict(X0)), auc_controls_only=roc_auc_score(y, mc.predict(Xc)), auc_emi_plus_controls=roc_auc_score(y, m1.predict(X1)),
                   auc_emi_residualised=roc_auc_score(y, resid), pseudo_r2_emi=m0.prsquared, pseudo_r2_controls=mc.prsquared, pseudo_r2_both=m1.prsquared))
L = pd.DataFrame(lg); L.to_csv(os.path.join(OUT, 'auc_logit.csv'), index=False)
log('logit\n' + L.round(3).to_string(index=False))
NUM['validation'] = {r['set']: {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in r.items() if k != 'set'} for r in lg}
NUM['correlations_bundestag_pooled'] = {f'{r.x}~{r.y}': round(float(r.rho), 3) for r in C[C.set == 'Bundestag pooled'].itertuples()}
if 'Twitter 2026' in groups: NUM['correlations_twitter'] = {f'{r.x}~{r.y}': round(float(r.rho), 3) for r in C[C.set == 'Twitter 2026'].itertuples()}

# ---------------------------------------------------------------- 4 speeches 2015-2025
par = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'parliament_nhb.csv'), usecols=['id', 'date', 'party', 'leaning', 'politicianId', 'text', EMI], dtype={'id': str})
par['date'] = pd.to_datetime(par.date, utc=True)
sp = os.path.join(OUT, 'speech_scores.csv')
if os.path.exists(sp):
    scs = pd.read_csv(sp, dtype={'id': str})
else:
    scs = score_frame(par.text); scs.insert(0, 'id', par.id.values); scs.to_csv(sp, index=False)
par = par.drop(columns=['text']).merge(scs, on='id'); log('speeches scored', len(par))
par['year'] = par.date.dt.year
yearly = par.groupby('year')[[EMI, 'moral', 'negemo', 'posemo']].mean(); yearly.to_csv(os.path.join(OUT, 'yearly_means.csv'))
pre, post = par[par.date < '2017-10-24'], par[par.date >= '2017-10-24']
NUM['speeches'] = {'n': int(len(par)), 'rate_means': par[['moral', 'negemo', 'posemo']].mean().round(3).to_dict(),
                   'pre_post_2017': {k: dict(pre=float(pre[k].mean()), post=float(post[k].mean()), d=float((post[k].mean() - pre[k].mean()) / par[k].std())) for k in ['moral', 'negemo', 'posemo', EMI]},
                   'speech_level_spearman': {k: float(stats.spearmanr(par[EMI], par[k]).correlation) for k in ['moral', 'negemo', 'posemo']}}
log('pre/post', NUM['speeches']['pre_post_2017'])


def prep_bp(d, y):
    m = d[[y, 'time_quarterly', 'leaning', 'politicianId', 'moral', 'negemo', 'posemo']].copy()
    q = m.groupby(['politicianId', 'time_quarterly'])[y].mean().rename('_q').reset_index(); q['y_lag1'] = q.groupby('politicianId')['_q'].shift(1)
    m = m.merge(q[['politicianId', 'time_quarterly', 'y_lag1']], on=['politicianId', 'time_quarterly']).dropna()
    tm = m.time_quarterly.mean(); m['time_centered'] = m.time_quarterly - tm
    for k, ts in BP.items():
        bq = int(f'{ts.year}{ts.quarter}'); m[f'slope_{k}'] = np.maximum(0, m.time_centered - (bq - tm)); m[f'level_{k}'] = (m.time_quarterly >= bq).astype(float)
    m['leaning'] = pd.Categorical(m.leaning, categories=['middle', 'left', 'right'])
    for c in ['moral', 'negemo', 'posemo']: m[c + '_z'] = z(m[c])
    return m.reset_index(drop=True)


def fit(m, formula, tag):
    t1 = time.time(); r = None
    for kw in ({}, {'method': 'powell'}, {'method': 'lbfgs', 'maxiter': 2000}):
        try:
            cand = MixedLM.from_formula(formula, groups=m.politicianId, data=m).fit(reml=False, **kw)
            if r is None: r = cand
            if cand.cov_re.iloc[0, 0] > 1e-6 and np.isfinite(cand.params).all(): r = cand; break
        except Exception as e:
            log(f'  {tag}: {kw or "default"} raised {type(e).__name__}: {str(e)[:100]}')
    ci = r.conf_int(); t = pd.DataFrame({'coef': r.params, 'se': r.bse, 'p': r.pvalues, 'lo': ci[0], 'hi': ci[1]})
    t['model'] = tag; t['n'] = len(m); t['n_pol'] = m.politicianId.nunique(); t['group_var'] = r.cov_re.iloc[0, 0]; t['converged'] = bool(r.converged)
    log(f'{tag}: N={len(m):,} level2017={r.params["level_elec2017"]:.3f} level2021={r.params["level_elec2021"]:.3f} level2025={r.params["level_elec2025"]:.3f} ({round(time.time() - t1)}s)')
    return t


BASE = ' ~ time_centered + leaning + time_centered:leaning + slope_elec2017 + slope_elec2021 + slope_elec2025 + level_elec2017 + level_elec2021 + level_elec2025 + y_lag1'
d = par.dropna(subset=['politicianId', 'party', 'leaning', EMI, 'moral']).copy(); d['politicianId'] = d.politicianId.astype(int)
d['time_quarterly'] = (d.date.dt.year * 10 + d.date.dt.quarter).astype(int)
m_emi = prep_bp(d, EMI); fits = []
fits.append(fit(m_emi, EMI + BASE, 'EMI|baseline'))
fits.append(fit(m_emi, EMI + BASE + ' + moral_z + negemo_z + posemo_z', 'EMI|+moral+emotion'))
for yv in ['moral', 'negemo', 'posemo']:
    dz = d.copy(); dz[yv + '_out'] = z(dz[yv]); my = prep_bp(dz.rename(columns={yv + '_out': 'yout'}), 'yout')
    fits.append(fit(my, 'yout' + BASE, f'{yv} (z) as outcome'))
R = pd.concat(fits); R.index.name = 'term'; R.to_csv(os.path.join(OUT, 'breakpoint_models.csv'))
KEY = ['level_elec2017', 'level_elec2021', 'level_elec2025', 'slope_elec2021', 'moral_z', 'negemo_z', 'posemo_z']
NUM['breakpoint'] = {tag: {k: dict(coef=float(R[(R.model == tag)].loc[k, 'coef']), se=float(R[(R.model == tag)].loc[k, 'se']), p=float(R[(R.model == tag)].loc[k, 'p'])) for k in KEY if k in R[R.model == tag].index} for tag in R.model.unique()}
json.dump(NUM, open(os.path.join(OUT, 'numbers.json'), 'w'), indent=1, default=float)

# ---------------------------------------------------------------- figure
plt.rcParams.update({'font.family': 'Arial', 'font.size': 11})
fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={'width_ratios': [1, 1.15]})
cp = groups['Bundestag pooled']; lab = {'emi': 'dictionary EMI', 'human_emi': 'human EMI', 'moral': 'moral', 'negemo': 'negative\nemotion', 'posemo': 'positive\nemotion'}
ks = ['emi', 'human_emi', 'moral', 'negemo', 'posemo']; M = cp[ks].corr(method='spearman').values
im = axes[0].imshow(M, cmap='RdBu_r', vmin=-1, vmax=1)
for i in range(len(ks)):
    for j in range(len(ks)): axes[0].text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=10, color='white' if abs(M[i, j]) > 0.6 else 'black')
axes[0].set_xticks(range(len(ks))); axes[0].set_xticklabels([lab[k] for k in ks], rotation=30, ha='right'); axes[0].set_yticks(range(len(ks))); axes[0].set_yticklabels([lab[k] for k in ks])
axes[0].set_title(f'Spearman correlations, {len(cp):,} Bundestag chunks', loc='left', fontsize=11); fig.colorbar(im, ax=axes[0], fraction=0.045)
axes[0].text(-0.15, 1.08, 'A', transform=axes[0].transAxes, fontsize=14, fontweight='bold')
tags = ['EMI|baseline', 'EMI|+moral+emotion', 'moral (z) as outcome', 'negemo (z) as outcome', 'posemo (z) as outcome']
names = {'EMI|baseline': 'EMI, baseline', 'EMI|+moral+emotion': 'EMI, with moral and\nemotion covariates', 'moral (z) as outcome': 'moral language (z)\nas outcome', 'negemo (z) as outcome': 'negative emotion (z)\nas outcome', 'posemo (z) as outcome': 'positive emotion (z)\nas outcome'}
cols = {'level_elec2017': '#D55E00', 'level_elec2021': '#4D4D4D', 'level_elec2025': '#0072B2'}
for i, tag in enumerate(tags):
    for j, k in enumerate(cols):
        r = R[R.model == tag].loc[k]; axes[1].errorbar(r.coef, i + (j - 1) * 0.22, xerr=1.96 * r.se, fmt='o', color=cols[k], capsize=3, label=k.replace('level_elec', '') if i == 0 else None)
axes[1].axvline(0, color='black', lw=0.8); axes[1].set_yticks(range(len(tags))); axes[1].set_yticklabels([names[t] for t in tags]); axes[1].invert_yaxis()
axes[1].set_xlabel('Level shift at the election (95% CI)'); axes[1].legend(title='election', frameon=False, loc='lower left'); axes[1].grid(axis='x', alpha=0.3)
axes[1].set_title('Constrained breakpoint model, 2015–2025 speeches', loc='left', fontsize=11); axes[1].text(-0.42, 1.08, 'B', transform=axes[1].transAxes, fontsize=14, fontweight='bold')
fig.tight_layout()
for ext in ('png', 'pdf'): fig.savefig(os.path.join(OUT, f'fig_discriminant.{ext}'), dpi=200, bbox_inches='tight')
log('done')
