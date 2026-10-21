"""Time series and breakpoint model on the LLM annotations of the full 2015-2025 corpus.

Inputs: output/corpus_{rating,use_mention}_{model}.parquet (one row per speech), inputs/corpus_meta.csv
 1. quarterly means of the LLM composite (evidence - intuition), the two ratings, the four use/mention scales and
    the procedural scale, overall and by leaning; alongside the embedding EMI on the same speeches
 2. the paper's constrained breakpoint model (Eq. 2: linear trend x leaning, level + slope change at the three election
    breakpoints, lagged actor mean, random speaker intercepts) on each LLM measure
 3. figure: 4 panels (composite vs EMI; evidence/intuition ratings; use vs mention; by leaning)
Outputs: results/ts_quarterly.csv, ts_quarterly_leaning.csv, ts_breakpoints.csv, ts_numbers.json, timeseries.png/pdf
"""
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))   # German-EMI repository root: data in data/, outputs in results/
import os, sys, json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from statsmodels.regression.mixed_linear_model import MixedLM
HERE = os.path.dirname(os.path.abspath(__file__))
DS = sys.argv[1] if len(sys.argv) > 1 else 'corpus'          # corpus (Bundestag speeches) | twitter (60k tweet sample)
OUTD = os.path.join(REPO_ROOT, 'results', 'llm', 'parliament' if DS == 'corpus' else DS); os.makedirs(OUTD, exist_ok=True)
ACTOR = 'politicianId' if DS == 'corpus' else 'author_id'; RIGHT = 'AFD' if DS == 'corpus' else 'right'
print(f'dataset: {DS}')
MODELS = {'meta-llama_Llama-3.1-8B-Instruct': 'Llama', 'Qwen_Qwen2.5-7B-Instruct': 'Qwen'}
EMI = 'emi_w2vparliament'
def load(prompt):
    out = None
    for tag, short in MODELS.items():
        f = os.path.join(REPO_ROOT, 'data', 'llm', 'output', f'{DS}_{prompt}_{tag}.parquet')
        if not os.path.exists(f): print('missing', f); continue
        d = pd.read_parquet(f)
        if 'id' not in d.columns: d = d.reset_index().rename(columns={d.index.name or 'index': 'id'})
        d = d[['id'] + [c for c in d.columns if c != 'id' and not c.endswith('_missing') and c != 'parse_error']].drop_duplicates('id'); d['id'] = d.id.astype(str)
        d.columns = ['id'] + [f'{c}_{short}' for c in d.columns if c != 'id']
        out = d if out is None else out.merge(d, on='id', how='outer')
    return out
L = load('rating'); U = load('use_mention'); L = L.merge(U, on='id', how='outer') if U is not None else L
for c in [c for c in L.columns if c != 'id']: L[c] = pd.to_numeric(L[c], errors='coerce')
SC = ['evidenz_basiert', 'intuition_basiert', 'evidence_use', 'evidence_mention', 'intuition_use', 'intuition_mention', 'procedural']
for b in SC:
    cs = [f'{b}_{s}' for s in MODELS.values() if f'{b}_{s}' in L]
    if cs: L[b] = L[cs].mean(axis=1)
L['llm_emi'] = L.evidenz_basiert - L.intuition_basiert
for s in MODELS.values():
    if f'evidenz_basiert_{s}' in L: L[f'llm_emi_{s}'] = L[f'evidenz_basiert_{s}'] - L[f'intuition_basiert_{s}']
M = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'llm', 'inputs', f'{DS}_meta.csv'), dtype={'id': str, ACTOR: str}, parse_dates=['date'])
M['date'] = M.date.dt.tz_localize(None) if M.date.dt.tz is not None else M.date
D = M.merge(L, on='id'); D['q'] = D.date.dt.to_period('Q'); D['politicianId'] = D[ACTOR].astype(str).str.replace(r'\.0$', '', regex=True)
print(f'texts with LLM ratings: {len(D):,} of {len(M):,}; missing composite {D.llm_emi.isna().mean():.2%}')
print('inter-model rho on speeches: ' + ', '.join(f'{b} {D[f"{b}_Llama"].corr(D[f"{b}_Qwen"], method="spearman"):.2f}' for b in SC if f'{b}_Llama' in D))
print(f'EMI vs LLM composite, text level: rho {D[EMI].corr(D.llm_emi, method="spearman"):.3f}; evidence_z vs evidence {D.evidence_z_w2vparliament.corr(D.evidenz_basiert, method="spearman"):.3f}; intuition_z vs intuition {D.intuition_z_w2vparliament.corr(D.intuition_basiert, method="spearman"):.3f}')

# ---------------- 1. quarterly series ---------------------------------------------------------------------------
MEAS = ['llm_emi', 'evidenz_basiert', 'intuition_basiert', 'evidence_use', 'evidence_mention', 'intuition_use', 'intuition_mention', 'procedural', EMI]
Q = D.groupby('q')[MEAS].mean(); Q['n'] = D.groupby('q').size(); Q.to_csv(os.path.join(OUTD, 'ts_quarterly.csv'))
QL = D.groupby(['leaning', 'q'])[MEAS].mean(); QL['n'] = D.groupby(['leaning', 'q']).size(); QL.to_csv(os.path.join(OUTD, 'ts_quarterly_leaning.csv'))
QP = D.groupby(['party', 'q'])[MEAS].mean(); QP.to_csv(os.path.join(OUTD, 'ts_quarterly_party.csv'))
yr = D.groupby(D.date.dt.year)[MEAS].mean().round(3); print('\nyearly means:\n' + yr.to_string())
NUM = {'n_speeches': int(len(D)), 'rho_emi_llm_speech': float(D[EMI].corr(D.llm_emi, method='spearman')), 'yearly': yr.to_dict()}
# simple pre/post 2017 step, non-AfD
non = D[(D.party != 'AFD') if DS == 'corpus' else (D.leaning != 'right')]; pre = non[(non.date >= '2016-01-01') & (non.date < '2017-07-01')]; post = non[(non.date >= '2017-10-24') & (non.date < '2019-04-01')]
step = pd.DataFrame({'pre': pre[MEAS].mean(), 'post': post[MEAS].mean()}); step['diff'] = step.post - step.pre
from scipy.stats import ttest_ind
step['p'] = [ttest_ind(pre[m].dropna(), post[m].dropna()).pvalue for m in MEAS]; print('\nnon-AfD six-quarter step at 2017:\n' + step.round(4).to_string()); step.to_csv(os.path.join(OUTD, 'ts_step2017_nonafd.csv')); NUM['step2017_nonafd'] = step.to_dict()

# ---------------- 2. breakpoint model (constrained, Eq. 2) --------------------------------------------------------
BP = {'elec2017': 20173, 'elec2021': 20213, 'elec2025': 20251}
def prep(d, y):
    m = d[[y, 'q', 'leaning', 'politicianId']].dropna().copy(); m['time_quarterly'] = m.q.dt.year * 10 + m.q.dt.quarter
    qm = m.groupby(['politicianId', 'time_quarterly'])[y].mean().rename('_q').reset_index(); qm['lag1'] = qm.groupby('politicianId')['_q'].shift(1)
    m = m.merge(qm[['politicianId', 'time_quarterly', 'lag1']], on=['politicianId', 'time_quarterly']).dropna(); tm = m.time_quarterly.mean(); m['time_centered'] = m.time_quarterly - tm
    for k, bq in BP.items(): m[f'slope_{k}'] = np.maximum(0, m.time_centered - (bq - tm)); m[f'level_{k}'] = (m.time_quarterly >= bq).astype(float)
    m['leaning'] = pd.Categorical(m.leaning, categories=['middle', 'left', 'right']); m = m.rename(columns={y: 'y'}); return m.reset_index(drop=True)
base = 'y ~ time_centered + leaning + time_centered:leaning + ' + ' + '.join(f'slope_{k} + level_{k}' for k in BP) + ' + lag1'
rows = []
print('\nconstrained breakpoint model, random speaker intercepts (coef = level shift at each election):')
for y in ['llm_emi', 'llm_emi_Llama', 'llm_emi_Qwen', 'evidenz_basiert', 'intuition_basiert', 'evidence_use', 'evidence_mention', 'intuition_use', 'intuition_mention', EMI]:
    if y not in D: continue
    m = prep(D, y); mod = MixedLM.from_formula(base, groups=m.politicianId, data=m)
    try: r = mod.fit(reml=False)
    except Exception: r = mod.fit(reml=False, method='powell')
    sd = m.y.std(); out = dict(measure=y, sd=sd, n=int(r.nobs))
    for k in BP: out[f'{k}_level'] = r.params[f'level_{k}']; out[f'{k}_se'] = r.bse[f'level_{k}']; out[f'{k}_slope'] = r.params[f'slope_{k}']
    rows.append(out); print(f'   {y:20s} (sd {sd:.2f}, n={int(r.nobs):,}): 2017 {r.params["level_elec2017"]:+.3f} (se {r.bse["level_elec2017"]:.3f}, {r.params["level_elec2017"]/sd:+.2f} sd) | 2021 {r.params["level_elec2021"]:+.3f} | 2025 {r.params["level_elec2025"]:+.3f}')
BPT = pd.DataFrame(rows); BPT.to_csv(os.path.join(OUTD, 'ts_breakpoints.csv'), index=False); NUM['breakpoints'] = rows
json.dump(NUM, open(os.path.join(OUTD, 'ts_numbers.json'), 'w'), indent=1, default=float)

# ---------------- 3. figure -----------------------------------------------------------------------------------
INK, MUTED, GRID, ORANGE, BLUE, GREEN = '#222222', '#6E6E6E', '#E6E6E6', '#D55E00', '#0072B2', '#009E73'
LC = {'left': '#E66101', 'middle': '#4D4D4D', 'right': '#0072B2'}
plt.rcParams.update({'font.family': 'Arial', 'font.size': 7, 'axes.labelsize': 7, 'legend.fontsize': 6.3, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'axes.edgecolor': MUTED, 'axes.linewidth': 0.6,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.labelcolor': INK, 'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': 'white', 'axes.facecolor': 'white'})
x = Q.index.to_timestamp() + pd.Timedelta(days=45); ELEC = ['2017-09-24', '2021-09-26', '2025-02-23']
def dress(ax, title):
    for d in ELEC: ax.axvline(pd.Timestamp(d), color=ORANGE if d.startswith('2017') else MUTED, lw=1.0 if d.startswith('2017') else 0.6, ls='-' if d.startswith('2017') else ':')
    ax.grid(True, axis='y', color=GRID, lw=0.5); ax.set_axisbelow(True); ax.set_title(title, loc='left', fontsize=7.5, pad=4)
fig, ax = plt.subplots(2, 2, figsize=(183 / 25.4, 120 / 25.4)); ax = ax.ravel()
for a, l in zip(ax, 'abcd'): a.text(-0.1, 1.06, l, transform=a.transAxes, fontsize=9, fontweight='bold', va='bottom', ha='right')
# a: composite vs EMI (both z-scored over quarters for display)
z = lambda s: (s - s.mean()) / s.std()
ax[0].plot(x, z(Q.llm_emi), color=BLUE, lw=1.3, label='LLM: evidence minus intuition'); ax[0].plot(x, z(Q[EMI]), color=INK, lw=1.0, ls='--', label='embedding EMI')
ax[0].set_ylabel('quarterly mean, standardised'); ax[0].legend(frameon=False, loc='lower left'); dress(ax[0], 'LLM composite and embedding EMI')
# b: evidence and intuition ratings
ax[1].plot(x, Q.evidenz_basiert, color=GREEN, lw=1.3, label='evidence-based (0-4)'); ax[1].plot(x, Q.intuition_basiert, color=ORANGE, lw=1.3, label='intuition-based (0-4)')
ax[1].set_ylabel('mean rating'); ax[1].legend(frameon=False, loc='center left'); dress(ax[1], 'Evidence-based and intuition-based ratings')
# c: use vs mention
ax[2].plot(x, Q.evidence_use, color=GREEN, lw=1.3, label='evidence used'); ax[2].plot(x, Q.evidence_mention, color=GREEN, lw=1.0, ls=':', label='evidence talked about')
ax[2].plot(x, Q.intuition_use, color=ORANGE, lw=1.3, label='intuition used'); ax[2].plot(x, Q.intuition_mention, color=ORANGE, lw=1.0, ls=':', label='intuition talked about')
ax[2].set_ylabel('mean rating (0-4)'); ax[2].legend(frameon=False, ncol=2, loc='upper left', columnspacing=1.0); dress(ax[2], 'Used vs talked about')
# d: composite by leaning
for lean, c in LC.items():
    q = QL.loc[lean]; ax[3].plot(q.index.to_timestamp() + pd.Timedelta(days=45), q.llm_emi, color=c, lw=1.2, label=lean)
ax[3].set_ylabel('LLM evidence minus intuition'); ax[3].legend(frameon=False, loc='lower left', title='leaning', title_fontsize=6.3); dress(ax[3], 'By leaning')
for a in ax: a.set_xlim(pd.Timestamp('2015-01-01'), pd.Timestamp('2025-09-01'))
fig.tight_layout(); fig.savefig(os.path.join(OUTD, 'timeseries.png'), dpi=300, bbox_inches='tight'); fig.savefig(os.path.join(OUTD, 'timeseries.pdf'), bbox_inches='tight'); print('\nfigure: results/timeseries.png')
