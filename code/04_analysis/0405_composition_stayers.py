"""Robustness check: is the parliamentary EMI decline compositional or within-person?

Restrict the Bundestag corpus to politicians who are present across electoral
terms ("stayers") and re-run (i) the early/late comparison, (ii) a within-
politician paired change, (iii) the paper's breakpoint mixed models (Eq. 2/3)
and (iv) a politician fixed-effects breakpoint model. Two panels:
  A: stayers WP18-WP20 (speak in 2015-17, 2017-21 and 2021-25; no AfD possible)
  B: stayers WP19-WP20 (speak in 2017-21 and 2021-25; includes AfD)
Outputs: results/parliament/stayers/*.csv, stayers_figure.pdf, and numbers.json
"""
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))   # German-EMI repository root: data in data/, outputs in results/
import os, json, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM
warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
SPLIT = os.environ.get('STAYERS_SPLIT', '2020-01-01')   # early/late split; '2017-10-24' = the AfD's entry
SPLIT_LABEL = os.environ.get('STAYERS_LABEL', '2020-25 minus 2015-19')
OUT = os.path.join(REPO_ROOT, 'results/parliament/stayers' + ('' if SPLIT == '2020-01-01' else '_afd')); os.makedirs(OUT, exist_ok=True)
print('early/late split at', SPLIT, '->', OUT)
EMI = 'emi_w2vparliament'
BP = {'elec2017': pd.Timestamp('2017-07-01', tz='UTC'), 'elec2021': pd.Timestamp('2021-07-01', tz='UTC'),
      'elec2025': pd.Timestamp('2025-01-01', tz='UTC')}
COLORS = {'left': '#E66101', 'middle': '#4D4D4D', 'right': '#0072B2'}
LAB = {'left': 'Left', 'middle': 'Centre', 'right': 'Right'}

df = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'parliament_nhb.csv'),
                 usecols=['politicianId', 'date', 'electoralTerm', 'party', 'leaning', EMI, 'firstName', 'lastName'])
df = df.dropna(subset=['politicianId', 'party', 'leaning', EMI, 'date'])
df['politicianId'] = df['politicianId'].astype(int)
df['date'] = pd.to_datetime(df['date'], utc=True)
df['time_quarterly'] = (df['date'].dt.year * 10 + df['date'].dt.quarter).astype(int)
df['late'] = (df['date'] >= SPLIT).astype(int)
df['year'] = df['date'].dt.year
NUM = {}
NUM['n_speeches_all'] = int(len(df)); NUM['n_pol_all'] = int(df.politicianId.nunique())

terms = df.groupby('politicianId')['electoralTerm'].agg(lambda s: set(s))
panelA = set(terms[terms.apply(lambda s: {18, 19, 20} <= s)].index)
panelB = set(terms[terms.apply(lambda s: {19, 20} <= s)].index)
# early/late presence (for the composition decomposition)
pres = df.groupby('politicianId')['late'].agg(['min', 'max'])
both = set(pres[(pres['min'] == 0) & (pres['max'] == 1)].index)
only_early = set(pres[pres['max'] == 0].index); only_late = set(pres[pres['min'] == 1].index)
NUM['n_pol_both_periods'] = len(both); NUM['n_pol_only_early'] = len(only_early); NUM['n_pol_only_late'] = len(only_late)
NUM['n_speeches_only_late'] = int(df.politicianId.isin(only_late).sum())
NUM['n_speeches_only_early'] = int(df.politicianId.isin(only_early).sum())
NUM['share_late_speeches_by_entrants'] = float(df[df.late == 1].politicianId.isin(only_late).mean())
NUM['share_early_speeches_by_leavers'] = float(df[df.late == 0].politicianId.isin(only_early).mean())


def boot_ci(x, n=2000, seed=0):
    r = np.random.default_rng(seed); m = [r.choice(x, len(x)).mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def early_late(d):
    e, l = d.loc[d.late == 0, EMI].values, d.loc[d.late == 1, EMI].values
    t, p = stats.ttest_ind(e, l, equal_var=False)
    sp = np.sqrt(((len(e) - 1) * e.var(ddof=1) + (len(l) - 1) * l.var(ddof=1)) / (len(e) + len(l) - 2))
    return dict(n_early=len(e), n_late=len(l), mean_early=e.mean(), mean_late=l.mean(),
                ci_early=boot_ci(e), ci_late=boot_ci(l), t=t, p=p, d=(l.mean() - e.mean()) / sp,
                mw_p=stats.mannwhitneyu(e, l).pvalue)


rows = []
for name, ids in [('all', None), ('both_periods', both), ('panelA_WP18-20', panelA), ('panelB_WP19-20', panelB)]:
    d = df if ids is None else df[df.politicianId.isin(ids)]
    for lean in ['all', 'middle', 'left', 'right']:
        dd = d if lean == 'all' else d[d.leaning == lean]
        if dd.late.nunique() < 2 or (dd.late == 0).sum() < 30:
            continue
        r = early_late(dd); r.update(sample=name, leaning=lean, n_pol=dd.politicianId.nunique()); rows.append(r)
el = pd.DataFrame(rows); el.to_csv(os.path.join(OUT, 'early_late_by_sample.csv'), index=False)
print(el[['sample', 'leaning', 'n_pol', 'n_early', 'n_late', 'mean_early', 'mean_late', 'd', 'p']].round(3).to_string())

# ---- composition decomposition of the raw early->late change ------------------------------
tot = df.loc[df.late == 1, EMI].mean() - df.loc[df.late == 0, EMI].mean()
# counterfactual late mean if only stayers (both-period politicians) were in the chamber
stay_late = df[(df.late == 1) & df.politicianId.isin(both)][EMI].mean()
stay_early = df[(df.late == 0) & df.politicianId.isin(both)][EMI].mean()
within = stay_late - stay_early
NUM['decomp'] = dict(total=float(tot), within_stayers=float(within), stayers_share=float(within / tot),
                     stay_early=float(stay_early), stay_late=float(stay_late),
                     entrants_late=float(df[(df.late == 1) & df.politicianId.isin(only_late)][EMI].mean()),
                     leavers_early=float(df[(df.late == 0) & df.politicianId.isin(only_early)][EMI].mean()))

# ---- within-politician paired change (both-period politicians, >=5 speeches per period) -----
g = df[df.politicianId.isin(both)].groupby(['politicianId', 'late'])[EMI].agg(['mean', 'size']).unstack()
g = g[(g['size'][0] >= 5) & (g['size'][1] >= 5)]
chg = (g['mean'][1] - g['mean'][0]).rename('change').to_frame()
chg['leaning'] = df.groupby('politicianId')['leaning'].first().reindex(chg.index)
chg['party'] = df.groupby('politicianId')['party'].first().reindex(chg.index)
chg.to_csv(os.path.join(OUT, 'within_politician_change.csv'))
tt = stats.ttest_1samp(chg.change, 0)
NUM['paired'] = dict(n=int(len(chg)), mean_change=float(chg.change.mean()), ci=boot_ci(chg.change.values),
                     t=float(tt.statistic), p=float(tt.pvalue), share_declining=float((chg.change < 0).mean()),
                     wilcoxon_p=float(stats.wilcoxon(chg.change).pvalue), median=float(chg.change.median()))
by = chg.groupby('leaning').change.agg(['size', 'mean', 'median', lambda s: (s < 0).mean()])
by.columns = ['n', 'mean', 'median', 'share_declining']
by['p'] = [stats.ttest_1samp(chg.change[chg.leaning == l], 0).pvalue for l in by.index]
byp = chg.groupby('party').change.agg(['size', 'mean', 'median', lambda s: (s < 0).mean()])
byp.columns = ['n', 'mean', 'median', 'share_declining']
byp['p'] = [stats.ttest_1samp(chg.change[chg.party == l], 0).pvalue for l in byp.index]
by.to_csv(os.path.join(OUT, 'within_change_by_leaning.csv')); byp.to_csv(os.path.join(OUT, 'within_change_by_party.csv'))
print(NUM['paired']); print(by.round(3)); print(byp.round(3))


# ---- breakpoint models (mirror fit_breakpoint_model / _full in analysis.ipynb) ---------------
def prep(d):
    m = d[[EMI, 'time_quarterly', 'leaning', 'politicianId']].copy()
    q = m.groupby(['politicianId', 'time_quarterly'])[EMI].mean().rename('_q').reset_index()
    q['emi_lag1'] = q.groupby('politicianId')['_q'].shift(1)
    m = m.merge(q[['politicianId', 'time_quarterly', 'emi_lag1']], on=['politicianId', 'time_quarterly']).dropna()
    tm = m.time_quarterly.mean(); m['time_centered'] = m.time_quarterly - tm
    for k, ts in BP.items():
        bq = int(f'{ts.year}{ts.quarter}'); m[f'slope_{k}'] = np.maximum(0, m.time_centered - (bq - tm))
        m[f'level_{k}'] = (m.time_quarterly >= bq).astype(float)
    cats = [c for c in ['middle', 'left', 'right'] if c in set(m.leaning)]
    m['leaning'] = pd.Categorical(m.leaning, categories=cats)
    return m.reset_index(drop=True), cats


def formula(cats, full, fe=False, skip_bp=()):
    t = ['time_centered'] + [f'slope_{k}' for k in BP] + [f'level_{k}' for k in BP]
    if len(cats) > 1:
        if not fe: t.append('leaning')
        t.append('time_centered:leaning')
        if full: t += [f'slope_{k}:leaning' for k in BP if k not in skip_bp] + [f'level_{k}:leaning' for k in BP if k not in skip_bp]
    t.append('emi_lag1')
    return f'{EMI} ~ ' + ' + '.join(t)


def tidy(res, tag, n, npol, extra=None):
    ci = res.conf_int()
    t = pd.DataFrame({'coef': res.params, 'se': res.bse, 'p': res.pvalues, 'lo': ci[0], 'hi': ci[1]})
    t['model'] = tag; t['n'] = n; t['n_pol'] = npol
    for k, v in (extra or {}).items(): t[k] = v
    return t


fits = []
for pname, ids in [('all', None), ('panelA', panelA), ('panelB', panelB)]:
    d = df if ids is None else df[df.politicianId.isin(ids)]
    m, cats = prep(d)
    # a camp with no observations before a breakpoint makes that breakpoint's leaning
    # interactions collinear with the camp main effect -> drop them (AfD enters 2017)
    skip = tuple(k for k, ts in BP.items()
                 if (m.groupby('leaning', observed=True).time_quarterly.min() >= int(f'{ts.year}{ts.quarter}')).any())
    if skip: print(pname, 'dropping leaning interactions for', skip, '(no pre-breakpoint data for a camp)')
    NUM.setdefault('skipped_bp_interactions', {})[pname] = list(skip)
    for full in (False, True):
        f = formula(cats, full, skip_bp=skip)
        r = MixedLM.from_formula(f, groups=m.politicianId, data=m).fit(reml=False)
        fits.append(tidy(r, f"{pname}_{'full' if full else 'constrained'}_RI", len(m), m.politicianId.nunique(),
                         dict(group_var=r.cov_re.iloc[0, 0], scale=r.scale, llf=r.llf)))
        print(pname, 'full' if full else 'constrained', 'N', len(m), 'pol', m.politicianId.nunique(),
              'slope2021', round(r.params['slope_elec2021'], 4), 'level2017', round(r.params['level_elec2017'], 3))
    # politician fixed effects (within estimator), cluster-robust by politician
    for full in (False, True):
        f = formula(cats, full, fe=True, skip_bp=skip) + ' + C(politicianId)'
        r = smf.ols(f, data=m).fit(cov_type='cluster', cov_kwds={'groups': m.politicianId})
        t = tidy(r, f"{pname}_{'full' if full else 'constrained'}_FE", len(m), m.politicianId.nunique(), dict(r2_within=r.rsquared))
        fits.append(t[~t.index.str.startswith('C(politicianId)')])
        print(pname, 'FE', 'full' if full else 'constrained', 'slope2021', round(r.params['slope_elec2021'], 4),
              'level2017', round(r.params['level_elec2017'], 3), 'level2021', round(r.params['level_elec2021'], 3))
models = pd.concat(fits); models.index.name = 'term'; models.to_csv(os.path.join(OUT, 'breakpoint_models.csv'))

# ---- figure -----------------------------------------------------------------------------------
plt.rcParams.update({'font.size': 13, 'axes.labelsize': 15, 'legend.fontsize': 12, 'figure.facecolor': 'white',
                     'axes.facecolor': 'white', 'text.color': 'black', 'axes.labelcolor': 'black',
                     'xtick.color': 'black', 'ytick.color': 'black', 'axes.edgecolor': 'black'})
fig, axes = plt.subplots(1, 3, figsize=(20, 5.6), gridspec_kw={'width_ratios': [1.5, 1.5, 1]})
df['qdate'] = df['date'].dt.to_period('Q').dt.to_timestamp()
ax = axes[0]
for name, ids, col, ls in [('All speeches', None, 'black', '-'), ('Stayers WP18–20', panelA, '#117733', '-'),
                            ('Stayers WP19–20', panelB, '#AA4499', '--')]:
    d = df if ids is None else df[df.politicianId.isin(ids)]
    q = d.groupby('qdate')[EMI].agg(['mean', 'sem'])
    ax.plot(q.index, q['mean'], color=col, ls=ls, lw=2, label=name)
    ax.fill_between(q.index, q['mean'] - 1.96 * q['sem'], q['mean'] + 1.96 * q['sem'], color=col, alpha=0.12)
for ts in BP.values(): ax.axvline(ts.tz_localize(None), color='grey', ls=':', lw=1)
ax.axhline(0, color='grey', lw=0.8); ax.set_ylabel('EMI score'); ax.set_xlabel('Date'); ax.legend(frameon=False)
ax.text(0.01, 0.97, 'A', transform=ax.transAxes, fontsize=18, fontweight='bold', va='top')
ax = axes[1]
dB = df[df.politicianId.isin(panelB)]
for lean in ['middle', 'left', 'right']:
    q = dB[dB.leaning == lean].groupby('qdate')[EMI].agg(['mean', 'sem'])
    ax.plot(q.index, q['mean'], color=COLORS[lean], lw=2, label=f'{LAB[lean]} (stayers WP19–20)')
    ax.fill_between(q.index, q['mean'] - 1.96 * q['sem'], q['mean'] + 1.96 * q['sem'], color=COLORS[lean], alpha=0.12)
for ts in BP.values(): ax.axvline(ts.tz_localize(None), color='grey', ls=':', lw=1)
ax.axhline(0, color='grey', lw=0.8); ax.set_xlabel('Date'); ax.legend(frameon=False)
ax.text(0.01, 0.97, 'B', transform=ax.transAxes, fontsize=18, fontweight='bold', va='top')
ax = axes[2]
order = [p for p in ['CDU/CSU', 'SPD', 'FDP', 'Alliance 90/The Greens', 'The Left Party', 'AFD'] if p in set(chg.party)]
pc = {'CDU/CSU': '#000000', 'SPD': '#E3000F', 'FDP': '#FFED00', 'Alliance 90/The Greens': '#46962B',
      'The Left Party': '#BE3075', 'AFD': '#009EE0'}
data = [chg.change[chg.party == p].values for p in order]
bp = ax.boxplot(data, vert=False, patch_artist=True, showfliers=False, widths=0.6)
for patch, p in zip(bp['boxes'], order): patch.set_facecolor(pc[p]); patch.set_alpha(0.55)
for med in bp['medians']: med.set_color('black')
ax.set_yticks(range(1, len(order) + 1)); ax.set_yticklabels([p.replace('Alliance 90/The Greens', 'Greens').replace('The Left Party', 'Left') for p in order])
ax.axvline(0, color='grey', lw=0.8); ax.set_xlabel('Within-politician change in mean EMI\n(' + SPLIT_LABEL + ')')
ax.text(0.01, 0.97, 'C', transform=ax.transAxes, fontsize=18, fontweight='bold', va='top')
for a in axes: a.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'stayers_figure.pdf'), bbox_inches='tight'); plt.savefig(os.path.join(OUT, 'stayers_figure.png'), dpi=150, bbox_inches='tight')

NUM['panelA'] = dict(n_pol=len(panelA), n_speeches=int(df.politicianId.isin(panelA).sum()),
                     leanings=df[df.politicianId.isin(panelA)].groupby('leaning').politicianId.nunique().to_dict())
NUM['panelB'] = dict(n_pol=len(panelB), n_speeches=int(df.politicianId.isin(panelB).sum()),
                     leanings=df[df.politicianId.isin(panelB)].groupby('leaning').politicianId.nunique().to_dict())
json.dump(NUM, open(os.path.join(OUT, 'numbers.json'), 'w'), indent=1, default=float)
print(json.dumps(NUM, indent=1, default=float))
