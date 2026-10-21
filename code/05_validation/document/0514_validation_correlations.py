"""Spearman correlations between computed scores and survey items, pooled and by era / decade / party.
Input : results/revision/07_validation_auc/validation_instances.csv (built by revision.ipynb section 7)
Output: results/revision/07_validation_auc/corr_{pooled,by_era,by_decade,by_party}.csv + corr_by_group.png
"""
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))   # German-EMI repository root: data in data/, outputs in results/
import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(REPO_ROOT, 'results', 'revision', '07_validation_auc')
v = pd.read_csv(os.path.join(OUT, 'validation_instances.csv'))
v['survey_emi'] = v.ev_mean - v.iu_mean
v['party_merged'] = v.party.replace({'DIE LINKE': 'DIE LINKE / PDS', 'PDS': 'DIE LINKE / PDS'})
ERA_ORDER = ['Kaiserreich 1867–1918', 'Weimar 1919–1932', 'NS 1933–1945', 'Bonn 1949–1990', 'Berlin 1991–2014', '2015–2025']
PAIRS = [('evidence', 'evidence_z', 'ev_mean'), ('intuition', 'intuition_z', 'iu_mean'), ('EMI', 'emi', 'survey_emi')]

def rho_ci(x, y, n_boot=1000, seed=0):
    m = ~(np.isnan(x) | np.isnan(y)); x, y = np.asarray(x)[m], np.asarray(y)[m]
    if len(x) < 10: return np.nan, np.nan, np.nan, len(x)
    r = spearmanr(x, y).correlation; g = np.random.default_rng(seed)
    bs = [spearmanr(x[i], y[i]).correlation for i in (g.integers(0, len(x), len(x)) for _ in range(n_boot))]
    return r, float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5)), len(x)

def table(df, group, min_n=20):
    rows = []
    for g, d in ([('all', df)] if group is None else df.groupby(group, observed=True)):
        if len(d) < min_n: continue
        for name, sc, sv in PAIRS:
            r, lo, hi, n = rho_ci(d[sc], d[sv]); rows.append(dict(group=str(g), construct=name, n=n, rho=r, lo=lo, hi=hi))
    return pd.DataFrame(rows)

pooled = table(v, None); by_era = table(v, 'era'); by_decade = table(v, 'decade'); by_party = table(v, 'party_merged')
for nm, t in [('pooled', pooled), ('by_era', by_era), ('by_decade', by_decade), ('by_party', by_party)]:
    t.to_csv(os.path.join(OUT, f'corr_{nm}.csv'), index=False)
print(pooled.round(3).to_string(index=False)); print()
print(by_era.pivot(index='group', columns='construct', values='rho').reindex(ERA_ORDER).round(2).to_string()); print()
print(by_party.pivot(index='group', columns='construct', values='rho').round(2).join(by_party[by_party.construct == 'EMI'].set_index('group').n).to_string()); print()
print(by_decade[by_decade.construct == 'EMI'][['group', 'n', 'rho', 'lo', 'hi']].round(2).to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={'width_ratios': [1, 1, 1.2]})
cols = {'evidence': '#0072B2', 'intuition': '#E66101', 'EMI': 'black'}
for ax, (title, t, order) in zip(axes, [('By era', by_era, ERA_ORDER),
                                        ('By party (n ≥ 20 chunks)', by_party, by_party[by_party.construct == 'EMI'].sort_values('n', ascending=False).group.tolist()),
                                        ('By decade', by_decade, sorted(by_decade.group.unique(), key=float))]):
    order = [o for o in order if o in set(t.group)]; yy = np.arange(len(order))[::-1]
    for k, (name, _, _) in enumerate(PAIRS):
        s = t[t.construct == name].set_index('group').reindex(order)
        ax.errorbar(s.rho, yy + (1 - k) * 0.22, xerr=[s.rho - s.lo, s.hi - s.rho], fmt='o', color=cols[name], capsize=2, ms=4, label=name)
    ax.axvline(0, color='grey', lw=0.8); ax.set_yticks(yy)
    ax.set_yticklabels([f'{g} (n={int(t[(t.group == g) & (t.construct == "EMI")].n.iloc[0])})' for g in order])
    ax.set_xlim(-0.4, 0.9); ax.set_xlabel('Spearman ρ, computed score vs survey item (95% bootstrap CI)'); ax.set_title(title); ax.grid(alpha=0.3)
axes[0].legend(frameon=False, loc='lower right')
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'corr_by_group.png'), dpi=150, bbox_inches='tight'); plt.savefig(os.path.join(OUT, 'corr_by_group.pdf'), bbox_inches='tight')
print('saved')
