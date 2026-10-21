"""Four-panel figure: the AfD's entry as a chamber-wide contagion.
A  EMI of all Bundestag speeches 2000-2025 (centred 180-day rolling mean), election breakpoints, recovery before 2017 vs none after
B  Level shift at each of the seven elections (constrained breakpoint model, 95% CI)
C  Within-sitting reaction: change in other parties' EMI from the speech before to the speech after, by the AfD speech's
   own EMI quartile, with the same statistic after intuition-heavy speeches of other parties as placebo
D  Non-AfD speeches that name the AfD vs those that do not: quarterly EMI (top) and the share naming the AfD (strip)
Output: results/revision/figures/fig_contagion.{pdf,png}
"""
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))   # German-EMI repository root: data in data/, outputs in results/
import os, json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import gridspec
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = REPO_ROOT; RES = os.path.join(REPO_ROOT, 'results', 'revision'); OUT = os.path.join(RES, 'figures'); os.makedirs(OUT, exist_ok=True)
EMI = 'emi_w2vparliament'
ORANGE, BLUE, GREEN, INK, MUTED, GRID = '#D55E00', '#0072B2', '#009E73', '#222222', '#6E6E6E', '#E6E6E6'
plt.rcParams.update({'font.family': 'Arial', 'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 8, 'legend.fontsize': 6.5, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
                     'axes.edgecolor': MUTED, 'axes.linewidth': 0.6, 'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.labelcolor': INK, 'text.color': INK,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white'})
def despine_grid(ax):
    ax.grid(True, axis='y', color=GRID, lw=0.5); ax.set_axisbelow(True)
def panel(ax, letter):
    ax.text(-0.08, 1.06, letter, transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom', ha='right')

fig = plt.figure(figsize=(183 / 25.4, 125 / 25.4))
gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1.55, 1], height_ratios=[1, 1], wspace=0.3, hspace=0.62)

# ---------------- A: 2000-2025 rolling series ------------------------------------------------------
roll = pd.read_csv(os.path.join(RES, '02_pre2015', 'rolling_all.csv'), parse_dates=['date']).set_index('date')
ELEC = {2002: '2002-09-22', 2005: '2005-09-18', 2009: '2009-09-27', 2013: '2013-09-22', 2017: '2017-09-24', 2021: '2021-09-26', 2025: '2025-02-23'}
axA = fig.add_subplot(gs[0, 0]); panel(axA, 'a')
axA.fill_between(roll.index, roll['mean'] - 1.96 * roll['sem'], roll['mean'] + 1.96 * roll['sem'], color=BLUE, alpha=0.15, lw=0)
axA.plot(roll.index, roll['mean'], color=BLUE, lw=1.1)
for y, d in ELEC.items():
    axA.axvline(pd.Timestamp(d), color=ORANGE if y == 2017 else MUTED, lw=1.2 if y == 2017 else 0.6, ls='-' if y == 2017 else ':')
    axA.text(pd.Timestamp(d), 0.56, str(y), ha='center', va='bottom', fontsize=6, color=ORANGE if y == 2017 else MUTED, fontweight='bold' if y == 2017 else 'normal')
# pre-election reference levels (mean of the 12 months before each election) and recovery markers
for y, d in ELEC.items():
    if y >= 2025: continue
    dd = pd.Timestamp(d); pre = roll.loc[(roll.index >= dd - pd.Timedelta(days=365)) & (roll.index < dd - pd.Timedelta(days=90)), 'mean'].mean()
    axA.hlines(pre, dd - pd.Timedelta(days=365), dd + pd.Timedelta(days=3 * 365), color=ORANGE if y == 2017 else MUTED, lw=0.6, ls='--', alpha=0.9)
axA.axhline(0, color=MUTED, lw=0.5)
axA.set_ylim(-1.05, 0.65); axA.set_ylabel('EMI, all speeches\n(z over 2000–2025, 180-day rolling mean)'); axA.set_xlabel('')
axA.set_xlim(pd.Timestamp('2000-01-01'), pd.Timestamp('2025-09-01')); despine_grid(axA)
axA.annotate('AfD enters\nthe Bundestag', xy=(pd.Timestamp('2017-10-24'), -0.6), xytext=(pd.Timestamp('2011-06-01'), -0.85), fontsize=6.5, color=ORANGE,
             arrowprops=dict(arrowstyle='-', color=ORANGE, lw=0.6), ha='left', va='center')
axA.text(pd.Timestamp('2000-03-01'), 0.42, 'dashed: level in the year before each election', fontsize=6, color=MUTED, va='bottom')
axA.set_title('Every election is a step down; only after 2017\nthe chamber does not recover', loc='left', fontsize=7.5, pad=6)

# ---------------- B: level shifts at seven elections -----------------------------------------------
t = pd.read_csv(os.path.join(RES, '02_pre2015', 'breakpoint7_constrained_RI.csv'), index_col=0)
lv = t.loc[[f'level_elec{y}' for y in ELEC]]; yy = np.arange(len(lv))[::-1]
axB = fig.add_subplot(gs[0, 1]); panel(axB, 'b')
for i, (idx, r) in enumerate(lv.iterrows()):
    y = int(idx[-4:]); c = ORANGE if y == 2017 else (MUTED if y != 2025 else MUTED)
    axB.plot([r.lo, r.hi], [yy[i], yy[i]], color=c, lw=1.0, solid_capstyle='butt'); axB.plot(r.coef, yy[i], 'o', color=c, ms=4.5, mec='white', mew=0.6)
    axB.text(r.lo - 0.02, yy[i], f'{r.coef:+.2f}', ha='right', va='center', fontsize=6, color=c)
axB.axvline(0, color=MUTED, lw=0.5); axB.set_yticks(yy); axB.set_yticklabels([str(y) for y in ELEC]); axB.set_xlim(-0.62, 0.08)
axB.set_xlabel('level shift in EMI at the election breakpoint\n(constrained model, 95% CI)'); axB.grid(True, axis='x', color=GRID, lw=0.5); axB.set_axisbelow(True)
axB.set_yticklabels([str(y) + ('*' if y == 2025 else '') for y in ELEC]); axB.text(1.0, -0.36, '*sparse post-election data (corpus ends June 2025)', fontsize=5.5, color=MUTED, ha='right', transform=axB.transAxes)
axB.set_title('The 2017 step is the largest of seven\nelections since 2002', loc='left', fontsize=7.5, pad=6)

# ---------------- C: within-sitting dose-response with placebo ---------------------------------------
L = pd.read_csv(os.path.join(RES, '12_afd_sequence', 'local_window_all_speeches.csv'), dtype={'id': str})
sc = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'parliament_nhb.csv'), usecols=['id', EMI, 'party'], dtype={'id': str})
Lm = L.merge(sc[['id', EMI]], on='id').dropna(subset=['diff1'])
A = Lm[Lm.party == 'AfD'].copy(); A['qt'] = pd.qcut(A[EMI], 4, labels=[0, 1, 2, 3])
P = Lm[(Lm.party != 'AfD') & (Lm.party.notna())].copy(); P['qt'] = P.groupby('party')[EMI].transform(lambda s: pd.qcut(s, 4, labels=False, duplicates='drop'))
def ci(g):
    m = g.diff1.mean(); u = g.groupby('sitting').diff1.sum() - m * g.groupby('sitting').size(); return m, 1.96 * np.sqrt((u ** 2).sum()) / len(g)
axC = fig.add_subplot(gs[1, 0]); panel(axC, 'c')
x = np.arange(4); w = 0.36
for k, (df, col, lab, off) in enumerate([(A, ORANGE, 'speech by the AfD', -w / 2 - 0.02), (P, GREEN, 'speech by any other party (placebo)', w / 2 + 0.02)]):
    ms, es = zip(*[ci(df[df.qt == q]) for q in range(4)])
    axC.bar(x + off, ms, width=w, color=col, alpha=0.9, lw=0, label=lab)
    axC.errorbar(x + off, ms, yerr=es, fmt='none', ecolor=INK, elinewidth=0.6, capsize=1.5)
    for xi, m, e in zip(x + off, ms, es): axC.text(xi, m + (e + 0.01 if m >= 0 else -(e + 0.01)), f'{m:+.2f}', ha='center', va='bottom' if m >= 0 else 'top', fontsize=5.8, color=INK)
axC.axhline(0, color=MUTED, lw=0.5); axC.set_xticks(x); axC.set_xticklabels(['most intuition-\nbased quartile', '2nd', '3rd', 'most evidence-\nbased quartile'])
axC.set_xlabel("EMI of the preceding speech (quartiles within the speaking party)"); axC.set_ylabel('change in other parties\' EMI:\nspeech after minus speech before')
axC.legend(frameon=False, loc='upper left', handlelength=1.2); despine_grid(axC); axC.set_ylim(-0.36, 0.25)
axC.set_title('Within a sitting, others answer an intuition-based AfD speech\nin kind; after any other party they rebound', loc='left', fontsize=7.5, pad=6)

# ---------------- D: speeches naming the AfD vs not, and the share naming it ----------------------------
qq = pd.read_csv(os.path.join(RES, '12_afd_sequence', 'confrontation_quarterly.csv'), index_col=0); qq.index = pd.PeriodIndex(qq.index, freq='Q').to_timestamp()
gsD = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 1], height_ratios=[3, 1.1], hspace=0.12)
axD = fig.add_subplot(gsD[0]); panel(axD, 'd'); axD2 = fig.add_subplot(gsD[1], sharex=axD)
axD.plot(qq.index, qq.emi_no_mention, color=BLUE, lw=1.1, label='not naming the AfD'); qm = qq[qq.index >= '2017-10-01']; axD.plot(qm.index, qm.emi_mention, color=ORANGE, lw=1.1, label='naming the AfD (from the entry)')
axD.axhline(0, color=MUTED, lw=0.5); axD.axvline(pd.Timestamp('2017-10-24'), color=MUTED, lw=0.6, ls=':'); axD.set_ylabel('quarterly EMI,\nnon-AfD speeches'); axD.legend(frameon=False, loc='upper right', handlelength=1.2, ncol=1, fontsize=6)
despine_grid(axD); plt.setp(axD.get_xticklabels(), visible=False); axD.set_ylim(-1.6, 1.45)
axD2.fill_between(qq.index, 0, qq.share_mention, color=ORANGE, alpha=0.35, lw=0); axD2.plot(qq.index, qq.share_mention, color=ORANGE, lw=0.8)
axD2.set_ylim(0, 0.45); axD2.set_yticks([0, 0.2, 0.4]); axD2.set_yticklabels(['0', '20%', '40%']); axD2.set_ylabel('share naming\nthe AfD'); axD2.axvline(pd.Timestamp('2017-10-24'), color=MUTED, lw=0.6, ls=':'); despine_grid(axD2)
axD2.set_xlim(pd.Timestamp('2015-01-01'), pd.Timestamp('2025-09-01')); axD2.set_xticks([pd.Timestamp(f'{y}-01-01') for y in range(2015, 2026, 2)]); axD2.set_xticklabels([str(y) for y in range(2015, 2026, 2)])
num = json.load(open(os.path.join(RES, '12_afd_sequence', 'numbers.json')))
axD.set_title('Naming the AfD lowers EMI by 0.5; the 2017 step is\n−0.44 overall and −0.35 in speeches that never name it', loc='left', fontsize=7.5, pad=6)

fig.savefig(os.path.join(OUT, 'fig_contagion.pdf'), bbox_inches='tight'); fig.savefig(os.path.join(OUT, 'fig_contagion.png'), dpi=300, bbox_inches='tight')
print('saved', os.path.join(OUT, 'fig_contagion.png'))
