"""Supplementary figure: the 2017 election in the 2000-2025 Bundestag series.

A  180-day rolling mean of EMI 2000-2025 (all speeches, and by camp) with the seven federal elections
B  level shifts at the seven elections from the constrained breakpoint model (random speaker intercepts)
C  single-break search: level-shift estimate of a single breakpoint placed at every quarter, with the
   placebo distribution (quarters at least two quarters away from an election) and the election quarters marked
D  event-time view: the rolling mean around each election relative to its pre-election level (year before)

Inputs: results/revision/02_pre2015/{rolling_all,rolling_by_leaning,breakpoint7_constrained_RI,single_break_grid,
placebo_single_break}.csv (written by revision.ipynb section 2). Output: results/revision/02_pre2015/fig_supp_afd_entry.{pdf,png}
"""
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))   # German-EMI repository root: data in data/, outputs in results/
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(REPO_ROOT, 'results', 'revision', '02_pre2015')
PLOT_LABEL_SIZE, PLOT_LEGEND_SIZE, PLOT_PANEL_SIZE, PLOT_TICK_SIZE = 20, 14, 20, 14
plt.rcParams.update({'font.family': 'Arial', 'axes.spines.top': False, 'axes.spines.right': False})
COLORS = {'left': '#E66101', 'middle': '#4D4D4D', 'right': '#0072B2'}; LAB = {'left': 'Left', 'middle': 'Centre', 'right': 'Right'}
E = {'2002': '2002-09-22', '2005': '2005-09-18', '2009': '2009-09-27', '2013': '2013-09-22', '2017': '2017-09-24', '2021': '2021-09-26', '2025': '2025-02-23'}
E = {k: pd.Timestamp(v) for k, v in E.items()}
HL = '#D55E00'

roll = pd.read_csv(os.path.join(D, 'rolling_all.csv'), parse_dates=['date']).set_index('date')
rl = pd.read_csv(os.path.join(D, 'rolling_by_leaning.csv'), parse_dates=['date'])
bp = pd.read_csv(os.path.join(D, 'breakpoint7_constrained_RI.csv'), index_col=0)
grid = pd.read_csv(os.path.join(D, 'single_break_grid.csv')); plc = pd.read_csv(os.path.join(D, 'placebo_single_break.csv'))
grid['date'] = pd.PeriodIndex([f'{int(q) // 10}Q{int(q) % 10}' for q in grid.quarter], freq='Q').to_timestamp()

fig = plt.figure(figsize=(20, 15))
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.32, wspace=0.22, left=0.06, right=0.98, top=0.95, bottom=0.07)

# ---- A: rolling series ----
ax = fig.add_subplot(gs[0, :])
ax.fill_between(roll.index, roll['mean'] - 1.96 * roll['sem'], roll['mean'] + 1.96 * roll['sem'], color='0.75', alpha=0.5, lw=0)
ax.plot(roll.index, roll['mean'], color='black', lw=2.4, label='All speeches', zorder=4)
for lean in ['middle', 'left', 'right']:
    s = rl[rl.leaning == lean].set_index('date')['mean']
    ax.plot(s.index, s, color=COLORS[lean], lw=1.3, alpha=0.9, label=LAB[lean], zorder=3)
for y, d in E.items():
    ax.axvline(d, color=HL if y == '2017' else '0.4', lw=2.0 if y == '2017' else 1.0, ls='-' if y == '2017' else '--', zorder=2)
    ax.text(d, 1.01, y, transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=PLOT_TICK_SIZE, color=HL if y == '2017' else '0.3', fontweight='bold' if y == '2017' else 'normal')
ax.axhline(0, color='grey', lw=0.8)
ax.set_xlim(pd.Timestamp('2000-01-01'), pd.Timestamp('2025-07-01')); ax.set_ylabel('EMI (180-day rolling mean)', fontsize=PLOT_LABEL_SIZE, fontweight='bold')
ax.tick_params(labelsize=PLOT_TICK_SIZE); ax.grid(axis='y', alpha=0.25)
ax.legend(frameon=False, fontsize=PLOT_LEGEND_SIZE, loc='lower left', ncol=4)
ax.text(-0.04, 1.06, 'A', transform=ax.transAxes, fontsize=PLOT_PANEL_SIZE, fontweight='bold', va='bottom')

# ---- B: forest plot of level shifts ----
axb = fig.add_subplot(gs[1, 0])
lv = bp.loc[[f'level_elec{y}' for y in E]]; yy = np.arange(len(lv))[::-1]
for i, (k, row) in enumerate(lv.iterrows()):
    y = k.replace('level_elec', ''); c = HL if y == '2017' else 'black'
    axb.errorbar(row.coef, yy[i], xerr=[[row.coef - row.lo], [row.hi - row.coef]], fmt='o', color=c, capsize=4, ms=9 if y == '2017' else 7, lw=2)
    axb.text(row.lo - 0.012, yy[i], f'{row.coef:+.2f}', ha='right', va='center', fontsize=PLOT_TICK_SIZE - 1, color=c)
axb.set_yticks(yy); axb.set_yticklabels(list(E), fontsize=PLOT_TICK_SIZE); axb.axvline(0, color='grey', lw=0.8)
axb.set_xlabel('Level shift at the election (EMI, 95% CI)', fontsize=PLOT_LABEL_SIZE - 2, fontweight='bold'); axb.set_ylabel('Federal election', fontsize=PLOT_LABEL_SIZE - 2, fontweight='bold')
axb.tick_params(labelsize=PLOT_TICK_SIZE); axb.grid(axis='x', alpha=0.25); axb.set_xlim(-0.62, 0.12)
axb.text(-0.14, 1.04, 'B', transform=axb.transAxes, fontsize=PLOT_PANEL_SIZE, fontweight='bold', va='bottom')

# ---- C: single-break search and placebo distribution ----
axc = fig.add_subplot(gs[1, 1])
lo, hi = plc.level.quantile(0.05), plc.level.quantile(0.95)
axc.axhspan(plc.level.min(), plc.level.max(), color='0.9', zorder=0, label='Range of 56 placebo breakpoints')
axc.axhspan(lo, hi, color='0.8', zorder=0, label='5th–95th percentile of placebos')
axc.plot(grid.date, grid.level, color='0.35', lw=1.4, marker='o', ms=3, label='Single break placed at this quarter', zorder=3)
for y, d in E.items():
    q = grid[(grid.date.dt.year == d.year) & (grid.date.dt.quarter == d.quarter)]
    if len(q): axc.scatter(q.date, q.level, color=HL if y == '2017' else 'black', s=90 if y == '2017' else 55, zorder=5)
    axc.axvline(d, color=HL if y == '2017' else '0.5', lw=1.4 if y == '2017' else 0.8, ls='-' if y == '2017' else '--', zorder=1)
best = grid.loc[grid.ssr.idxmin()]
axc.annotate(f"best fit: {best.date.year}Q{best.date.quarter}", xy=(best.date, best.level), xytext=(pd.Timestamp('2010-06-01'), -0.45), fontsize=PLOT_LEGEND_SIZE, arrowprops=dict(arrowstyle='->', color=HL), color=HL)
axc.axhline(0, color='grey', lw=0.8)
axc.set_ylabel('Level shift of a single break (EMI)', fontsize=PLOT_LABEL_SIZE - 2, fontweight='bold'); axc.set_xlabel('Quarter of the break', fontsize=PLOT_LABEL_SIZE - 2, fontweight='bold')
axc.tick_params(labelsize=PLOT_TICK_SIZE); axc.legend(frameon=False, fontsize=PLOT_LEGEND_SIZE - 1, loc='lower left'); axc.grid(alpha=0.2)
axc.text(-0.14, 1.04, 'C', transform=axc.transAxes, fontsize=PLOT_PANEL_SIZE, fontweight='bold', va='bottom')

for ext in ('pdf', 'png'):
    out = os.path.join(D, f'fig_supp_afd_entry.{ext}')
    try: fig.savefig(out, dpi=300, bbox_inches='tight')
    except PermissionError: fig.savefig(out.replace(f'.{ext}', f'_tmp.{ext}'), dpi=300, bbox_inches='tight'); print('locked, wrote _tmp')

# ---- separate figure: event-time recovery ----
fig2, ax2 = plt.subplots(figsize=(11, 6.5))
s = roll['mean'].dropna()
for y, d in E.items():
    pre = s[(s.index >= d - pd.Timedelta(days=365)) & (s.index < d)].mean()
    w = s[(s.index >= d - pd.Timedelta(days=730)) & (s.index <= d + pd.Timedelta(days=4 * 365))]
    x = (w.index - d).days / 365.25
    ax2.plot(x, w - pre, color=HL if y == '2017' else ('0.2' if y in ('2021', '2025') else '0.6'), lw=3 if y == '2017' else 1.6, ls='-' if y in ('2017', '2021', '2025') else '--', label=y, zorder=5 if y == '2017' else 3)
ax2.axhline(0, color='black', lw=0.9); ax2.axvline(0, color='black', lw=0.9, ls=':')
ax2.set_xlabel('Years since the election', fontsize=PLOT_LABEL_SIZE - 2, fontweight='bold'); ax2.set_ylabel('EMI relative to the year before the election', fontsize=PLOT_LABEL_SIZE - 2, fontweight='bold')
ax2.tick_params(labelsize=PLOT_TICK_SIZE); ax2.grid(alpha=0.25); ax2.set_xlim(-2, 4)
ax2.legend(frameon=False, fontsize=PLOT_LEGEND_SIZE, title='Election', title_fontsize=PLOT_LEGEND_SIZE, ncol=2, loc='lower left')
fig2.tight_layout()
for ext in ('pdf', 'png'):
    out = os.path.join(D, f'fig_supp_afd_recovery.{ext}')
    try: fig2.savefig(out, dpi=300, bbox_inches='tight')
    except PermissionError: fig2.savefig(out.replace(f'.{ext}', f'_tmp.{ext}'), dpi=300, bbox_inches='tight'); print('locked, wrote _tmp')
print('written', D)
