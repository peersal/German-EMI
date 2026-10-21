"""Figure: EMI of the discourse before and after AfD speeches within sittings.
a  EMI of all speeches 2000-2025 (180-day rolling mean) with the level in the year before each election
b  speaker-transition network between parties; edge colour = reply EMI after minus before the preceding speech
c  other parties' EMI by distance (in speeches) from an AfD / SPD / CDU-CSU speech
Output: results/revision/figures/fig_sequence.{pdf,png}
"""
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))   # German-EMI repository root: data in data/, outputs in results/
import os, json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import gridspec, colors as mcolors
from matplotlib.patches import FancyArrowPatch, Circle
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = REPO_ROOT; RES = os.path.join(REPO_ROOT, 'results', 'revision'); OUT = os.path.join(RES, 'figures'); R12 = os.path.join(RES, '12_afd_sequence'); os.makedirs(OUT, exist_ok=True)
EMI = 'emi_w2vparliament'
S = json.load(open(os.path.join(R12, 'sequence_stats.json')))
INK, MUTED, GRID, ORANGE, TEAL = '#222222', '#6E6E6E', '#E6E6E6', '#D55E00', '#009E73'
PC = {'CDU/CSU': '#1A1A1A', 'SPD': '#E3000F', 'FDP': '#C9A800', 'BÜNDNIS 90/DIE GRÜNEN': '#46962B', 'DIE LINKE': '#BE3075', 'AfD': '#009EE0'}
SHORT = {'CDU/CSU': 'CDU/CSU', 'SPD': 'SPD', 'FDP': 'FDP', 'BÜNDNIS 90/DIE GRÜNEN': 'Greens', 'DIE LINKE': 'Left', 'AfD': 'AfD'}
plt.rcParams.update({'font.family': 'Arial', 'font.size': 7, 'axes.labelsize': 7, 'legend.fontsize': 6.3, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'axes.edgecolor': MUTED, 'axes.linewidth': 0.6,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.labelcolor': INK, 'text.color': INK, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': 'white'})
def panel(ax, letter, dx=-0.06, dy=1.04): ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom', ha='right')
def ygrid(ax): ax.grid(True, axis='y', color=GRID, lw=0.5); ax.set_axisbelow(True)
pfmt = lambda p: 'p < 0.001' if p < 0.001 else f'p = {p:.3f}'

# ---------------------------------------------------------------- reply-party-specific local effect (edges of b)
raw = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'revision', 'bundestag_speeches_2000_2025_raw.csv'), usecols=['Source_ID', 'Date', 'Sitting', 'Party', 'Role'], dtype=str)
raw = raw[(raw.Date >= '2017-10-24') & (raw.Role == 'Regular')]
sc = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'parliament_nhb.csv'), usecols=['id', EMI], dtype={'id': str})
d = raw.merge(sc, left_on='Source_ID', right_on='id'); d['order'] = pd.to_numeric(d.Source_ID.str.extract(r'ID(\d+)')[0], errors='coerce'); d = d.sort_values(['Sitting', 'order']).reset_index(drop=True)
PARTIES = list(PC); rows = []
for sit, g in d.groupby('Sitting', sort=False):
    party = g.Party.values; emi = g[EMI].values; n = len(g)
    for i in range(n):
        X = party[i]
        if X not in PARTIES: continue
        before = [j for j in range(i - 1, -1, -1) if party[j] != X][:1]; after = [j for j in range(i + 1, n) if party[j] != X][:1]
        if before and after and party[after[0]] in PARTIES: rows.append(dict(src=X, dst=party[after[0]], diff=emi[after[0]] - emi[before[0]]))
E = pd.DataFrame(rows); edges = E.groupby(['src', 'dst']).agg(n=('diff', 'size'), diff=('diff', 'mean'), se=('diff', lambda s: s.std(ddof=1) / np.sqrt(len(s)))).reset_index()
edges.to_csv(os.path.join(R12, 'network_edges_reply_effect.csv'), index=False); node_n = d[d.Party.isin(PARTIES)].groupby('Party').size()

fig = plt.figure(figsize=(183 / 25.4, 150 / 25.4))
gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[0.9, 1.3], width_ratios=[1.0, 1.15], hspace=0.45, wspace=0.28)

# ---------------------------------------------------------------- a: 2000-2025 series with pre-election baselines
BLUE = '#0072B2'
roll = pd.read_csv(os.path.join(RES, '02_pre2015', 'rolling_all.csv'), parse_dates=['date']).set_index('date')
ELEC = {2002: '2002-09-22', 2005: '2005-09-18', 2009: '2009-09-27', 2013: '2013-09-22', 2017: '2017-09-24', 2021: '2021-09-26', 2025: '2025-02-23'}
axA = fig.add_subplot(gs[0, :]); panel(axA, 'a', dx=-0.035, dy=1.04)
axA.fill_between(roll.index, roll['mean'] - 1.96 * roll['sem'], roll['mean'] + 1.96 * roll['sem'], color=BLUE, alpha=0.15, lw=0)
axA.plot(roll.index, roll['mean'], color=BLUE, lw=1.1, label='all speeches, 180-day mean')
for y, d in ELEC.items():
    c = ORANGE if y == 2017 else MUTED
    axA.axvline(pd.Timestamp(d), color=c, lw=1.2 if y == 2017 else 0.6, ls='-' if y == 2017 else ':')
    axA.text(pd.Timestamp(d), 0.56, str(y), ha='center', va='bottom', fontsize=6, color=c, fontweight='bold' if y == 2017 else 'normal')
    dd = pd.Timestamp(d); pre = roll.loc[(roll.index >= dd - pd.Timedelta(days=365)) & (roll.index < dd - pd.Timedelta(days=90)), 'mean'].mean()
    prev = pd.Timestamp('2000-01-01') if y == 2002 else pd.Timestamp(list(ELEC.values())[list(ELEC).index(y) - 1])
    axA.hlines(pre, prev, dd, color=c, lw=0.6, ls='--', alpha=0.9, label='pre-election level' if y == 2002 else None)
axA.axhline(0, color=MUTED, lw=0.5); ygrid(axA)
axA.set_ylim(-1.05, 0.65); axA.set_ylabel('EMI, all speeches'); axA.set_xlabel('')
axA.set_xlim(pd.Timestamp('2000-01-01'), pd.Timestamp('2025-09-01'))
axA.legend(frameon=False, loc='lower left', ncol=2, handlelength=1.8)
axA.annotate('AfD enters', xy=(pd.Timestamp('2017-10-24'), -0.6), xytext=(pd.Timestamp('2014-03-01'), -0.85), fontsize=6.5, color=ORANGE, arrowprops=dict(arrowstyle='-', color=ORANGE, lw=0.6), ha='left', va='center')
axA.set_title('Recovery after every election, none after 2017', loc='left', fontsize=7.5, pad=6)

# ---------------------------------------------------------------- b: transition network
axB = fig.add_subplot(gs[1, 0]); panel(axB, 'b', dx=-0.04, dy=1.07); axB.set_aspect('equal'); axB.axis('off')
order = ['AfD', 'CDU/CSU', 'FDP', 'SPD', 'BÜNDNIS 90/DIE GRÜNEN', 'DIE LINKE']; ang = {p: np.pi / 2 - 2 * np.pi * i / 6 for i, p in enumerate(order)}
pos = {p: (np.cos(a), np.sin(a)) for p, a in ang.items()}
cmap = mcolors.LinearSegmentedColormap.from_list('div', [ORANGE, '#C8C8C8', TEAL]); norm = mcolors.Normalize(-0.2, 0.2)
nmax = node_n.max(); rad = {p: 0.09 + 0.15 * np.sqrt(node_n[p] / nmax) for p in order}
eb = edges[(edges.src.isin(order)) & (edges.dst.isin(order)) & (edges.n >= 120)].copy(); wmax = eb.n.max()
for r in sorted(eb.itertuples(), key=lambda r: (r.src == 'AfD', r.n)):
    (x0, y0), (x1, y1) = pos[r.src], pos[r.dst]; vx, vy = x1 - x0, y1 - y0; L_ = np.hypot(vx, vy); ux, uy = vx / L_, vy / L_
    start = (x0 + ux * rad[r.src] * 1.05, y0 + uy * rad[r.src] * 1.05); end = (x1 - ux * rad[r.dst] * 1.08, y1 - uy * rad[r.dst] * 1.08)
    lw = 0.5 + 3.0 * np.sqrt(r.n / wmax); col = cmap(norm(r.diff)); alpha = 0.95 if r.src == 'AfD' else 0.5
    axB.add_patch(FancyArrowPatch(start, end, connectionstyle='arc3,rad=0.22', arrowstyle='-|>', mutation_scale=6, lw=lw, color=col, alpha=alpha, zorder=3 if r.src == 'AfD' else 2, shrinkA=0, shrinkB=0))
for p in order:
    axB.add_patch(Circle(pos[p], rad[p], facecolor=PC[p], edgecolor='white', lw=1.0, zorder=4))
    tx, ty = pos[p][0] * (1 + rad[p] + 0.26), pos[p][1] * (1 + rad[p] + 0.26)
    axB.text(tx, ty, SHORT[p], ha='center', va='center', fontsize=6.5, color=INK, zorder=5)
for r in eb[eb.src == 'AfD'].itertuples():
    (x0, y0), (x1, y1) = pos['AfD'], pos[r.dst]; mx, my = (x0 + x1) / 2, (y0 + y1) / 2; vx, vy = x1 - x0, y1 - y0; L_ = np.hypot(vx, vy); nx_, ny_ = -vy / L_, vx / L_
    axB.text(mx + nx_ * 0.18, my + ny_ * 0.18, f'{r.diff:+.2f}', fontsize=6, color=cmap(norm(r.diff)), ha='center', va='center', fontweight='bold', zorder=6, bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.9))
axB.set_xlim(-2.0, 2.0); axB.set_ylim(-1.9, 1.9)
sm_ = plt.cm.ScalarMappable(cmap=cmap, norm=norm); cb = fig.colorbar(sm_, ax=axB, orientation='horizontal', fraction=0.04, pad=0.0, aspect=30, shrink=0.8)
cb.set_label('reply EMI, after minus before', fontsize=6.5); cb.ax.tick_params(labelsize=6)
axB.set_title('Who replies to whom', loc='left', fontsize=7.5, pad=6)

# ---------------------------------------------------------------- c: distance profile
axC = fig.add_subplot(gs[1, 1]); panel(axC, 'c', dx=-0.1, dy=1.04)
offs = [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6]
for X, col, lab, z in [('CDU/CSU', PC['CDU/CSU'], 'CDU/CSU', 1), ('SPD', PC['SPD'], 'SPD', 1), ('AfD', PC['AfD'], 'AfD', 3)]:
    pr = S['profile'][X]; m = np.array([pr[str(o)]['mean'] for o in offs]); se = np.array([pr[str(o)]['se'] for o in offs]); x = np.array(offs, float)
    axC.fill_between(x, m - 1.96 * se, m + 1.96 * se, color=col, alpha=0.12 if X == 'AfD' else 0.07, lw=0)
    axC.plot(x, m, 'o-', color=col, ms=3.2 if X == 'AfD' else 2.4, lw=1.4 if X == 'AfD' else 0.9, label=lab, zorder=z, alpha=1 if X == 'AfD' else 0.85)
axC.axvspan(-0.5, 0.5, color='#DDDDDD', alpha=0.6, lw=0); axC.text(0, 0.05, 'speech', ha='center', va='top', fontsize=6, color=MUTED, rotation=90); axC.set_ylim(-0.26, 0.06)
axC.axhline(0, color=MUTED, lw=0.5); ygrid(axC); axC.set_xticks(offs); axC.set_xticklabels([str(o) if o < 0 else f'+{o}' for o in offs])
axC.set_xlabel('speeches before / after'); axC.set_ylabel("EMI of other parties' speeches"); axC.legend(frameon=False, loc='lower right', handlelength=1.6, title='speech by', title_fontsize=6.5)
axC.set_title('Others fall after an AfD speech only', loc='left', fontsize=7.5, pad=6)

fig.savefig(os.path.join(OUT, 'fig_sequence.pdf'), bbox_inches='tight'); fig.savefig(os.path.join(OUT, 'fig_sequence.png'), dpi=300, bbox_inches='tight'); print('saved fig_sequence')
