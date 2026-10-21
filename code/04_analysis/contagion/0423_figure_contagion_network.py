"""Network version of panels A + B of the main reply figure (since 24 Oct 2017).

Nodes: parties; AfD in the centre. Edge X -> Y: mean contagion effect when a speech by X is followed by a speech by Y
(EMI of Y's speech minus the previous speech by another party). AfD edges are drawn bold with their values; edges
between the other parties thin (all zero or positive). Edges INTO the AfD are not drawn: in the raw statistic they
mostly reflect the AfD's low level rather than a reaction (see the within-speaker analysis). The ring around each
node and the number beside it give the party's overall contagion effect (panel A): what happens to the next speaker
after this party has spoken. Output: results/contagion_reply_network.{pdf,png}.
"""
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))   # German-EMI repository root: data in data/, outputs in results/
import os, numpy as np, pandas as pd, statsmodels.api as sm, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch, Circle
os.chdir(REPO_ROOT)
src = open('code/04_analysis/contagion/0411_contagion_sequence.py', encoding='utf-8').read().split('def mdb_mention_network')[0]
party_colors = {}; exec(src)
PLOT_LABEL_SIZE, PLOT_LEGEND_SIZE, PLOT_PANEL_SIZE, PLOT_TICK_SIZE = 24, 16, 20, 16
party_colors = {'CDU/CSU': 'black', 'SPD': 'red', 'Alliance 90/The Greens': 'green', 'FDP': 'gold', 'The Left Party': 'purple', 'AFD': 'blue'}
SHORT = {'CDU/CSU': 'CDU/CSU', 'SPD': 'SPD', 'Alliance 90/The Greens': 'Greens', 'FDP': 'FDP', 'The Left Party': 'Left', 'AFD': 'AfD'}
EMI = 'emi_w2vparliament'; plt.rcParams.update({'font.family': 'Arial'})


def cl(x, groups):
    r = sm.OLS(np.asarray(x, float), np.ones(len(x))).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(groups)[0]})
    return float(r.params[0]), float(r.bse[0])


d = load_speech_sequences(EMI, start='2017-10-24'); R = reply_rows(d, EMI)
A = {X: cl(R[R.src == X]['diff'], R[R.src == X].sitting) for X in _SEQ_PARTIES}
E = {(X, Y): cl(R[(R.src == X) & (R.dst == Y)]['diff'], R[(R.src == X) & (R.dst == Y)].sitting) + (int(((R.src == X) & (R.dst == Y)).sum()),)
     for X in _SEQ_PARTIES for Y in _SEQ_PARTIES if X != Y and ((R.src == X) & (R.dst == Y)).sum() >= 50}
pd.DataFrame([dict(src=k[0], dst=k[1], effect=v[0], se=v[1], n=v[2]) for k, v in E.items()]).to_csv('results/parliament/tables/contagion_reply_network_edges.csv', index=False)

norm = mcolors.Normalize(-0.2, 0.2); cmap = _SEQ_EMI_CMAP
Ys = ['CDU/CSU', 'SPD', 'FDP', 'Alliance 90/The Greens', 'The Left Party']; Rr = 1.5
pos = {'AFD': (0.0, 0.0)}; pos.update({Y: (Rr * np.cos(np.pi / 2 + 2 * np.pi * i / 5), Rr * np.sin(np.pi / 2 + 2 * np.pi * i / 5)) for i, Y in enumerate(Ys)})
NODE_R = 0.24
fig, ax = plt.subplots(figsize=(14, 13))


def edge(X, Y, m, se, bold, rad):
    (x0, y0), (x1, y1) = pos[X], pos[Y]; vx, vy = x1 - x0, y1 - y0; L = np.hypot(vx, vy); ux, uy = vx / L, vy / L
    p0, p1 = (x0 + ux * (NODE_R + 0.05), y0 + uy * (NODE_R + 0.05)), (x1 - ux * (NODE_R + 0.06), y1 - uy * (NODE_R + 0.06))
    col = cmap(norm(m)); lw = (3 + 45 * abs(m)) if bold else (1.2 + 15 * abs(m))
    ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f'arc3,rad={rad}', arrowstyle='-|>', mutation_scale=30 if bold else 16, lw=lw, color=col,
                                 alpha=0.95 if bold else 0.55, zorder=3 if bold else 2, shrinkA=0, shrinkB=0))
    if bold:
        sx, sy = p0; ex, ey = p1; cx, cy = (sx + ex) / 2 + rad * (ey - sy), (sy + ey) / 2 - rad * (ex - sx)
        mx, my = 0.25 * sx + 0.5 * cx + 0.25 * ex, 0.25 * sy + 0.5 * cy + 0.25 * ey
        sig = abs(m) > 1.96 * se
        ax.text(mx, my, f'{m:+.2f}' + ('' if sig else ' (n.s.)'), fontsize=PLOT_TICK_SIZE + 1, fontweight='bold', color=col, ha='center', va='center', zorder=6,
                bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='none', alpha=0.92))


# thin edges among the five other parties (never negative at the aggregate level)
for (X, Y), (m, se, n) in E.items():
    if X == 'AFD' or Y == 'AFD': continue
    edge(X, Y, m, se, bold=False, rad=0.22)
# bold edges from the AfD
for Y in Ys:
    m, se, n = E[('AFD', Y)]; edge('AFD', Y, m, se, bold=True, rad=0.0)
# nodes with ring = overall contagion effect after this party (panel A)
for p, (x, y) in pos.items():
    m, se = A[p]
    ax.add_patch(Circle((x, y), NODE_R + 0.07, facecolor=cmap(norm(m)), edgecolor='none', zorder=4))
    ax.add_patch(Circle((x, y), NODE_R, facecolor=party_colors[p], edgecolor='white', lw=2, zorder=5))
    ux, uy = (x / Rr, y / Rr) if p != 'AFD' else (0.0, -1.0)
    col2 = cmap(norm(m)) if abs(m) > 1.96 * se else '#777777'
    if p != 'AFD' and abs(ux) > 0.5:          # lateral nodes: labels beside the node, anchored at its outer edge
        ha = 'right' if ux < 0 else 'left'; ax_x = x + np.sign(ux) * (NODE_R + 0.16)
        ax.text(ax_x, y + 0.13, SHORT[p], ha=ha, va='center', fontsize=PLOT_TICK_SIZE + 5, color=party_colors[p], fontweight='bold', zorder=7)
        ax.text(ax_x, y - 0.15, f'after {SHORT[p]}: {m:+.2f}', ha=ha, va='center', fontsize=PLOT_TICK_SIZE - 1, color=col2, fontweight='bold', zorder=7)
    else:
        r1 = Rr + 0.5 if p != 'AFD' else 0.55
        nx, ny = ux * r1, uy * r1; dy = 0.22 if uy > 0.5 else -0.22
        ax.text(nx, ny, SHORT[p], ha='center', va='center', fontsize=PLOT_TICK_SIZE + 5, color=party_colors[p], fontweight='bold', zorder=7)
        ax.text(nx, ny + dy, f'after {SHORT[p]}: {m:+.2f}', ha='center', va='center', fontsize=PLOT_TICK_SIZE - 1, color=col2, fontweight='bold', zorder=7)
ax.set_xlim(-3.1, 3.1); ax.set_ylim(-2.4, 2.4); ax.set_aspect('equal'); ax.axis('off')
sm_ = plt.cm.ScalarMappable(cmap=cmap, norm=norm); cb = fig.colorbar(sm_, ax=ax, orientation='horizontal', fraction=0.035, pad=0.01, shrink=0.6)
cb.set_label('Contagion effect: EMI of the reply minus EMI of the speech before the speaker\n(red = reply more intuition-based, blue = more evidence-based)', fontsize=PLOT_LEGEND_SIZE); cb.ax.tick_params(labelsize=PLOT_TICK_SIZE)
ax.set_title("Who moves the next speaker? Arrow from the party that spoke to the party replying (bold: after AfD speeches, with values);\n"
             "node ring and 'after X' = overall contagion effect after that party's speeches. Only the AfD's edges are red.", loc='left', fontsize=PLOT_LEGEND_SIZE, pad=12)
for ext in ('pdf', 'png'):
    try: fig.savefig(f'results/contagion_reply_network.{ext}', dpi=300, bbox_inches='tight', facecolor='white')
    except PermissionError: fig.savefig(f'results/contagion_reply_network_tmp.{ext}', dpi=300, bbox_inches='tight', facecolor='white'); print('locked, wrote _tmp')
print('saved results/contagion_reply_network.pdf/png')
