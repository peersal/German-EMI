"""Figure: the one-sided contagion effect (who moves whom on the Bundestag floor since the AfD's entry).

A  Who moves the next speaker: contagion effect (next minus previous speech by another party) after speeches by
   each party, 95% CI clustered by sitting. Only the AfD bar is negative.
B  AfD -> others: contagion effect after AfD speeches by replying party (bars) and the within-speaker, position-
   adjusted contrast "after AfD minus after other parties" (diamonds).
C  Others -> AfD: within-speaker, position-adjusted contrast for AfD speeches "after party Y minus after the
   other parties". Nothing moves the AfD towards intuition.
Outputs results/contagion_asymmetry.{pdf,png}; numbers in results/parliament/tables/contagion_asymmetry.csv.
"""
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))   # German-EMI repository root: data in data/, outputs in results/
import os, numpy as np, pandas as pd, statsmodels.api as sm, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
os.chdir(REPO_ROOT)
src = open('code/04_analysis/contagion/0411_contagion_sequence.py', encoding='utf-8').read().split('def mdb_mention_network')[0].replace('from matplotlib.patches', 'from matplotlib.patches')
party_colors = {}; exec(src)
PLOT_LABEL_SIZE, PLOT_LEGEND_SIZE, PLOT_PANEL_SIZE, PLOT_TICK_SIZE = 24, 16, 20, 16
party_colors = {'CDU/CSU': 'black', 'SPD': 'red', 'Alliance 90/The Greens': 'green', 'FDP': 'gold', 'The Left Party': 'purple', 'AFD': 'blue'}
SHORT = {'CDU/CSU': 'CDU/CSU', 'SPD': 'SPD', 'Alliance 90/The Greens': 'Greens', 'FDP': 'FDP', 'The Left Party': 'Left', 'AFD': 'AfD'}
EMI = 'emi_w2vparliament'; plt.rcParams.update({'font.family': 'Arial', 'axes.spines.top': False, 'axes.spines.right': False})


def cl(x, groups):
    r = sm.OLS(np.asarray(x, float), np.ones(len(x))).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(groups)[0]})
    return float(r.params[0]), float(r.bse[0])


# ---- sequences since the AfD's entry -------------------------------------------------------------------------
d = load_speech_sequences(EMI, start='2017-10-24')
R = reply_rows(d, EMI)
# A: contagion effect by speaking party
A = {X: cl(R[R.src == X]['diff'], R[R.src == X].sitting) + (int((R.src == X).sum()),) for X in _SEQ_PARTIES if X in set(R.src)}
# B (bars): contagion effect after AfD speeches by replying party
B = {Y: cl(R[(R.src == 'AFD') & (R.dst == Y)]['diff'], R[(R.src == 'AFD') & (R.dst == Y)].sitting) + (int(((R.src == 'AFD') & (R.dst == Y)).sum()),) for Y in _SEQ_PARTIES if Y != 'AFD'}
# within-speaker, position-adjusted levels: speech j of party Y after preceding party X
raw = pd.read_csv('data/revision/bundestag_speeches_2000_2025_raw.csv', usecols=['Source_ID', 'Sitting', 'Agenda'], dtype=str)
d = d.merge(raw[['Source_ID', 'Agenda']], on='Source_ID', how='left'); d['item'] = d.Sitting + '|' + d.Agenda.fillna(''); d['pos'] = d.groupby('item').cumcount().clip(upper=6)
rows = []
for sit, g in d.groupby('Sitting', sort=False):
    p, e, pos, pid = g.party.values, g[EMI].values, g.pos.values, g.politicianId.values
    for j in range(len(g)):
        Y = p[j]
        if Y not in _SEQ_PARTIES: continue
        i = next((k for k in range(j - 1, -1, -1) if p[k] != Y), None)
        if i is None or p[i] not in _SEQ_PARTIES: continue
        rows.append(dict(sitting=sit, pid=pid[j], Y=Y, X=p[i], pos=pos[j], y=e[j]))
W = pd.DataFrame(rows).dropna(subset=['pid']); W = W[W.groupby('pid').pid.transform('size') >= 5].copy()
W['dev'] = W.y - W.groupby('pid').y.transform('mean'); W['dev'] = W.dev - W.groupby(['Y', 'pos']).dev.transform('mean')
def within(sub):   # mean deviation of these speeches from the speaker's own mean (position-adjusted), s.e. clustered by sitting
    m, se = cl(sub.dev, sub.sitting); return m, se, len(sub)
B2 = {Y: within(W[(W.Y == Y) & (W.X == 'AFD')]) for Y in B}       # Y's speeches after AfD vs Y's own mean
C = {X: within(W[(W.Y == 'AFD') & (W.X == X)]) for X in B}        # AfD speeches after X vs the AfD speaker's own mean
# full matrix for the heatmap / network: preceding party X (rows) -> replying party Y (columns)
MAT = {(X, Y): within(W[(W.X == X) & (W.Y == Y)]) for X in _SEQ_PARTIES for Y in _SEQ_PARTIES if X != Y and ((W.X == X) & (W.Y == Y)).sum() >= 30}
out = [dict(panel='A', party=k, est=v[0], se=v[1], n=v[2]) for k, v in A.items()] + [dict(panel='B_raw', party=k, est=v[0], se=v[1], n=v[2]) for k, v in B.items()] + \
      [dict(panel='B_within', party=k, est=v[0], se=v[1], n=v[2]) for k, v in B2.items()] + [dict(panel='C_within', party=k, est=v[0], se=v[1], n=v[2]) for k, v in C.items()]
pd.DataFrame(out).to_csv('results/parliament/tables/contagion_asymmetry.csv', index=False); print(pd.DataFrame(out).round(3).to_string(index=False))

# ---- figure ----------------------------------------------------------------------------------------------------
order_A = ['AFD', 'The Left Party', 'Alliance 90/The Greens', 'SPD', 'FDP', 'CDU/CSU']
order_Y = ['The Left Party', 'Alliance 90/The Greens', 'SPD', 'FDP', 'CDU/CSU']
fig, ax = plt.subplots(1, 3, figsize=(24, 7.5), gridspec_kw=dict(width_ratios=[1.1, 1, 1], wspace=0.35))


def dots(a, keys, data, hollow=None, offset=0.0, label=None, label2=None):
    for k_i, k in enumerate(keys):
        est, se, n = data[k]; y = k_i + offset; c = party_colors[k]
        a.plot([est - 1.96 * se, est + 1.96 * se], [y, y], color=c, lw=3.0, solid_capstyle='butt', zorder=2)
        a.plot(est, y, 'o', ms=14, color=c, mec='white', mew=1.5, zorder=3)
        if hollow is not None:
            e2, s2, _ = hollow[k]; a.plot([e2 - 1.96 * s2, e2 + 1.96 * s2], [y - 0.28, y - 0.28], color=c, lw=1.6, alpha=0.9, zorder=2)
            a.plot(e2, y - 0.28, 'D', ms=10, mfc='white', mec=c, mew=2.2, zorder=3)
    a.axvline(0, color='black', lw=1.0, alpha=0.7, zorder=1); a.set_yticks(range(len(keys))); a.set_yticklabels([SHORT[k] for k in keys], fontsize=PLOT_TICK_SIZE + 2)
    a.tick_params(labelsize=PLOT_TICK_SIZE); a.grid(True, axis='x', alpha=0.25); a.invert_yaxis()
    for lab in a.get_yticklabels(): lab.set_color(party_colors[[k for k in keys if SHORT[k] == lab.get_text()][0]]); lab.set_fontweight('bold')


dots(ax[0], order_A, A); ax[0].set_xlabel('Contagion effect after a speech by this party\n(EMI, next minus previous speech by another party)', fontsize=PLOT_LEGEND_SIZE + 1)
ax[0].set_title('A   Who moves the next speaker?', loc='left', fontsize=PLOT_PANEL_SIZE, fontweight='bold', pad=12)
ax[0].set_xlim(-0.16, 0.12)
dots(ax[1], order_Y, B, hollow=B2); ax[1].set_xlabel('Change in EMI of the reply after an AfD speech\n(by replying party)', fontsize=PLOT_LEGEND_SIZE + 1)
ax[1].set_title('B   AfD → others', loc='left', fontsize=PLOT_PANEL_SIZE, fontweight='bold', pad=12); ax[1].set_xlim(-0.36, 0.2)
dots(ax[2], order_Y, C); ax[2].set_xlabel('EMI of AfD speeches after this party\n(deviation from the AfD speaker\'s own mean)', fontsize=PLOT_LEGEND_SIZE + 1)
ax[2].set_title('C   Others → AfD', loc='left', fontsize=PLOT_PANEL_SIZE, fontweight='bold', pad=12); ax[2].set_xlim(-0.36, 0.4)
for a in ax[1:]: a.set_ylim(len(order_Y) - 0.5, -0.7)
h = [plt.Line2D([], [], color='gray', marker='o', ms=12, lw=3, label='contagion effect (raw, 95% CI)'), plt.Line2D([], [], color='gray', marker='D', ms=9, mfc='white', mew=2, lw=1.6, label='deviation from the speaker\'s own mean, position-adjusted (95% CI)')]
ax[1].legend(handles=h, frameon=False, fontsize=PLOT_LEGEND_SIZE - 1, loc='lower left', bbox_to_anchor=(0.0, -0.42), ncol=1)
for ext in ('pdf', 'png'):
    try: fig.savefig(f'results/contagion_asymmetry.{ext}', dpi=300, bbox_inches='tight', facecolor='white')
    except PermissionError: fig.savefig(f'results/contagion_asymmetry_tmp.{ext}', dpi=300, bbox_inches='tight', facecolor='white'); print(f'contagion_asymmetry.{ext} locked, wrote _tmp')
print('saved results/contagion_asymmetry.pdf/png')

# ---- figure 2: matrix and network of the within-speaker reply levels ------------------------------------------
from matplotlib.patches import FancyArrowPatch, Circle
import matplotlib.colors as mcolors
order = ['AFD', 'CDU/CSU', 'SPD', 'FDP', 'Alliance 90/The Greens', 'The Left Party']
M = pd.DataFrame(index=order, columns=order, dtype=float); SE = M.copy(); NN = M.copy()
for (X, Y), (m, se, n) in MAT.items(): M.loc[X, Y] = m; SE.loc[X, Y] = se; NN.loc[X, Y] = n
M.to_csv('results/parliament/tables/contagion_within_matrix.csv'); SE.to_csv('results/parliament/tables/contagion_within_matrix_se.csv')
fig2, (axm, axn) = plt.subplots(1, 2, figsize=(22, 9.5), gridspec_kw=dict(width_ratios=[1.15, 1], wspace=0.25))
norm = mcolors.TwoSlopeNorm(vmin=-0.2, vcenter=0, vmax=0.2); cmap = _SEQ_EMI_CMAP
im = axm.imshow(M.values.astype(float), cmap=cmap, norm=norm, aspect='equal')
for i, X in enumerate(order):
    for j, Y in enumerate(order):
        if X == Y: axm.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color='#EEEEEE')); continue
        m, se = M.loc[X, Y], SE.loc[X, Y]
        if np.isnan(m): continue
        sig = abs(m) > 1.96 * se
        axm.text(j, i, f'{m:+.2f}' + ('*' if sig else ''), ha='center', va='center', fontsize=PLOT_TICK_SIZE + (2 if sig else 0), fontweight='bold' if sig else 'normal', color='black')
axm.set_xticks(range(6)); axm.set_yticks(range(6)); axm.set_xticklabels([SHORT[k] for k in order], fontsize=PLOT_TICK_SIZE + 1); axm.set_yticklabels([SHORT[k] for k in order], fontsize=PLOT_TICK_SIZE + 1)
for lab, k in zip(axm.get_xticklabels(), order): lab.set_color(party_colors[k]); lab.set_fontweight('bold')
for lab, k in zip(axm.get_yticklabels(), order): lab.set_color(party_colors[k]); lab.set_fontweight('bold')
axm.set_xlabel('Replying party (speaks next)', fontsize=PLOT_LEGEND_SIZE + 3, fontweight='bold'); axm.set_ylabel('Preceding party (spoke before)', fontsize=PLOT_LEGEND_SIZE + 3, fontweight='bold')
axm.add_patch(plt.Rectangle((-0.5, -0.5), 6, 1, fill=False, ec=party_colors['AFD'], lw=3)); axm.add_patch(plt.Rectangle((-0.5, -0.5), 1, 6, fill=False, ec=party_colors['AFD'], lw=3, ls='--'))
axm.text(-0.02, 1.03, 'A', transform=axm.transAxes, fontsize=PLOT_PANEL_SIZE, fontweight='bold', ha='right', va='bottom')
axm.set_title('How a party speaks right after another party\n(EMI, deviation from the speaker\'s own mean; * = 95% CI excludes 0)', loc='left', fontsize=PLOT_LEGEND_SIZE + 1, pad=10)
cb = fig2.colorbar(im, ax=axm, fraction=0.046, pad=0.03); cb.set_label('EMI relative to own mean', fontsize=PLOT_LEGEND_SIZE); cb.ax.tick_params(labelsize=PLOT_TICK_SIZE)
# network: significant edges only, X -> Y, width by size, colour by sign
ang = {p: np.pi / 2 - 2 * np.pi * i / 6 for i, p in enumerate(order)}; pos = {p: (np.cos(a), np.sin(a)) for p, a in ang.items()}
for (X, Y), (m, se, n) in MAT.items():
    if abs(m) <= 1.96 * se: continue
    (x0, y0), (x1, y1) = pos[X], pos[Y]; vx, vy = x1 - x0, y1 - y0; L = np.hypot(vx, vy); ux, uy = vx / L, vy / L
    axn.add_patch(FancyArrowPatch((x0 + ux * 0.2, y0 + uy * 0.2), (x1 - ux * 0.22, y1 - uy * 0.22), connectionstyle='arc3,rad=0.18', arrowstyle='-|>', mutation_scale=28,
                                  lw=2 + 40 * abs(m), color=cmap(norm(m)), alpha=0.95, zorder=3, shrinkA=0, shrinkB=0))
    sx, sy, ex, ey = x0 + ux * 0.2, y0 + uy * 0.2, x1 - ux * 0.22, y1 - uy * 0.22; cx, cy = (sx + ex) / 2 + 0.18 * (ey - sy), (sy + ey) / 2 - 0.18 * (ex - sx)   # arc3 control point
    mx, my = 0.25 * sx + 0.5 * cx + 0.25 * ex, 0.25 * sy + 0.5 * cy + 0.25 * ey; mx, my = mx + 0.10 * (cx - (sx + ex) / 2) / 0.18 / L * 0.18, my + 0.10 * (cy - (sy + ey) / 2) / 0.18 / L * 0.18
    axn.text(mx, my, f'{m:+.2f}', fontsize=PLOT_TICK_SIZE - 1, fontweight='bold', color=cmap(norm(m)), ha='center', va='center', zorder=6, bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.85))
for p in order:
    axn.add_patch(Circle(pos[p], 0.17, facecolor=party_colors[p], edgecolor='white', lw=2, zorder=4))
    axn.text(pos[p][0] * 1.42, pos[p][1] * 1.42, SHORT[p], ha='center', va='center', fontsize=PLOT_TICK_SIZE + 3, color=party_colors[p], fontweight='bold', zorder=5)
axn.set_xlim(-1.75, 1.75); axn.set_ylim(-1.65, 1.65); axn.set_aspect('equal'); axn.axis('off')
axn.text(0.0, 1.03, 'B', transform=axn.transAxes, fontsize=PLOT_PANEL_SIZE, fontweight='bold', ha='right', va='bottom')
axn.set_title('Significant reactions only (arrow from the preceding to the replying party;\nred = reply more intuition-based, blue = more evidence-based)', loc='left', fontsize=PLOT_LEGEND_SIZE + 1, pad=10)
for ext in ('pdf', 'png'):
    try: fig2.savefig(f'results/contagion_asymmetry_network.{ext}', dpi=300, bbox_inches='tight', facecolor='white')
    except PermissionError: fig2.savefig(f'results/contagion_asymmetry_network_tmp.{ext}', dpi=300, bbox_inches='tight', facecolor='white'); print('network file locked, wrote _tmp')
print('saved results/contagion_asymmetry_network.pdf/png')

# ---- figure 3: hub layout, AfD in the centre; outgoing = how Y speaks after the AfD, incoming = how the AfD speaks after Y
fig3, axh = plt.subplots(figsize=(13, 11))
Ys = ['CDU/CSU', 'SPD', 'FDP', 'Alliance 90/The Greens', 'The Left Party']; Rr = 1.45
angs = {Y: np.pi / 2 + 2 * np.pi * i / 5 for i, Y in enumerate(Ys)}
def arc(a, p0, p1, m, se, rad):
    sig = abs(m) > 1.96 * se; col = cmap(norm(m)) if sig else '#BBBBBB'
    a.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f'arc3,rad={rad}', arrowstyle='-|>', mutation_scale=30, lw=2.5 + 45 * abs(m), color=col, alpha=1.0 if sig else 0.7, zorder=3, shrinkA=0, shrinkB=0))
    sx, sy = p0; ex, ey = p1; cx, cy = (sx + ex) / 2 + rad * (ey - sy), (sy + ey) / 2 - rad * (ex - sx)
    mx, my = 0.25 * sx + 0.5 * cx + 0.25 * ex, 0.25 * sy + 0.5 * cy + 0.25 * ey
    a.text(mx, my, f'{m:+.2f}' + ('*' if sig else ''), fontsize=PLOT_TICK_SIZE, fontweight='bold' if sig else 'normal', color=col if sig else '#777777', ha='center', va='center', zorder=6, bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.9))
for Y in Ys:
    x, y = Rr * np.cos(angs[Y]), Rr * np.sin(angs[Y]); ux, uy = x / Rr, y / Rr
    m_out, se_out, _ = B2[Y]; m_in, se_in, _ = C[Y]
    arc(axh, (ux * 0.28, uy * 0.28), (x - ux * 0.3, y - uy * 0.3), m_out, se_out, 0.28)     # AfD -> Y
    arc(axh, (x - ux * 0.3, y - uy * 0.3), (ux * 0.28, uy * 0.28), m_in, se_in, 0.28)       # Y -> AfD
    axh.add_patch(Circle((x, y), 0.26, facecolor=party_colors[Y], edgecolor='white', lw=2, zorder=4))
    axh.text(x * 1.3, y * 1.3, SHORT[Y], ha='center', va='center', fontsize=PLOT_TICK_SIZE + 4, color=party_colors[Y], fontweight='bold', zorder=5)
axh.add_patch(Circle((0, 0), 0.26, facecolor=party_colors['AFD'], edgecolor='white', lw=2, zorder=4)); axh.text(0, -0.42, 'AfD', ha='center', va='center', fontsize=PLOT_TICK_SIZE + 4, color=party_colors['AFD'], fontweight='bold', zorder=5)
axh.set_xlim(-2.1, 2.1); axh.set_ylim(-1.95, 2.05); axh.set_aspect('equal'); axh.axis('off')
axh.set_title("Who changes register after whom? EMI of a party's speech right after the other party,\nrelative to the speaker's own mean (position-adjusted). Arrow from the party that spoke first to the one replying.\nRed = reply more intuition-based, blue = more evidence-based, grey = not significant; * = 95% CI excludes 0.", loc='left', fontsize=PLOT_LEGEND_SIZE, pad=14)
for ext in ('pdf', 'png'):
    try: fig3.savefig(f'results/contagion_asymmetry_hub.{ext}', dpi=300, bbox_inches='tight', facecolor='white')
    except PermissionError: fig3.savefig(f'results/contagion_asymmetry_hub_tmp.{ext}', dpi=300, bbox_inches='tight', facecolor='white')
print('saved results/contagion_asymmetry_hub.pdf/png')
