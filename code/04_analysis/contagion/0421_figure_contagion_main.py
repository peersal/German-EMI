"""1 x 2 figure: A = reply network (no value labels), B = locality of the contagion effect (k-window).
Same statistics as figure_reply_network.py and figure_reply_main.py (since 24 Oct 2017, s.e. clustered by sitting).
Output: results/contagion_reply_ab.{pdf,png}.
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
EMI = 'emi_w2vparliament'; GREY = '#7F7F7F'; AFD = party_colors['AFD']
plt.rcParams.update({'font.family': 'Arial', 'axes.spines.top': False, 'axes.spines.right': False})


def cl(x, groups):
    r = sm.OLS(np.asarray(x, float), np.ones(len(x))).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(groups)[0]})
    return float(r.params[0]), float(r.bse[0])


d = load_speech_sequences(EMI, start='2017-10-24'); R = reply_rows(d, EMI)
d['emi_dm'] = d[EMI] - d.groupby(d.party.fillna('other'))[EMI].transform('mean')     # relative to the party's own mean: makes arrows into the AfD comparable
Rdm = reply_rows(d, 'emi_dm')
A = {X: cl(Rdm[Rdm.src == X]['diff'], Rdm[Rdm.src == X].sitting) for X in _SEQ_PARTIES}
E = {(X, Y): cl(Rdm[(Rdm.src == X) & (Rdm.dst == Y)]['diff'], Rdm[(Rdm.src == X) & (Rdm.dst == Y)].sitting) for X in _SEQ_PARTIES for Y in _SEQ_PARTIES if X != Y and ((Rdm.src == X) & (Rdm.dst == Y)).sum() >= 50}
pd.DataFrame([dict(src=k[0], dst=k[1], effect=v[0], se=v[1]) for k, v in E.items()]).to_csv('results/parliament/tables/contagion_network_edges_partydemeaned.csv', index=False)
print('party-demeaned: after AfD', {SHORT[Y]: round(E[('AFD', Y)][0], 3) for Y in _SEQ_PARTIES if Y != 'AFD'}); print('party-demeaned: into AfD', {SHORT[X]: round(E[(X, 'AFD')][0], 3) for X in _SEQ_PARTIES if X != 'AFD'}); print('aggregate by speaking party', {SHORT[X]: round(v[0], 3) for X, v in A.items()})
K = pd.read_csv('results/parliament/tables/contagion_sequence_k.csv')

from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(22, 14))   # same canvas as the main cross-platform figure (analysis.ipynb cell 12)
gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1.2, 0.8], wspace=0.18, hspace=0.3)
ax, axk, axS = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
# ---- A: network --------------------------------------------------------------------------------------------------
norm = mcolors.Normalize(-0.2, 0.2); cmap = _SEQ_EMI_CMAP
Ys = ['CDU/CSU', 'SPD', 'FDP', 'Alliance 90/The Greens', 'The Left Party']; Rr = 1.5; NODE_R = 0.24
pos = {'AFD': (0.0, 0.0)}; pos.update({Y: (Rr * np.cos(np.pi / 2 + 2 * np.pi * i / 5), Rr * np.sin(np.pi / 2 + 2 * np.pi * i / 5)) for i, Y in enumerate(Ys)})


def edge(X, Y, m, bold, rad):
    (x0, y0), (x1, y1) = pos[X], pos[Y]; vx, vy = x1 - x0, y1 - y0; L = np.hypot(vx, vy); ux, uy = vx / L, vy / L
    p0, p1 = (x0 + ux * (NODE_R + 0.05), y0 + uy * (NODE_R + 0.05)), (x1 - ux * (NODE_R + 0.06), y1 - uy * (NODE_R + 0.06))
    ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f'arc3,rad={rad}', arrowstyle='-|>', mutation_scale=30 if bold else 16, lw=(3 + 45 * abs(m)) if bold else (1.2 + 15 * abs(m)),
                                 color=cmap(norm(m)), alpha=0.95 if bold else 0.55, zorder=3 if bold else 2, shrinkA=0, shrinkB=0))


for (X, Y), (m, se) in E.items():
    if X != 'AFD' and Y != 'AFD': edge(X, Y, m, bold=False, rad=0.22)
for Y in Ys: edge('AFD', Y, E[('AFD', Y)][0], bold=True, rad=0.18); edge(Y, 'AFD', E[(Y, 'AFD')][0], bold=True, rad=0.18)
for p, (x, y) in pos.items():
    m, se = A[p]
    ax.add_patch(Circle((x, y), NODE_R + 0.07, facecolor=cmap(norm(m)), edgecolor='none', zorder=4))
    ax.add_patch(Circle((x, y), NODE_R, facecolor=party_colors[p], edgecolor='white', lw=2, zorder=5))
    ux, uy = (x / Rr, y / Rr) if p != 'AFD' else (0.0, -1.0)
    if p != 'AFD' and abs(ux) > 0.5:
        ax.text(x + np.sign(ux) * (NODE_R + 0.16), y, SHORT[p], ha='right' if ux < 0 else 'left', va='center', fontsize=PLOT_TICK_SIZE + 2, color=party_colors[p], fontweight='bold', zorder=7)
    else:
        r1 = Rr + 0.45 if p != 'AFD' else 0.5; ax.text(ux * r1, uy * r1, SHORT[p], ha='center', va='center', fontsize=PLOT_TICK_SIZE + 2, color=party_colors[p], fontweight='bold', zorder=7)
ax.set_xlim(-2.3, 2.3); ax.set_ylim(-1.85, 1.95); ax.set_aspect('equal'); ax.axis('off')
sm_ = plt.cm.ScalarMappable(cmap=cmap, norm=norm); cb = fig.colorbar(sm_, ax=ax, orientation='horizontal', fraction=0.05, pad=0.0, shrink=0.7)
cb.set_label('Contagion effect (next minus previous speech by another party,\nEMI relative to each speaker\'s party mean)', fontsize=PLOT_LEGEND_SIZE, fontweight='bold'); cb.ax.tick_params(labelsize=PLOT_TICK_SIZE)
ax.text(-0.02, 0.98, 'A', transform=ax.transAxes, fontsize=PLOT_PANEL_SIZE, fontweight='bold', va='top', ha='left')
# ---- B: locality, one line per speaking party (k = 1..5) ---------------------------------------------------------
KS = (1, 2, 3, 4, 5); rows = []
for sit, g in d.groupby('Sitting', sort=False):
    party, emi = g.party.values, g[EMI].values; n = len(g)
    for i in range(n):
        X = party[i]
        if X not in _SEQ_PARTIES: continue
        before = [j for j in range(i - 1, -1, -1) if party[j] != X]; after = [j for j in range(i + 1, n) if party[j] != X]
        rec = dict(sitting=sit, src=X)
        for k in KS: rec[f'd{k}'] = (emi[after[:k]].mean() - emi[before[:k]].mean()) if len(before) >= k and len(after) >= k else np.nan
        rows.append(rec)
KP = pd.DataFrame(rows); kp_rows = []
for X in _SEQ_PARTIES:
    for k in KS:
        sub = KP[(KP.src == X)].dropna(subset=[f'd{k}']); m, se = cl(sub[f'd{k}'], sub.sitting); kp_rows.append(dict(party=X, k=k, effect=m, se=se, n=len(sub)))
KPP = pd.DataFrame(kp_rows); KPP.to_csv('results/parliament/tables/contagion_sequence_k_by_party.csv', index=False)
order_k = ['AFD', 'CDU/CSU', 'SPD', 'FDP', 'Alliance 90/The Greens', 'The Left Party']; off = {X: (i - 2.5) * 0.045 for i, X in enumerate(order_k)}
for X in order_k:
    r = KPP[KPP.party == X]; big = X == 'AFD'
    axk.errorbar(r.k + off[X], r.effect, yerr=1.96 * r.se, color=party_colors[X], marker='D' if big else 'o', ms=12 if big else 9, lw=3.0 if big else 1.8, capsize=4, alpha=1.0 if big else 0.9, label=SHORT[X], zorder=5 if big else 3)
axk.axhline(0, color='black', lw=1.0, alpha=0.7); axk.set_xticks(KS); axk.set_xlim(0.2, 5.45)
axk.axvspan(0.3, 0.6, color='#DDDDDD', alpha=0.8, lw=0, zorder=0)
axk.set_xlabel('Window: k speeches on each side', fontsize=PLOT_LABEL_SIZE, fontweight='bold'); axk.set_ylabel('Contagion effect (EMI)', fontsize=PLOT_LABEL_SIZE, fontweight='bold')
axk.tick_params(labelsize=PLOT_TICK_SIZE); axk.grid(True, alpha=0.25); axk.legend(frameon=False, fontsize=PLOT_LEGEND_SIZE, loc='lower right', ncol=2, handlelength=1.4, columnspacing=1.0)
axk.text(-0.12, 0.98, 'B', transform=axk.transAxes, fontsize=PLOT_PANEL_SIZE, fontweight='bold', va='top', ha='left')
# ---- C: one sitting as a strip -----------------------------------------------------------------------------------
example_sitting = '19/133'
ex = d[d.Sitting == example_sitting].reset_index(drop=True); ex['seq'] = np.arange(1, len(ex) + 1)
is_afd = (ex.party == 'AFD').to_numpy(); non = ex[~is_afd].copy(); non['run'] = non[EMI].rolling(3, center=True, min_periods=2).mean()
reply, seen = [], set()
for i in np.where(is_afd)[0]:
    b = next((j for j in range(i - 1, -1, -1) if not is_afd[j]), None); a = next((j for j in range(i + 1, len(ex)) if not is_afd[j]), None)
    if b is not None and a is not None and (b, a) not in seen: seen.add((b, a)); reply.append((i, b, a, ex[EMI].iloc[a] - ex[EMI].iloc[b]))
norm_s = mcolors.Normalize(-1.5, 1.5); ymin, ymax = ex[EMI].min(), ex[EMI].max(); pad = 0.12 * (ymax - ymin)
for i in np.where(is_afd)[0]: axS.axvspan(i + 0.5, i + 1.5, color=party_colors['AFD'], alpha=0.10, lw=0, zorder=1)
for i, b, a, dv in reply:
    axS.annotate('', xy=(a + 1, ex[EMI].iloc[a]), xytext=(b + 1, ex[EMI].iloc[b]), arrowprops=dict(arrowstyle='-|>', color=cmap(norm_s(dv)), lw=2.2, mutation_scale=18, connectionstyle='arc3,rad=-0.25', alpha=0.9), zorder=4)
axS.plot(non.seq, non.run, color='black', lw=2.0, alpha=0.75, zorder=2, label='other parties, running mean (3 speeches)')
for X in _SEQ_PARTIES:
    gx = ex[ex.party == X]
    if len(gx): axS.scatter(gx.seq, gx[EMI], s=150 if X == 'AFD' else 70, marker='D' if X == 'AFD' else 'o', color=party_colors[X], edgecolor='white', linewidth=1.0, zorder=5, label=SHORT[X])
mm = non[non.mentions_afd == 1]; axS.scatter(mm.seq, mm[EMI], s=330, facecolor='none', edgecolor='#D62728', linewidth=2.0, zorder=6, label='speech names the AfD')
axS.axhline(0, color='black', lw=0.8, alpha=0.6); axS.grid(True, axis='y', alpha=0.25)
axS.set_xlim(0.3, len(ex) + 0.7); axS.set_ylim(ymin - pad * 0.6, ymax + pad * 0.6); axS.set_yticks(np.arange(np.ceil(ymin), np.floor(ymax) + 1))
axS.set_xlabel('Speech order within the sitting', fontsize=PLOT_LABEL_SIZE, fontweight='bold'); axS.set_ylabel('EMI of the speech', fontsize=PLOT_LABEL_SIZE, fontweight='bold')
axS.tick_params(labelsize=PLOT_TICK_SIZE); axS.set_title(f"Sitting {example_sitting}, {pd.Timestamp(ex.date.iloc[0]).strftime('%d %B %Y')}", fontsize=PLOT_LEGEND_SIZE + 2, loc='left', pad=8)
axS.legend(frameon=False, fontsize=PLOT_LEGEND_SIZE, loc='upper left', bbox_to_anchor=(0.0, -0.14), ncol=8, handletextpad=0.4, columnspacing=1.2)
axS.text(-0.05, 1.03, 'C', transform=axS.transAxes, fontsize=PLOT_PANEL_SIZE, fontweight='bold', va='bottom', ha='left')
rr = np.array([dv for *_, dv in reply]); print(f'sitting {example_sitting}: {len(ex)} speeches, {int(is_afd.sum())} AfD; {int((rr < 0).sum())} of {len(rr)} replies lower (mean {rr.mean():+.2f})')
for ext in ('pdf', 'png'):
    try: fig.savefig(f'results/contagion_reply_ab.{ext}', dpi=300, bbox_inches='tight', facecolor='white')
    except PermissionError: fig.savefig(f'results/contagion_reply_ab_tmp.{ext}', dpi=300, bbox_inches='tight', facecolor='white'); print('locked, wrote _tmp')
print('saved results/contagion_reply_ab.pdf/png')
