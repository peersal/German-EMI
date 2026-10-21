"""Main-text figure: the reply to AfD speeches on the Bundestag floor (since 24 Oct 2017).

A  Contagion effect after a speech by each party (next minus previous speech by another party): only the AfD is negative;
   bracket = AfD-specific contrast (-0.12).
B  Contagion effect after AfD speeches by replying party: SPD, FDP, Greens, Left react, CDU/CSU does not.
C  Replies that name the AfD versus replies that do not (-0.37 vs -0.01).
D  Locality: contagion effect with k speeches on each side, AfD vs other parties.
E  Robustness: raw effect, permutation null, and the matched/adjusted estimates of the reaction (reply level after an
   AfD speech vs after other speeches, conditioning on position, year, replying party, EMI of the speech before and
   the debate's prior register).
Inputs: results/parliament/tables/contagion_sequence_*.csv, contagion_spec_curve.csv; reply rows recomputed from
code/04_analysis/contagion/0411_contagion_sequence.py for panel B. Output: results/contagion_reply_main.{pdf,png}.
"""
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))   # German-EMI repository root: data in data/, outputs in results/
import os, numpy as np, pandas as pd, statsmodels.api as sm, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
os.chdir(REPO_ROOT)
src = open('code/04_analysis/contagion/0411_contagion_sequence.py', encoding='utf-8').read().split('def mdb_mention_network')[0]
party_colors = {}; exec(src)
PLOT_LABEL_SIZE, PLOT_LEGEND_SIZE, PLOT_PANEL_SIZE, PLOT_TICK_SIZE = 24, 16, 20, 16
party_colors = {'CDU/CSU': 'black', 'SPD': 'red', 'Alliance 90/The Greens': 'green', 'FDP': 'gold', 'The Left Party': 'purple', 'AFD': 'blue'}
SHORT = {'CDU/CSU': 'CDU/CSU', 'SPD': 'SPD', 'Alliance 90/The Greens': 'Greens', 'FDP': 'FDP', 'The Left Party': 'Left', 'AFD': 'AfD'}
EMI = 'emi_w2vparliament'; T = 'results/parliament/tables/'; AFD = party_colors['AFD']; GREY = '#7F7F7F'
plt.rcParams.update({'font.family': 'Arial', 'axes.spines.top': False, 'axes.spines.right': False})


def cl(x, groups):
    r = sm.OLS(np.asarray(x, float), np.ones(len(x))).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(groups)[0]})
    return float(r.params[0]), float(r.bse[0])


# ---- data -----------------------------------------------------------------------------------------------------
E = pd.read_csv(T + 'contagion_sequence_effects.csv'); E = E[E.period == 'since Oct 2017'].set_index('party')
d = load_speech_sequences(EMI, start='2017-10-24'); R = reply_rows(d, EMI)
B = {Y: cl(R[(R.src == 'AFD') & (R.dst == Y)]['diff'], R[(R.src == 'AFD') & (R.dst == Y)].sitting) + (int(((R.src == 'AFD') & (R.dst == Y)).sum()),) for Y in _SEQ_PARTIES if Y != 'AFD'}
oth = R[R.src != 'AFD']; afd = R[R.src == 'AFD']
m_a, s_a = cl(afd['diff'], afd.sitting); m_o, s_o = cl(oth['diff'], oth.sitting); contrast, c_se = m_a - m_o, float(np.hypot(s_a, s_o))
NM = pd.read_csv(T + 'contagion_sequence_naming.csv'); K = pd.read_csv(T + 'contagion_sequence_k.csv')
null = pd.read_csv(T + 'contagion_sequence_permutation_null.csv')['null'].to_numpy()
SC = pd.read_csv(T + 'contagion_spec_curve.csv'); SC = SC[(SC.outcome == 'level') & (SC.covariates == '+debate register (pre-treatment)')].set_index('estimator')
print(f'A: AfD {m_a:+.3f} ({s_a:.3f}); others pooled {m_o:+.3f} ({s_o:.3f}); contrast {contrast:+.3f} ({c_se:.3f})'); print('B:', {SHORT[k]: round(v[0], 3) for k, v in B.items()})

# ---- figure -----------------------------------------------------------------------------------------------------
fig = plt.figure(figsize=(26, 15))
gs = GridSpec(2, 1, height_ratios=[1.0, 0.85], hspace=0.42)
gs_top = gs[0].subgridspec(1, 3, wspace=0.38); gs_bot = gs[1].subgridspec(1, 2, width_ratios=[0.8, 1.7], wspace=0.6)
axA, axB, axC = fig.add_subplot(gs_top[0]), fig.add_subplot(gs_top[1]), fig.add_subplot(gs_top[2])
axD, axE = fig.add_subplot(gs_bot[0]), fig.add_subplot(gs_bot[1])


def letter(ax, L, x=-0.18):
    ax.text(x, 1.06, L, transform=ax.transAxes, fontsize=PLOT_PANEL_SIZE + 2, fontweight='bold', va='bottom', ha='left')


def hdots(ax, keys, vals, ses, cols, labels, big=None):
    for i, (k, m, s, c) in enumerate(zip(keys, vals, ses, cols)):
        ax.plot([m - 1.96 * s, m + 1.96 * s], [i, i], color=c, lw=3.5, zorder=2)
        ax.plot(m, i, 'o', ms=18 if (big and k in big) else 14, color=c, mec='white', mew=1.5, zorder=3)
    ax.axvline(0, color='black', lw=1.0, alpha=0.7); ax.set_yticks(range(len(keys))); ax.set_yticklabels(labels, fontsize=PLOT_TICK_SIZE + 2)
    for lab, c in zip(ax.get_yticklabels(), cols): lab.set_color(c); lab.set_fontweight('bold')
    ax.tick_params(labelsize=PLOT_TICK_SIZE); ax.grid(True, axis='x', alpha=0.25); ax.invert_yaxis()


# A: who moves the next speaker
orderA = ['AFD', 'The Left Party', 'Alliance 90/The Greens', 'SPD', 'FDP', 'CDU/CSU']
hdots(axA, orderA, [E.loc[k, 'reply_effect'] for k in orderA], [E.loc[k, 'se'] for k in orderA], [party_colors[k] for k in orderA], [SHORT[k] for k in orderA], big={'AFD'})
axA.set_xlim(-0.165, 0.125); axA.set_xlabel('Contagion effect after a speech by this party\n(EMI of next minus previous speech by another party)', fontsize=PLOT_LEGEND_SIZE + 1)
axA.axvspan(m_o - 1.96 * s_o, m_o + 1.96 * s_o, color=GREY, alpha=0.15, zorder=0); axA.axvline(m_o, color=GREY, lw=1.5, ls='--', zorder=1)
axA.annotate('', xy=(m_a, -0.75), xytext=(m_o, -0.75), arrowprops=dict(arrowstyle='<->', color=AFD, lw=2.5)); axA.set_ylim(5.6, -1.5)
axA.text((m_a + m_o) / 2, -1.05, f'AfD-specific contrast {contrast:+.2f}\n(95% CI {contrast - 1.96 * c_se:+.2f} to {contrast + 1.96 * c_se:+.2f})', ha='center', va='bottom', fontsize=PLOT_LEGEND_SIZE, color=AFD, fontweight='bold')
axA.text(m_o + 0.006, -0.42, 'all other parties pooled', fontsize=PLOT_LEGEND_SIZE - 2, color=GREY, ha='left', va='center')
axA.set_title('Only AfD speeches are followed by\na more intuition-based reply', loc='left', fontsize=PLOT_LEGEND_SIZE + 2, pad=10); letter(axA, 'A')

# B: by replying party
orderB = ['The Left Party', 'Alliance 90/The Greens', 'SPD', 'FDP', 'CDU/CSU']
hdots(axB, orderB, [B[k][0] for k in orderB], [B[k][1] for k in orderB], [party_colors[k] for k in orderB], [f'{SHORT[k]}  (n = {B[k][2]:,})' for k in orderB])
axB.set_xlim(-0.36, 0.2); axB.set_xlabel('Contagion effect after an AfD speech, by replying party', fontsize=PLOT_LEGEND_SIZE + 1)
axB.set_title('SPD, FDP, Greens and Left react,\nCDU/CSU does not', loc='left', fontsize=PLOT_LEGEND_SIZE + 2, pad=10); letter(axB, 'B', x=-0.3)

# C: naming the AfD
nm = NM.set_index('reply'); keysC = ['names the AfD', 'does not name the AfD']
hdots(axC, keysC, [nm.loc[k, 'effect'] for k in keysC], [nm.loc[k, 'se'] for k in keysC], [AFD, GREY], [f'reply names\nthe AfD\n(n = {int(nm.loc[keysC[0], "n"]):,})', f'reply does not\nname the AfD\n(n = {int(nm.loc[keysC[1], "n"]):,})'], big={'names the AfD'})
axC.set_xlim(-0.5, 0.12); axC.set_ylim(1.6, -0.6); axC.set_xlabel('Contagion effect after an AfD speech', fontsize=PLOT_LEGEND_SIZE + 1)
for i, k in enumerate(keysC): axC.text(nm.loc[k, 'effect'], i - 0.3, f'{nm.loc[k, "effect"]:+.2f}', ha='center', va='bottom', fontsize=PLOT_LEGEND_SIZE + 2, fontweight='bold', color=AFD if i == 0 else GREY)
axC.set_title('The reaction is confrontation:\nreplies that name the AfD', loc='left', fontsize=PLOT_LEGEND_SIZE + 2, pad=10); letter(axC, 'C', x=-0.3)

# D: locality
axD.errorbar(K.k, K.afd, yerr=1.96 * K.afd_se, color=AFD, marker='o', ms=12, lw=2.5, capsize=5, label='after AfD speeches')
axD.errorbar(K.k, K.others, yerr=1.96 * K.others_se, color=GREY, marker='s', ms=10, lw=2.5, capsize=5, label='after other parties')
axD.axhline(0, color='black', lw=1.0, alpha=0.7); axD.set_xticks(K.k); axD.set_xlabel('Window: k speeches on each side', fontsize=PLOT_LEGEND_SIZE + 1); axD.set_ylabel('Contagion effect (EMI)', fontsize=PLOT_LEGEND_SIZE + 1)
axD.tick_params(labelsize=PLOT_TICK_SIZE); axD.grid(True, alpha=0.25); axD.legend(frameon=False, fontsize=PLOT_LEGEND_SIZE - 1, loc='lower right', handlelength=1.5)
axD.set_title('The reaction is local: it fades\nwithin a few speeches', loc='left', fontsize=PLOT_LEGEND_SIZE + 2, pad=10); letter(axD, 'D')

# E: robustness ladder
rows = [('raw contagion effect after AfD speeches', m_a, s_a, AFD),
        ('permutation null (2,000 draws)', float(null.mean()), None, GREY),
        ('OLS adjustment', SC.loc['OLS adjustment', 'est'], SC.loc['OLS adjustment', 'se'], AFD),
        ('speaker fixed effects', SC.loc['speaker FE', 'est'], SC.loc['speaker FE', 'se'], AFD),
        ('coarsened exact matching', SC.loc['CEM', 'est'], SC.loc['CEM', 'se'], AFD),
        ('inverse-probability weighting', SC.loc['IPW-ATT', 'est'], SC.loc['IPW-ATT', 'se'], AFD),
        ('1:1 nearest-neighbour matching', SC.loc['1:1 NN on PS (caliper)', 'est'], SC.loc['1:1 NN on PS (caliper)', 'se'], AFD)]
for i, (lab, m, s, c) in enumerate(rows):
    if s is None:
        lo, hi = np.percentile(null, [2.5, 97.5]); axE.plot([lo, hi], [i, i], color=c, lw=10, alpha=0.35, solid_capstyle='butt', zorder=2); axE.plot(m, i, 'D', ms=12, color=c, zorder=3)
    else:
        axE.plot([m - 1.96 * s, m + 1.96 * s], [i, i], color=c, lw=3.5, zorder=2); axE.plot(m, i, 'o', ms=16 if i == 0 else 13, color=c, mec='white', mew=1.5, zorder=3)
        axE.text(m + 1.96 * s + 0.006, i, f'{m:+.2f}', va='center', ha='left', fontsize=PLOT_LEGEND_SIZE, color=c, fontweight='bold')
axE.axvline(0, color='black', lw=1.0, alpha=0.7); axE.set_yticks(range(len(rows))); axE.set_yticklabels([r[0] for r in rows], fontsize=PLOT_TICK_SIZE); axE.invert_yaxis()
axE.tick_params(labelsize=PLOT_TICK_SIZE); axE.grid(True, axis='x', alpha=0.25); axE.set_xlim(-0.15, 0.06)
axE.axhspan(1.5, 6.9, color=GREY, alpha=0.06, zorder=0); axE.set_ylim(7.3, -0.6)
axE.text(-0.148, 6.75, "adjusted estimates condition on debate position, year, replying party, EMI of the speech before and the debate's prior register", ha='left', va='center', fontsize=PLOT_LEGEND_SIZE - 2, color='#444444')
axE.set_xlabel('Effect of an AfD speech on the EMI of the reply', fontsize=PLOT_LEGEND_SIZE + 1)
axE.set_title('About two thirds of the reaction survives adjustment for who speaks when\n(permutation: AfD label reassigned at random within sittings; mean and 95% range)', loc='left', fontsize=PLOT_LEGEND_SIZE + 2, pad=10); letter(axE, 'E', x=-0.3)
for ext in ('pdf', 'png'):
    try: fig.savefig(f'results/contagion_reply_main.{ext}', dpi=300, bbox_inches='tight', facecolor='white')
    except PermissionError: fig.savefig(f'results/contagion_reply_main_tmp.{ext}', dpi=300, bbox_inches='tight', facecolor='white'); print('locked, wrote _tmp')
print('saved results/contagion_reply_main.pdf/png')
