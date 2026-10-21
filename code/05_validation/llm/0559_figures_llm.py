"""Paper-style figures for the LLM annotation (main-analysis fonts and leaning colours).

  python code/05_validation/llm/0559_figures_llm.py
Outputs (PNG 300 dpi + PDF) in results/llm/figures/:
  fig_llm_timeseries   2x2: Bundestag / Twitter x embedding EMI / LLM composite, quarterly means by leaning
  fig_llm_yearly_sd    Bundestag yearly means as deviation from the 2015-2017 mean in SD units (4 measures)
  fig_other_directed   share of other-directed sentences (Bundestag) / tweets (Twitter), quarterly by leaning
"""
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))   # German-EMI repository root: data in data/, outputs in results/
import os, json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(REPO_ROOT, 'results', 'llm'); OUT = os.path.join(RES, 'figures'); os.makedirs(OUT, exist_ok=True)
PLOT_LABEL_SIZE, PLOT_LEGEND_SIZE, PLOT_PANEL_SIZE, PLOT_TICK_SIZE = 24, 16, 20, 16
COLORS = {'left': '#E66101', 'middle': '#4D4D4D', 'right': '#0072B2'}; MARK = {'left': 'o', 'middle': 's', 'right': '^'}
LABEL = {'left': 'Left', 'middle': 'Centre', 'right': 'Right'}
ELECTIONS = ['2017-09-24', '2021-09-26', '2025-02-23']; AFD_ENTRY = '2017-10-24'
plt.rcParams.update({'font.family': 'Arial', 'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': 'white', 'savefig.facecolor': 'white'})


def qdate(s):
    return pd.PeriodIndex(s, freq='Q').to_timestamp(how='start') + pd.Timedelta(days=45)


def decorate(ax, letter, ylabel, afd=True, zero=True):
    for d in ELECTIONS: ax.axvline(pd.Timestamp(d), color='gray', ls='--', lw=1.2, zorder=1)
    if afd: ax.axvline(pd.Timestamp(AFD_ENTRY), color='#D62728', ls='-', lw=1.6, zorder=1)
    if zero: ax.axhline(0, color='black', lw=0.8, alpha=0.6, zorder=1)
    ax.set_ylabel(ylabel, fontsize=PLOT_LABEL_SIZE - 6, fontweight='bold'); ax.tick_params(labelsize=PLOT_TICK_SIZE); ax.grid(True, axis='y', alpha=0.25)
    ax.text(-0.08, 1.04, letter, transform=ax.transAxes, fontsize=PLOT_PANEL_SIZE, fontweight='bold', va='bottom', ha='right')
    ax.set_xlim(pd.Timestamp('2014-10-01'), pd.Timestamp('2025-09-30'))


def lines(ax, Q, col, min_n=30, lw=2.2, ms=6):
    for lean in ['left', 'middle', 'right']:
        g = Q[(Q.leaning == lean) & (Q.n >= min_n)].sort_values('q')
        if len(g) == 0: continue
        ax.plot(qdate(g.q), g[col], color=COLORS[lean], marker=MARK[lean], ms=ms, lw=lw, label=LABEL[lean], zorder=3)


def save(fig, name):
    for ext in ('png', 'pdf'): fig.savefig(os.path.join(OUT, f'{name}.{ext}'), dpi=300, bbox_inches='tight')
    plt.close(fig); print('saved', os.path.join(OUT, name))


# ---- 1. quarterly series, 2 x 2 -------------------------------------------------------------
QP = pd.read_csv(os.path.join(RES, 'parliament', 'ts_quarterly_leaning.csv')); QT = pd.read_csv(os.path.join(RES, 'twitter', 'ts_quarterly_leaning.csv'))
fig, ax = plt.subplots(2, 2, figsize=(20, 12), sharex=True)
lines(ax[0, 0], QP, 'emi_w2vparliament'); decorate(ax[0, 0], 'A', 'EMI (embedding)'); ax[0, 0].set_title('Bundestag, embedding EMI', fontsize=PLOT_LEGEND_SIZE + 2, loc='left', pad=6)
lines(ax[0, 1], QP, 'llm_emi'); decorate(ax[0, 1], 'B', 'LLM composite\n(evidence − intuition rating)', zero=False); ax[0, 1].set_title('Bundestag, LLM rating', fontsize=PLOT_LEGEND_SIZE + 2, loc='left', pad=6)
lines(ax[1, 0], QT, 'emi_w2vparliament'); decorate(ax[1, 0], 'C', 'EMI (embedding)', afd=False); ax[1, 0].set_title('Twitter, embedding EMI', fontsize=PLOT_LEGEND_SIZE + 2, loc='left', pad=6)
lines(ax[1, 1], QT, 'llm_emi'); decorate(ax[1, 1], 'D', 'LLM composite\n(evidence − intuition rating)', afd=False, zero=False); ax[1, 1].set_title('Twitter, LLM rating', fontsize=PLOT_LEGEND_SIZE + 2, loc='left', pad=6)
h, l = ax[0, 0].get_legend_handles_labels()
h += [plt.Line2D([], [], color='gray', ls='--', lw=1.2), plt.Line2D([], [], color='#D62728', lw=1.6)]; l += ['federal election', 'AfD enters the Bundestag (24 Oct 2017)']
fig.legend(h, l, frameon=False, fontsize=PLOT_LEGEND_SIZE, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.02))
for a in ax[1]: a.set_xlabel('Quarter', fontsize=PLOT_LABEL_SIZE - 6, fontweight='bold')
fig.tight_layout(rect=(0, 0.03, 1, 1)); save(fig, 'fig_llm_timeseries')

# ---- 2. Bundestag yearly deviations in SD units ----------------------------------------------
Y = json.load(open(os.path.join(RES, 'parliament', 'ts_numbers.json')))['yearly']; SD = pd.read_csv(os.path.join(RES, 'parliament', 'ts_breakpoints.csv')).set_index('measure').sd
spec = [('emi_w2vparliament', 'Embedding EMI', '#4D4D4D', '-', 'o'), ('llm_emi', 'LLM composite', '#0072B2', '-', 'D'), ('evidenz_basiert', 'LLM evidence rating', '#1B9E77', '--', 's'), ('intuition_basiert', 'LLM intuition rating', '#E66101', '--', '^')]
fig, ax = plt.subplots(figsize=(12, 7))
for key, lab, c, ls, m in spec:
    s = pd.Series({int(k): v for k, v in Y[key].items()}).sort_index(); pre = s.loc[2015:2017].mean(); dev = (s - pre) / SD[key]
    ax.plot(dev.index, dev.values, color=c, ls=ls, marker=m, ms=7, lw=2.4, label=lab)
ax.axvline(2017.73, color='#D62728', lw=1.6); ax.axvline(2021.73, color='gray', ls='--', lw=1.2); ax.axvline(2025.15, color='gray', ls='--', lw=1.2); ax.axhline(0, color='black', lw=0.8, alpha=0.6)
ax.set_xlabel('Year', fontsize=PLOT_LABEL_SIZE - 6, fontweight='bold'); ax.set_ylabel('Deviation from 2015–2017 mean\n(speech-level SD units)', fontsize=PLOT_LABEL_SIZE - 6, fontweight='bold')
ax.tick_params(labelsize=PLOT_TICK_SIZE); ax.grid(True, axis='y', alpha=0.25); ax.set_xticks(range(2015, 2026)); ax.legend(frameon=False, fontsize=PLOT_LEGEND_SIZE, loc='lower left')
fig.tight_layout(); save(fig, 'fig_llm_yearly_sd')

# ---- 3. other-directed share, two panels -----------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(20, 6.5))
NQ = {'parliament': pd.read_csv(os.path.join(RES, 'parliament', 'ts_quarterly_leaning.csv'))[['leaning', 'q', 'n']], 'twitter': pd.read_csv(os.path.join(RES, 'twitter', 'ts_quarterly_leaning.csv'))[['leaning', 'q', 'n']]}
for a, name, letter, title, unit in [(ax[0], 'parliament', 'A', 'Bundestag: sentences coded other-directed (Llama)', 'sentences'), (ax[1], 'twitter', 'B', 'Twitter: tweets coded other-directed (both models agree)', 'tweets')]:
    QL = pd.read_csv(os.path.join(RES, 'categories', f'{name}_share_quarterly_leaning.csv')); Q = pd.read_csv(os.path.join(RES, 'categories', f'{name}_share_quarterly.csv'))
    vcol = QL.columns[-1]; QL = QL.merge(NQ[name], on=['leaning', 'q'], how='left'); QL = QL[QL.n.fillna(0) >= 30]
    for lean in ['left', 'middle', 'right']:
        g = QL[QL.leaning == lean].sort_values('q')
        if len(g) == 0: continue
        a.plot(qdate(g.q), 100 * g[vcol], color=COLORS[lean], marker=MARK[lean], ms=6, lw=2.2, label=LABEL[lean], zorder=3)
    a.plot(qdate(Q.q), 100 * Q[Q.columns[1]], color='black', lw=2.6, alpha=0.8, label='All', zorder=4)
    decorate(a, letter, f'Other-directed {unit} (%)', afd=(name == 'parliament'), zero=False); a.set_title(title, fontsize=PLOT_LEGEND_SIZE + 2, loc='left', pad=6)
    a.set_xlabel('Quarter', fontsize=PLOT_LABEL_SIZE - 6, fontweight='bold'); a.set_ylim(0, 100 * max(QL[vcol].max(), Q[Q.columns[1]].max()) * 1.08)
h, l = ax[1].get_legend_handles_labels(); h += [plt.Line2D([], [], color='gray', ls='--', lw=1.2), plt.Line2D([], [], color='#D62728', lw=1.6)]; l += ['federal election', 'AfD enters the Bundestag']
fig.legend(h, l, frameon=False, fontsize=PLOT_LEGEND_SIZE, loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout(); save(fig, 'fig_other_directed')
