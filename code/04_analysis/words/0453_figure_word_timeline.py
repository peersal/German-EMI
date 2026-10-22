"""Timeline figure: quarterly EMI per leaning with callouts of the most influential evidence and intuition words
in each two-year window (A Twitter, B Bundestag). Inputs: results/word_influence/<corpus>_words_leaning_biennial.csv
and <corpus>_quarterly_leaning_biennial.csv (code/04_analysis/words/0451_word_influence.py with WI_GROUP=leaning WI_WINDOWS=biennial).
Word ranking as in figure_word_influence.py: word-shift contribution (p_group - p_corpus) * (s_w - s_bar) among words
the group uses more than the platform corpus; evidence-type words push the EMI up, intuition-type words down.

  python code/04_analysis/words/0453_figure_word_timeline.py        -> results/word_influence/fig_word_timeline.{pdf,png}
"""
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))   # German-EMI repository root: data in data/, outputs in results/
import os, re, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = REPO_ROOT; D = os.path.join(ROOT, 'results', 'word_influence')
PLOT_LABEL_SIZE, PLOT_LEGEND_SIZE, PLOT_PANEL_SIZE, PLOT_TICK_SIZE = 22, 16, 22, 16
plt.rcParams.update({'font.family': 'Arial', 'axes.spines.top': False, 'axes.spines.right': False})
COLORS = {'left': '#E66101', 'middle': '#4D4D4D', 'right': '#0072B2'}; LAB = {'left': 'Left', 'middle': 'Centre', 'right': 'Right'}
EV_COL, IU_COL = '#1B7837', '#762A83'
N_WORDS = int(os.environ.get('N_WORDS', 4))
WINDOWS = [('2015-2016', '2015-01-01', '2017-01-01'), ('2017-2018', '2017-01-01', '2019-01-01'), ('2019-2020', '2019-01-01', '2021-01-01'),
           ('2021-2022', '2021-01-01', '2023-01-01'), ('2023-2025', '2023-01-01', '2025-07-01')]
# exclusion lists shared with figure_word_influence.py (forms of address, party names, handles, English tokens, MdB names)
_src = open(os.path.join(REPO_ROOT, 'code/04_analysis/words/0452_figure_word_influence.py'), encoding='utf-8').read()
exec(re.search(r'^PROC = \{.*?\}\n', _src, re.S | re.M).group(0))
_names = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'parliament_nhb.csv'), usecols=['firstName', 'lastName'])
NAMES = set(tok for col in ('firstName', 'lastName') for v in _names[col].dropna().unique() for tok in str(v).split() if len(tok) > 3)


def load(corpus):
    R = pd.read_csv(os.path.join(D, f'{corpus}_words_leaning_biennial.csv')); Q = pd.read_csv(os.path.join(D, f'{corpus}_quarterly_leaning_biennial.csv'))
    R = R.dropna(subset=['word']); R['word'] = R.word.astype(str)
    R = R[~R.word.isin(PROC) & ~R.word.isin(NAMES) & (R.word.str.len() > 3) & ~R.word.str.startswith('http') & ~R.word.str.contains(r'\d')]
    R = R[R.p > R.p_ref]
    Q['date'] = pd.PeriodIndex(Q.quarter, freq='Q').to_timestamp() + pd.Timedelta(days=45)
    return R, Q


def top_words(R, lean, win, n):
    sub = R[(R.party == lean) & (R.period == win)]
    return sub[sub['shift'] > 0].nlargest(n, 'shift').word.tolist(), sub[sub['shift'] < 0].nsmallest(n, 'shift').word.tolist()


def draw_panel(fig, gs_cell, corpus, letter, title):
    R, Q = load(corpus)
    sub = gs_cell.subgridspec(3, 1, hspace=0.12)
    axes = []
    for i, lean in enumerate(['left', 'middle', 'right']):
        ax = fig.add_subplot(sub[i], sharex=axes[0] if axes else None); axes.append(ax)
        q = Q[(Q.party == lean) & (Q.n >= 20)].sort_values('date')
        ax.plot(q.date, q.emi_mean, color=COLORS[lean], lw=2.4, marker='o', ms=3.5, zorder=3)
        ax.axhline(0, color='grey', lw=0.8, zorder=1)
        for w0, a, b in WINDOWS:
            ax.axvline(pd.Timestamp(a), color='0.85', lw=0.8, zorder=0)
        lo, hi = q.emi_mean.min(), q.emi_mean.max(); span = max(hi - lo, 0.3)
        ax.set_ylim(lo - 0.15 * span, hi + 2.1 * span)                     # headroom for the callouts
        ax.set_yticks([tk for tk in ax.get_yticks() if lo - 0.15 * span <= tk <= hi + 0.3 * span])   # no tick labels in the callout headroom
        ax.text(0.005, 0.97, LAB[lean], transform=ax.transAxes, ha='left', va='top', fontsize=PLOT_LEGEND_SIZE + 2, fontweight='bold', color=COLORS[lean])
        ax.tick_params(labelsize=PLOT_TICK_SIZE); ax.grid(axis='y', alpha=0.2)
        if i < 2: plt.setp(ax.get_xticklabels(), visible=False)
        # callouts
        for w0, a, b in WINDOWS:
            ev, iu = top_words(R, lean, w0, N_WORDS)
            if not ev and not iu: continue
            mid = pd.Timestamp(a) + (pd.Timestamp(b) - pd.Timestamp(a)) / 2
            qq = q[(q.date >= pd.Timestamp(a)) & (q.date < pd.Timestamp(b))]
            if qq.empty: continue
            y_line = float(np.interp(mid.value, qq.date.astype('int64'), qq.emi_mean)) if len(qq) > 1 else float(qq.emi_mean.iloc[0])
            y_box = hi + 0.75 * span
            ax.annotate('', xy=(mid, y_line), xytext=(mid, hi + 0.72 * span), arrowprops=dict(arrowstyle='-|>', color='0.3', lw=1.2, shrinkA=0, shrinkB=3), zorder=4)
            dx = pd.Timedelta(days=28)
            ax.text(mid - dx, y_box, 'E\n' + '\n'.join(ev), ha='right', va='bottom', fontsize=PLOT_TICK_SIZE - 2, color=EV_COL, fontweight='bold', zorder=5, linespacing=1.15,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85))
            ax.text(mid + dx, y_box, 'I\n' + '\n'.join(iu), ha='left', va='bottom', fontsize=PLOT_TICK_SIZE - 2, color=IU_COL, fontweight='bold', zorder=5, linespacing=1.15,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85))
    axes[0].set_xlim(pd.Timestamp('2015-01-01'), pd.Timestamp('2025-07-01'))
    axes[0].set_title(title, fontsize=PLOT_LABEL_SIZE, fontweight='bold', loc='left', pad=12)
    axes[0].text(-0.035, 1.12, letter, transform=axes[0].transAxes, fontsize=PLOT_PANEL_SIZE + 4, fontweight='bold', va='bottom', ha='left')
    axes[1].set_ylabel('EMI (quarterly mean)', fontsize=PLOT_LABEL_SIZE - 2, fontweight='bold')
    for ax in axes:
        ax.xaxis.set_major_locator(matplotlib.dates.YearLocator()); ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
    return axes


fig = plt.figure(figsize=(26, 30))
gs = fig.add_gridspec(2, 1, hspace=0.14, left=0.05, right=0.99, top=0.96, bottom=0.05)
draw_panel(fig, gs[0], 'parliament' if os.environ.get('TEST_PARL') else 'twitter', 'A', 'Twitter')
draw_panel(fig, gs[1], 'parliament', 'B', 'Bundestag')
fig.legend(handles=[Patch(color=EV_COL, label='E: evidence-type words with the largest contribution (push EMI up)'),
                    Patch(color=IU_COL, label='I: intuition-type words with the largest contribution (push EMI down)')],
           frameon=False, fontsize=PLOT_LEGEND_SIZE + 2, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.005))
for ext in ('pdf', 'png'):
    out = os.path.join(D, f'fig_word_timeline.{ext}')
    try: fig.savefig(out, dpi=200, bbox_inches='tight')
    except PermissionError: fig.savefig(out.replace(f'.{ext}', f'_tmp.{ext}'), dpi=200, bbox_inches='tight'); print('locked, wrote _tmp')
print('written', D)
