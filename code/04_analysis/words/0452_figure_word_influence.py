"""Figure: the words that drive the DDR EMI per party and period, with the quarterly EMI as lines.

Panel A Twitter, panel B Bundestag. In each panel the top row shows the quarterly mean EMI per party (lines) over
2015-2025 with the two periods of the paper (2015-2019, 2020-2025); the grid below lists, per party (rows) and
period (columns), the words with the largest word-shift contribution to the group's EMI relative to the platform
corpus: shift_w = (p_{w,g} - p_{w,ref}) * (s_w - s_bar), restricted to words the group uses more than the corpus
(p_g > p_ref). Evidence-type words (s_w > s_bar) push the EMI up (bars to the right), intuition-type words
(s_w < s_bar) push it down (bars to the left). Inputs from code/04_analysis/words/0451_word_influence.py.

  python code/04_analysis/words/0452_figure_word_influence.py                 -> results/word_influence/fig_word_influence.{pdf,png}
"""
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))   # German-EMI repository root: data in data/, outputs in results/
import os, sys, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = REPO_ROOT; D = os.path.join(ROOT, 'results', 'word_influence')
PLOT_LABEL_SIZE, PLOT_LEGEND_SIZE, PLOT_PANEL_SIZE, PLOT_TICK_SIZE = 24, 16, 20, 16
plt.rcParams.update({'font.family': 'Arial', 'axes.spines.top': False, 'axes.spines.right': False})
PARTY_COL = {'CDU/CSU': '#000000', 'SPD': '#E3000F', 'FDP': '#E0B400', 'Alliance 90/The Greens': '#46962B', 'The Left Party': '#BE3075', 'AFD': '#009EE0'}
SHORT = {'CDU/CSU': 'CDU/CSU', 'SPD': 'SPD', 'FDP': 'FDP', 'Alliance 90/The Greens': 'Greens', 'The Left Party': 'Left', 'AFD': 'AfD'}
PARTIES = ['CDU/CSU', 'SPD', 'FDP', 'Alliance 90/The Greens', 'The Left Party', 'AFD']
PERIODS = ['2015-2019', '2020-2025']
EV_COL, IU_COL = '#1B7837', '#762A83'                      # evidence-type / intuition-type words (colour-blind safe)
N_WORDS = int(os.environ.get('N_WORDS', 5))
# tokens that are forms of address, party names or speaker-header artefacts; excluded from the word lists
PROC = {'Damen', 'Herren', 'Liebe', 'Lieber', 'liebe', 'lieber', 'Kollegen', 'Kolleginnen', 'Kollege', 'Kollegin', 'Präsident', 'Präsidentin', 'Dank', 'Danke',
        'vielen', 'Vielen', 'Herr', 'Frau', 'geehrte', 'geehrter', 'verehrten', 'verehrte', 'Sehr', 'sehr', 'Bundestag', 'Bundestages', 'Drucksache', 'Abgeordnete',
        'Abgeordneten', 'Fraktion', 'Wort', 'Beifall', 'Zuruf', 'Vizepräsident', 'Vizepräsidentin', 'Rede', 'Redner', 'Dame', 'Herrn', 'Minister', 'Ministerin',
        'Bundesminister', 'Bundesministerin', 'Staatssekretär', 'Staatssekretärin', 'CDUCSU', 'SPD', 'FDP', 'GRÜNEN', 'LINKE', 'AfD', 'AFD', 'CDU', 'CSU',
        'Grünen', 'Linken', 'Union', 'Grüne', 'Linke', 'BÜNDNIS', 'Bündnis', 'DIE', 'Die', 'Demokraten', 'Sozialdemokraten', 'Liberalen', 'schließe', 'spd', 'fdp',
        'cdu', 'csu', 'afd', 'grünen', 'linke', 'gruene', 'spdbt', 'cducsubt', 'fdpbt', 'linksfraktion', 'AfDimBundestag', 'Bundesregierung', 'Regierung',
        'Koalition', 'Ampel', 'Opposition', 'Merkel', 'Scholz', 'Habeck', 'Lindner', 'Baerbock', 'Weidel', 'Gauland', 'Wagenknecht', 'Söder', 'Merz', 'Laschet', 'AKK',
        # English tokens of retweeted or quoted English-language content and Twitter handles of party accounts
        'the', 'The', 'for', 'For', 'from', 'From', 'and', 'And', 'with', 'With', 'that', 'That', 'this', 'This', 'you', 'You', 'are', 'Are', 'have',
        'Have', 'not', 'Not', 'our', 'Our', 'your', 'Your', 'will', 'Will', 'about', 'About', 'what', 'What', 'their', 'They', 'they', 'thank', 'Thank',
        'spdde', 'spdbt', 'cducsubt', 'fdpbt', 'gruene', 'dielinke', 'Linksfraktion', 'linksfraktion', 'AfDimBundestag', 'afdimbundestag', 'Bundestag',
        'Jewish', 'Orbit', 'born', 'Wahlrechtde', 'faznet', 'welt', 'gehts', 'BMVI', 'BMFSFJ', 'EinAugenschmaus'}


_names = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'parliament_nhb.csv'), usecols=['firstName', 'lastName'])
NAMES = set(tok for col in ('firstName', 'lastName') for v in _names[col].dropna().unique() for tok in str(v).split() if len(tok) > 3)   # MdB names (speaker headers)


def load(corpus):
    R = pd.read_csv(os.path.join(D, f'{corpus}_words.csv')); Q = pd.read_csv(os.path.join(D, f'{corpus}_quarterly.csv'))
    R = R.dropna(subset=['word']); R['word'] = R.word.astype(str)
    R = R[~R.word.isin(PROC) & ~R.word.isin(NAMES) & (R.word.str.len() > 3) & ~R.word.str.startswith('http') & ~R.word.str.contains(r'\d')]
    R = R[R.p > R.p_ref]                                   # words the group uses more than the platform corpus
    Q['date'] = pd.PeriodIndex(Q.quarter, freq='Q').to_timestamp()
    return R, Q


def words(R, party, period, n):
    sub = R[(R.party == party) & (R.period == period)]
    ev = sub[sub['shift'] > 0].nlargest(n, 'shift'); iu = sub[sub['shift'] < 0].nsmallest(n, 'shift')
    return ev, iu


def draw_panel(fig, gs_col, corpus, letter, title):
    R, Q = load(corpus)
    outer = gs_col.subgridspec(2, 1, height_ratios=[2.6, len(PARTIES) * 1.08], hspace=0.16)
    sub = outer[1].subgridspec(len(PARTIES), 2, hspace=0.2, wspace=0.08)
    # ---- line panel ----
    ax = fig.add_subplot(outer[0])
    for p in PARTIES:
        q = Q[(Q.party == p) & (Q.n >= 20)].sort_values('date')
        if q.empty: continue
        ax.plot(q.date, q.emi_mean, color=PARTY_COL[p], lw=2.2, marker='o', ms=3.5, label=SHORT[p], zorder=3)
    ax.axhline(0, color='grey', lw=0.8, zorder=1)
    ax.axvline(pd.Timestamp('2020-01-01'), color='black', lw=1.0, ls=':', zorder=2)
    ax.axvspan(pd.Timestamp('2015-01-01'), pd.Timestamp('2020-01-01'), color='0.93', zorder=0)
    ax.text(pd.Timestamp('2017-07-01'), 0.97, '2015–2019', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=PLOT_LEGEND_SIZE, color='0.35')
    ax.text(pd.Timestamp('2022-10-01'), 0.97, '2020–2025', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=PLOT_LEGEND_SIZE, color='0.35')
    ax.set_xlim(pd.Timestamp('2015-01-01'), pd.Timestamp('2025-07-01'))
    ax.set_ylabel('EMI (quarterly mean)', fontsize=PLOT_LABEL_SIZE - 4, fontweight='bold'); ax.tick_params(labelsize=PLOT_TICK_SIZE)
    ax.set_title(title, fontsize=PLOT_LABEL_SIZE, fontweight='bold', loc='left', pad=14)
    ax.text(-0.09, 1.12, letter, transform=ax.transAxes, fontsize=PLOT_PANEL_SIZE + 4, fontweight='bold', va='bottom', ha='left')
    ax.legend(frameon=False, fontsize=PLOT_LEGEND_SIZE, ncol=6, loc='lower right', bbox_to_anchor=(1.0, 1.0), handlelength=1.6, columnspacing=1.0)
    ax.grid(axis='y', alpha=0.25)
    # ---- word grid ----
    scale = 1000.0
    allv = []
    cells = {}
    for p in PARTIES:
        for per in PERIODS:
            ev, iu = words(R, p, per, N_WORDS); cells[(p, per)] = (ev, iu); allv += list(ev['shift'] * scale) + list(-iu['shift'] * scale)
    vmax = max(allv) * 1.02 if allv else 1
    for i, p in enumerate(PARTIES):
        for j, per in enumerate(PERIODS):
            axc = fig.add_subplot(sub[i, j]); ev, iu = cells[(p, per)]
            ys = np.arange(N_WORDS)[::-1]
            axc.barh(ys + 0.2, ev['shift'].to_numpy() * scale, height=0.38, color=EV_COL, alpha=0.9)
            axc.barh(ys - 0.2, iu['shift'].to_numpy() * scale, height=0.38, color=IU_COL, alpha=0.9)
            for y, (w, v) in zip(ys, zip(ev.word, ev['shift'] * scale)):
                axc.text(v + vmax * 0.02, y + 0.2, w, ha='left', va='center', fontsize=PLOT_TICK_SIZE - 3, color=EV_COL)
            for y, (w, v) in zip(ys, zip(iu.word, iu['shift'] * scale)):
                axc.text(v - vmax * 0.02, y - 0.2, w, ha='right', va='center', fontsize=PLOT_TICK_SIZE - 3, color=IU_COL)
            axc.axvline(0, color='black', lw=0.8)
            axc.set_xlim(-vmax * 1.9, vmax * 1.9); axc.set_ylim(-0.7, N_WORDS - 0.3)
            axc.set_yticks([]); axc.spines['left'].set_visible(False)
            if i < len(PARTIES) - 1: axc.set_xticks([]); axc.spines['bottom'].set_visible(False)
            else:
                axc.tick_params(labelsize=PLOT_TICK_SIZE - 2); axc.set_xlabel('Contribution to EMI (×10³)', fontsize=PLOT_LEGEND_SIZE, fontweight='bold')
            if i == 0: axc.set_title(per.replace('-', '–'), fontsize=PLOT_LEGEND_SIZE + 2, fontweight='bold', pad=6)
            if j == 0: axc.text(-0.04, 0.5, SHORT[p], transform=axc.transAxes, ha='right', va='center', fontsize=PLOT_TICK_SIZE + 2, fontweight='bold', color=PARTY_COL[p])
            axc.axvspan(-vmax * 1.9, 0, color='0.965', zorder=0)
    return ax


fig = plt.figure(figsize=(26, 24))
gs = fig.add_gridspec(1, 2, wspace=0.22, left=0.06, right=0.985, top=0.95, bottom=0.05)
draw_panel(fig, gs[0], 'parliament' if os.environ.get('TEST_PARL') else 'twitter', 'A', 'Twitter')
draw_panel(fig, gs[1], 'parliament', 'B', 'Bundestag')
handles = [Patch(color=EV_COL, label='evidence-type word (pushes EMI up)'), Patch(color=IU_COL, label='intuition-type word (pushes EMI down)')]
fig.legend(handles=handles, frameon=False, fontsize=PLOT_LEGEND_SIZE + 2, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.005))
for ext in ('pdf', 'png'):
    out = os.path.join(D, f'fig_word_influence.{ext}')
    try: fig.savefig(out, dpi=300, bbox_inches='tight')
    except PermissionError: fig.savefig(out.replace(f'.{ext}', f'_tmp.{ext}'), dpi=300, bbox_inches='tight'); print('locked, wrote _tmp')
print('written', D)
