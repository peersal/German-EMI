"""Document-level validation of the dictionary EMI against human ratings: the 2024 Bundestag survey (1,142 chunks over
all eras), the 2026 Bundestag survey (300 chunks, 2015-2025) and the 2026 Twitter survey (300 tweets), pooled and by
era / period and party, with the LLM composite on the same items where available.

  python code/05_validation/document/0517_validation_2026.py export     build the anonymised ground-truth files in results/13_validation_2026/data/
                                               (no rater ids, no tweet texts) from the potato exports and the 2024 set
  python code/05_validation/document/0517_validation_2026.py            analysis: tables, numbers.json and figures in results/13_validation_2026/

The analysis reads only the anonymised data folder, so it runs unchanged from the public repository
(code/05_validation/document/0517_validation_2026.py with data_2026/ next to it).
"""
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))   # German-EMI repository root: data in data/, outputs in results/
import os, sys, json, numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = REPO_ROOT
OUT = os.environ.get('VAL_OUT', os.path.join(REPO_ROOT, 'results', 'validation_2026')); DATA = os.environ.get('VAL_DATA', os.path.join(HERE, 'data_2026'))
os.makedirs(DATA, exist_ok=True)
LIK = {'Stimme überhaupt nicht zu': -2, 'Stimme nicht zu': -1, 'Neutral': 0, 'Stimme zu': 1, 'Stimme voll und ganz zu': 2}
EV, IU = 'Rating.Evidenz basiert', 'Rating.Intuition basiert'
PLOT_LABEL_SIZE, PLOT_LEGEND_SIZE, PLOT_PANEL_SIZE, PLOT_TICK_SIZE = 18, 13, 18, 13
plt.rcParams.update({'font.family': 'Arial', 'axes.spines.top': False, 'axes.spines.right': False})
PARTY = {'AFD': 'AfD', 'AfD': 'AfD', 'Alliance 90/The Greens': 'Greens', 'BÜNDNIS 90/DIE GRÜNEN': 'Greens', 'Bündnis 90/Die Grünen': 'Greens', 'GRÜNE': 'Greens', 'Grüne': 'Greens',
         'The Left Party': 'Left', 'DIE LINKE': 'Left', 'DIE LINKE / PDS': 'Left', 'PDS': 'Left', 'Die Linke': 'Left', 'CDU/CSU': 'CDU/CSU', 'CDU': 'CDU/CSU', 'CSU': 'CDU/CSU', 'SPD': 'SPD', 'FDP': 'FDP'}
PARTY_ORDER = ['CDU/CSU', 'SPD', 'FDP', 'Greens', 'Left', 'AfD']
ERA_ORDER = ['Kaiserreich 1867–1918', 'Weimar 1919–1932', 'NS 1933–1945', 'Bonn 1949–1990', 'Berlin 1991–2014', '2015–2025']
TAGS = {'Llama': 'meta-llama_Llama-3.1-8B-Instruct', 'Qwen': 'Qwen_Qwen2.5-7B-Instruct'}


# ====================================================================================================== export
def export():
    SURVEY = os.path.join(HERE, 'survey'); LLMD = os.path.join(HERE, 'llm')
    # --- 2024 Bundestag survey (texts are public plenary speeches; rater ids were never in the item table)
    v = pd.read_csv(os.path.join(REPO_ROOT, 'results', 'revision', '07_validation_auc', 'validation_instances.csv'), dtype={'id': str, 'instance_id': str})
    gm = pd.read_csv(os.path.join(LLMD, 'inputs', 'groundtruth_meta.csv'), dtype={'id': str, 'instance_id': str})[['instance_id', 'text']]
    v = v.merge(gm, on='instance_id', how='left')
    keep = ['instance_id', 'text', 'party', 'era', 'decade', 'year', 'date', 'electoralTerm', 'chunk_length', 'n_raters', 'ev_mean', 'iu_mean', 'ev_label', 'iu_label', 'emi_label',
            'evidence_score', 'intuition_score', 'evidence_z', 'intuition_z', 'emi']
    v[keep].rename(columns={'instance_id': 'id'}).to_csv(os.path.join(DATA, 'groundtruth_parliament_2024.csv'), index=False)
    llm24 = llm_composite(os.path.join(LLMD, 'output'), '', 'gt_')
    if llm24 is not None: llm24.rename_axis('id').reset_index().assign(id=lambda d: d.id.str.replace('gt_', '', regex=False)).to_csv(os.path.join(DATA, 'llm_parliament_2024.csv'), index=False)
    # --- 2026 surveys: items (aggregated) and anonymised ratings
    for corpus, P, textcols in [('parliament', os.path.join(SURVEY, 'parliament_results', 'pilot'), ['text']), ('twitter', os.path.join(SURVEY, 'twitter_results', 'twitter', 'pilot'), [])]:
        key = pd.read_csv(os.path.join(SURVEY, corpus, 'data_files', 'survey_sample_key.csv'), dtype={'id': str, 'tweet_id': str, 'author_id': str, 'politicianId': str})
        a = pd.read_csv(os.path.join(P, 'exports', 'csv', 'annotations.csv'), encoding='utf-8'); a = a[~a.instance_id.str.startswith('gold')].copy()
        a['ev'] = a[EV].map(LIK); a['iu'] = a[IU].map(LIK)
        rid = {u: f'r{i + 1:03d}' for i, u in enumerate(sorted(a.user_id.unique()))}          # anonymised rater ids (no Prolific ids)
        a['rater'] = a.user_id.map(rid)
        a[['instance_id', 'rater', 'ev', 'iu']].rename(columns={'instance_id': 'id', 'ev': 'evidence_rating', 'iu': 'intuition_rating'}).sort_values(['id', 'rater']).to_csv(os.path.join(DATA, f'ratings_{corpus}_2026.csv'), index=False)
        h = a.groupby('instance_id').agg(ev_mean=('ev', 'mean'), iu_mean=('iu', 'mean'), n_raters=('ev', 'size')).reset_index()
        cols = ['id'] + textcols + [c for c in ['speech_id', 'chunk_idx', 'tweet_id', 'party', 'leaning', 'date', 'electoralTerm', 'n_words', 'emi', 'speech_emi'] if c in key.columns]
        k = key[cols].merge(h, left_on='id', right_on='instance_id', how='left').drop(columns=['instance_id'])
        k.to_csv(os.path.join(DATA, f'groundtruth_{corpus}_2026.csv'), index=False)
        print(corpus, len(k), 'items,', len(a), 'ratings,', len(rid), 'raters')
    llm26 = llm_composite(os.path.join(LLMD, 'output'), 'survey_', '')
    if llm26 is not None: llm26.rename_axis('id').reset_index().to_csv(os.path.join(DATA, 'llm_2026.csv'), index=False)
    open(os.path.join(DATA, 'README.md'), 'w', encoding='utf-8', newline='\n').write(
        '# Ground truth of the document-level validation surveys\n\n'
        '| File | Content |\n|---|---|\n'
        '| `groundtruth_parliament_2024.csv` | 1,142 Bundestag chunks of the 2024 survey (all eras): text, party, era, mean human evidence / intuition rating (−2…+2), majority labels, dictionary scores |\n'
        '| `groundtruth_parliament_2026.csv` | 300 Bundestag chunks (2015–2025) of the 2026 survey: text, party, date, chunk and whole-speech dictionary EMI, mean human ratings, number of raters |\n'
        '| `ratings_parliament_2026.csv` | the single ratings of the 2026 Bundestag survey (anonymised rater id, evidence and intuition rating) |\n'
        '| `groundtruth_twitter_2026.csv` | 300 tweets of the 2026 survey: tweet id, party, date, dictionary EMI, mean human ratings (no tweet text, in line with the X Developer Agreement) |\n'
        '| `ratings_twitter_2026.csv` | the single ratings of the 2026 Twitter survey (anonymised rater id) |\n'
        '| `llm_parliament_2024.csv`, `llm_2026.csv` | ratings of Llama 3.1 8B and Qwen 2.5 7B on the same items (evidence, intuition, 0–4) |\n\n'
        'Rater ids are sequential codes; Prolific ids, demographics and timing data are not included.\n')
    print('exported to', DATA)


def llm_composite(outdir, prefix, idprefix):
    frames = {}
    for name, tag in TAGS.items():
        f = os.path.join(outdir, f'{prefix}rating_{tag}.parquet')
        if not os.path.exists(f): return None
        d = pd.read_parquet(f); d.index = d.index.astype(str)
        if idprefix: d = d[d.index.str.startswith(idprefix)]
        frames[name] = d[['evidenz_basiert', 'intuition_basiert']].rename(columns={'evidenz_basiert': f'ev_{name}', 'intuition_basiert': f'iu_{name}'})
    D = pd.concat(frames.values(), axis=1); D['llm_emi'] = ((D.ev_Llama - D.iu_Llama) + (D.ev_Qwen - D.iu_Qwen)) / 2
    return D


# ====================================================================================================== analysis
def auc_ci(score, label, n_boot=1000, seed=0):
    m = pd.DataFrame({'s': score, 'l': label}).dropna(); m = m[m.l != 0]
    y = (m.l > 0).astype(int).to_numpy(); s = m.s.to_numpy()
    if len(m) < 10 or y.min() == y.max(): return np.nan, np.nan, np.nan, len(m)
    a = roc_auc_score(y, s); g = np.random.default_rng(seed); bs = []
    for _ in range(n_boot):
        i = g.integers(0, len(y), len(y))
        if y[i].min() != y[i].max(): bs.append(roc_auc_score(y[i], s[i]))
    return a, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(m)


def rho(x, y):
    m = pd.DataFrame({'x': x, 'y': y}).dropna()
    return stats.spearmanr(m.x, m.y).correlation if len(m) >= 10 else np.nan


def icc1(wide):
    k = wide.notna().sum(axis=1).mean(); gm = wide.stack().mean()
    msb = (wide.notna().sum(axis=1) * (wide.mean(axis=1) - gm) ** 2).sum() / (len(wide) - 1)
    msw = ((wide.sub(wide.mean(axis=1), axis=0)) ** 2).stack().sum() / (wide.notna().sum().sum() - len(wide))
    return (msb - msw) / (msb + (k - 1) * msw), (msb - msw) / msb, k


def analyze():
    p24 = pd.read_csv(os.path.join(DATA, 'groundtruth_parliament_2024.csv'), dtype={'id': str}); p24['set'] = 'Bundestag 2024'
    p26 = pd.read_csv(os.path.join(DATA, 'groundtruth_parliament_2026.csv'), dtype={'id': str}); p26['set'] = 'Bundestag 2026'
    t26 = pd.read_csv(os.path.join(DATA, 'groundtruth_twitter_2026.csv'), dtype={'id': str}); t26['set'] = 'Twitter 2026'
    for d in (p24, p26, t26):
        d['human_emi'] = d.ev_mean - d.iu_mean; d['party_s'] = d.party.map(PARTY).fillna('Other')
    p26['date'] = pd.to_datetime(p26.date, utc=True); p26['year'] = p26.date.dt.year; p26['era'] = '2015–2025'
    p26['period'] = pd.cut(p26.year, [2014, 2017, 2021, 2026], labels=['2015–2017', '2018–2021', '2022–2025'])
    t26['date'] = pd.to_datetime(t26.date, utc=True); t26['year'] = t26.date.dt.year
    t26['period'] = pd.cut(t26.year, [2014, 2017, 2020, 2022, 2026], labels=['2015–2017', '2018–2020', '2021–2022', '2023–2025'])
    # LLM composites
    l24 = pd.read_csv(os.path.join(DATA, 'llm_parliament_2024.csv'), dtype={'id': str}).set_index('id') if os.path.exists(os.path.join(DATA, 'llm_parliament_2024.csv')) else None
    l26 = pd.read_csv(os.path.join(DATA, 'llm_2026.csv'), dtype={'id': str}).set_index('id') if os.path.exists(os.path.join(DATA, 'llm_2026.csv')) else None
    for d, l in ((p24, l24), (p26, l26), (t26, l26)):
        d['llm_emi'] = d.id.map(l[~l.index.duplicated()].llm_emi) if l is not None else np.nan
    pool = pd.concat([p24, p26], ignore_index=True)
    sets = {'Bundestag 2024': p24, 'Bundestag 2026': p26, 'Bundestag pooled': pool, 'Twitter 2026': t26}
    NUM = {}; rows = []

    def add(setname, group, d, extra=''):
        for meas, col in [('dictionary EMI', 'emi'), ('LLM composite', 'llm_emi')] + ([('evidence score', 'evidence_z'), ('intuition score', 'intuition_z')] if 'evidence_z' in d.columns and setname != 'Bundestag pooled' else []):
            if col not in d.columns or d[col].isna().all(): continue
            lab = d.human_emi if col not in ('evidence_z', 'intuition_z') else (d.ev_mean if col == 'evidence_z' else d.iu_mean)
            a, lo, hi, n = auc_ci(d[col], np.sign(lab)); r = rho(d[col], lab)
            rows.append(dict(set=setname, group=group, measure=meas, n=n, pos_rate=float((lab > 0).mean()), auc=a, lo=lo, hi=hi, rho=r))

    # ---- pooled, by era / period, by party
    for name, d in sets.items():
        add(name, 'all', d)
        if name == 'Bundestag pooled':
            for era, g in d.groupby('era'): add(name, era, g)
            for party, g in d.groupby('party_s'): add(name, party, g)
        if name in ('Bundestag 2024',):
            for era, g in d.groupby('era'): add(name, era, g)
            for party, g in d.groupby('party_s'): add(name, party, g)
        if name in ('Bundestag 2026', 'Twitter 2026'):
            for per, g in d.groupby('period', observed=True): add(name, str(per), g)
            for party, g in d.groupby('party_s'): add(name, party, g)
    T = pd.DataFrame(rows); T.to_csv(os.path.join(OUT, 'auc_tables.csv'), index=False)
    # ---- the 2024 majority-label AUC as published (for continuity)
    m24 = p24.dropna(subset=['emi_label']); NUM['auc_2024_majority_label'] = float(roc_auc_score(m24.emi_label.astype(int), m24.emi))
    # ---- agreement / ICC for the 2026 sets
    for corpus, d in (('parliament', p26), ('twitter', t26)):
        R = pd.read_csv(os.path.join(DATA, f'ratings_{corpus}_2026.csv'), dtype={'id': str}); R['emi'] = R.evidence_rating - R.intuition_rating
        res = {}
        for c in ['evidence_rating', 'intuition_rating', 'emi']:
            w = R.pivot_table(index='id', columns='rater', values=c); i1, ik, k = icc1(w); res[c] = dict(icc1=float(i1), icc1k=float(ik), k=float(k))
        modal = {c: float(R.groupby('id')[c].agg(lambda s: s.value_counts(normalize=True).iloc[0]).mean()) for c in ['evidence_rating', 'intuition_rating']}
        binar = {c: float(R.assign(b=(R[c] >= 1).astype(int)).groupby('id').b.agg(lambda s: s.value_counts(normalize=True).iloc[0]).mean()) for c in ['evidence_rating', 'intuition_rating']}
        NUM[f'agreement_{corpus}_2026'] = dict(n_items=int(R.id.nunique()), n_raters=int(R.rater.nunique()), n_ratings=int(len(R)), icc=res, modal_agreement=modal, binary_agreement=binar,
                                              within_response_r=float(R.evidence_rating.corr(R.intuition_rating)), share_evidence=float((d.human_emi > 0).mean()), share_intuition=float((d.human_emi < 0).mean()))
    NUM['n'] = {k: int(len(v)) for k, v in sets.items()}
    NUM['rho_llm_vs_dictionary'] = {k: float(rho(v.llm_emi, v.emi)) for k, v in sets.items() if v.llm_emi.notna().any()}
    json.dump(NUM, open(os.path.join(OUT, 'numbers.json'), 'w'), indent=1)
    pd.set_option('display.width', 250); print(T.round(3).to_string(index=False)); print(json.dumps(NUM, indent=1)[:3000])

    # ====================================================================================== figures
    COL = {'dictionary EMI': '#0072B2', 'LLM composite': '#D55E00', 'evidence score': '#009E73', 'intuition score': '#CC79A7'}
    # --- Figure 1: ROC curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (name, d, letter) in zip(axes, [('Bundestag 2024', p24, 'A'), ('Bundestag 2026', p26, 'B'), ('Twitter 2026', t26, 'C')]):
        for meas, col, lab in [('dictionary EMI', 'emi', 'human_emi'), ('LLM composite', 'llm_emi', 'human_emi'), ('evidence score', 'evidence_z', 'ev_mean'), ('intuition score', 'intuition_z', 'iu_mean')]:
            if col not in d.columns or d[col].isna().all(): continue
            m = d[[col, lab]].dropna(); m = m[np.sign(m[lab]) != 0]
            fpr, tpr, _ = roc_curve((m[lab] > 0).astype(int), m[col]); a = roc_auc_score((m[lab] > 0).astype(int), m[col])
            ax.plot(fpr, tpr, color=COL[meas], lw=2.4 if meas in ('dictionary EMI', 'LLM composite') else 1.4, ls='-' if meas != 'LLM composite' else '--', label=f'{meas} (AUC {a:.2f})')
        ax.plot([0, 1], [0, 1], color='grey', lw=0.8, ls=':')
        n = int((np.sign(d.human_emi) != 0).sum()); ax.set_title(f'{name} (n = {n:,})', fontsize=PLOT_LABEL_SIZE - 2, fontweight='bold', loc='left')
        ax.set_xlabel('False positive rate', fontsize=PLOT_LABEL_SIZE - 3, fontweight='bold'); ax.tick_params(labelsize=PLOT_TICK_SIZE); ax.legend(frameon=False, fontsize=PLOT_LEGEND_SIZE - 1, loc='lower right')
        ax.text(-0.12, 1.04, letter, transform=ax.transAxes, fontsize=PLOT_PANEL_SIZE, fontweight='bold', va='bottom')
    axes[0].set_ylabel('True positive rate', fontsize=PLOT_LABEL_SIZE - 3, fontweight='bold'); fig.tight_layout()
    for ext in ('pdf', 'png'): fig.savefig(os.path.join(OUT, f'fig_validation_roc.{ext}'), dpi=300, bbox_inches='tight')
    # --- Figure 2: AUC over time and by party
    fig = plt.figure(figsize=(21, 11)); gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.5, wspace=0.55)

    def forest(ax, sub, order, title, letter, xlabel=''):
        sub = sub[sub.group.isin(order)]; ys = {g: i for i, g in enumerate(order[::-1])}
        for meas, off in [('dictionary EMI', 0.17), ('LLM composite', -0.17)]:
            s = sub[sub.measure == meas]
            if s.empty: continue
            ax.errorbar(s.auc, [ys[g] + off for g in s.group], xerr=[s.auc - s.lo, s.hi - s.auc], fmt='o', color=COL[meas], capsize=3, ms=7, lw=1.6, label=meas)
        labels = []
        for g in ys:
            n = sub[(sub.group == g) & (sub.measure == 'dictionary EMI')].n; labels.append(f'{g} (n = {int(n.iloc[0])})' if len(n) else g)
        ax.axvline(0.5, color='grey', lw=0.8, ls=':'); ax.set_yticks(list(ys.values())); ax.set_yticklabels(labels, fontsize=PLOT_TICK_SIZE)
        ax.set_xlim(0.3, 1.0); ax.tick_params(labelsize=PLOT_TICK_SIZE); ax.grid(axis='x', alpha=0.25); ax.set_title(title, fontsize=PLOT_LABEL_SIZE - 2, fontweight='bold', loc='left')
        if xlabel: ax.set_xlabel(xlabel, fontsize=PLOT_LABEL_SIZE - 3, fontweight='bold')
        ax.text(-0.3, 1.06, letter, transform=ax.transAxes, fontsize=PLOT_PANEL_SIZE, fontweight='bold', va='bottom')
    ax1 = fig.add_subplot(gs[0, 0]); forest(ax1, T[T.set == 'Bundestag pooled'], ERA_ORDER, 'Bundestag, by era (2024 and 2026 surveys pooled)', 'A')
    ax2 = fig.add_subplot(gs[0, 1]); per = T[(T.set == 'Bundestag 2026') & T.group.str.contains('–')]; per2 = T[(T.set == 'Twitter 2026') & T.group.str.contains('–')]
    per = pd.concat([per.assign(group='Bundestag ' + per.group), per2.assign(group='Twitter ' + per2.group)])
    forest(ax2, per, list(per[per.measure == 'dictionary EMI'].group), '2015–2025 by period (2026 surveys)', 'B')
    ax3 = fig.add_subplot(gs[1, 0]); forest(ax3, T[T.set == 'Bundestag pooled'], PARTY_ORDER, 'Bundestag, by party (2024 and 2026 surveys pooled)', 'C', 'AUC against the human label (95% bootstrap CI)')
    ax4 = fig.add_subplot(gs[1, 1]); forest(ax4, T[T.set == 'Twitter 2026'], PARTY_ORDER, 'Twitter, by party (2026 survey)', 'D', 'AUC against the human label (95% bootstrap CI)')
    ax1.legend(frameon=False, fontsize=PLOT_LEGEND_SIZE, loc='lower left')
    for ext in ('pdf', 'png'): fig.savefig(os.path.join(OUT, f'fig_validation_auc_groups.{ext}'), dpi=300, bbox_inches='tight')
    demographics()
    print('written', OUT)


# ====================================================================================================== demographics
def demographics():
    """Aggregated annotator demographics of the 2026 surveys from the Prolific exports (local only; the aggregated table
    demographics_2026.csv and the figure are what gets published). Participants are counted if their submission was
    approved or awaits review and they have at least one rating."""
    import glob
    SURVEY = os.path.join(HERE, 'survey'); rows = []; ages = {}
    for corpus, label in (('parliament', 'Bundestag 2026'), ('twitter', 'Twitter 2026')):
        files = glob.glob(os.path.join(SURVEY, f'{corpus}_results', 'prolific_demographic_export_*.csv'))
        if not files: return
        d = pd.read_csv(files[0]); R = pd.read_csv(os.path.join(DATA, f'ratings_{corpus}_2026.csv'), dtype={'id': str})
        d = d[d.Status.isin(['APPROVED', 'AWAITING REVIEW'])].copy()
        d['Age'] = pd.to_numeric(d.Age, errors='coerce'); d = d[d.Sex.ne('CONSENT_REVOKED')]
        n = len(d); ages[label] = d.Age.dropna()
        rows.append(dict(survey=label, n_participants=n, n_raters_in_export=int(R.rater.nunique()), age_mean=d.Age.mean(), age_sd=d.Age.std(), age_min=d.Age.min(), age_max=d.Age.max(),
                         female=int((d.Sex == 'Female').sum()), male=int((d.Sex == 'Male').sum()), other_sex=int((~d.Sex.isin(['Female', 'Male'])).sum()),
                         residence_germany=int((d['Country of residence'] == 'Germany').sum()), nationality_german=int((d.Nationality == 'Germany').sum()), language_german=int((d.Language == 'German').sum()),
                         students=int((d['Student status'] == 'Yes').sum()), employed_full_time=int((d['Employment status'] == 'Full-Time').sum()), employed_part_time=int((d['Employment status'] == 'Part-Time').sum()),
                         time_median_min=float(d['Time taken'].median() / 60), time_min_min=float(d['Time taken'].min() / 60), time_max_min=float(d['Time taken'].max() / 60)))
    D = pd.DataFrame(rows); D.to_csv(os.path.join(OUT, 'demographics_2026.csv'), index=False); print(D.round(1).T.to_string())
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    bins = np.arange(15, 80, 5)
    for label, col in zip(ages, ['#4D4D4D', '#0072B2']):
        axes[0].hist(ages[label], bins=bins, alpha=0.55, color=col, label=f'{label} (n = {len(ages[label])}, mean {ages[label].mean():.1f})')
    axes[0].set_xlabel('Age (years)', fontsize=PLOT_LABEL_SIZE - 4, fontweight='bold'); axes[0].set_ylabel('Participants', fontsize=PLOT_LABEL_SIZE - 4, fontweight='bold'); axes[0].legend(frameon=False, fontsize=PLOT_LEGEND_SIZE - 2)
    x = np.arange(2); w = 0.38
    axes[1].bar(x - w / 2, D.female, w, color='#E66101', label='female'); axes[1].bar(x + w / 2, D.male, w, color='#0072B2', label='male')
    axes[1].set_xticks(x); axes[1].set_xticklabels(D.survey, fontsize=PLOT_TICK_SIZE); axes[1].set_ylabel('Participants', fontsize=PLOT_LABEL_SIZE - 4, fontweight='bold'); axes[1].legend(frameon=False, fontsize=PLOT_LEGEND_SIZE - 2)
    cats = ['students', 'employed_full_time', 'employed_part_time']; x = np.arange(len(cats)); w = 0.38
    for i, (_, r) in enumerate(D.iterrows()):
        axes[2].bar(x + (i - 0.5) * w, [r[c] / r.n_participants * 100 for c in cats], w, color=['#4D4D4D', '#0072B2'][i], label=r.survey)
    axes[2].set_xticks(x); axes[2].set_xticklabels(['Student', 'Full-time', 'Part-time'], fontsize=PLOT_TICK_SIZE); axes[2].set_ylabel('Share of participants (%)', fontsize=PLOT_LABEL_SIZE - 4, fontweight='bold'); axes[2].legend(frameon=False, fontsize=PLOT_LEGEND_SIZE - 2)
    for ax, letter in zip(axes, 'ABC'):
        ax.tick_params(labelsize=PLOT_TICK_SIZE); ax.text(-0.14, 1.04, letter, transform=ax.transAxes, fontsize=PLOT_PANEL_SIZE, fontweight='bold', va='bottom')
    fig.tight_layout()
    for ext in ('pdf', 'png'): fig.savefig(os.path.join(OUT, f'fig_annotator_demographics_2026.{ext}'), dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    export() if (len(sys.argv) > 1 and sys.argv[1] == 'export') else analyze()
