"""Other-directed rhetoric: share over time and the EMI trend without it / with it as a control.

Categories (prompts/system_category.txt): 0 neutral, 1 own intuitive justification, 2 criticism of others'
reasoning, 3 credibility or lying accusation. "Other-directed" = 2 or 3. Two-model agreement is reported on the
10k sentence sample (both models) and on the tweets (both models); the full parliament corpus is coded by Llama only.

  python code/05_validation/llm/0558_sentence_categories.py twitter      60k tweets: share by quarter/leaning, breakpoint model with
                                                          and without other-directed tweets and with the flag as control
  python code/05_validation/llm/0558_sentence_categories.py parliament   all sentences of all speeches: per-speech share, quarterly share,
                                                          EMI re-scored on speeches without other-directed sentences
                                                          (parliament model), breakpoint models
Outputs: results/llm/categories/
"""
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))   # German-EMI repository root: data in data/, outputs in results/
import os, sys, json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from statsmodels.regression.mixed_linear_model import MixedLM
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = REPO_ROOT
OUTD = os.path.join(REPO_ROOT, 'results', 'llm', 'categories'); os.makedirs(OUTD, exist_ok=True)
MODELS = {'meta-llama_Llama-3.1-8B-Instruct': 'Llama', 'Qwen_Qwen2.5-7B-Instruct': 'Qwen'}
EMI = 'emi_w2vparliament'; LC = {'left': '#E66101', 'middle': '#4D4D4D', 'right': '#0072B2'}
BP = {'elec2017': 20173, 'elec2021': 20213, 'elec2025': 20251}
NUM = {}


def load_cat(prefix):
    out = None
    for tag, short in MODELS.items():
        f = os.path.join(REPO_ROOT, 'data', 'llm', 'output', f'{prefix}_category_{tag}.parquet')
        if not os.path.exists(f): continue
        d = pd.read_parquet(f); d = d.reset_index() if 'id' not in d.columns else d
        idc = [c for c in d.columns if c not in ('category', 'parse_error') and not c.endswith('_missing')][0]
        d = d[[idc, 'category']].rename(columns={idc: 'id', 'category': short}); d['id'] = d.id.astype(str); d = d.drop_duplicates('id')
        out = d if out is None else out.merge(d, on='id', how='outer')
    return out


def breakpoint_fit(d, y, actor, extra=''):
    m = d[[y, 'q', 'leaning', actor] + ([extra] if extra else [])].dropna().copy(); m['time_quarterly'] = m.q.dt.year * 10 + m.q.dt.quarter
    qm = m.groupby([actor, 'time_quarterly'])[y].mean().rename('_q').reset_index(); qm['lag1'] = qm.groupby(actor)['_q'].shift(1)
    m = m.merge(qm[[actor, 'time_quarterly', 'lag1']], on=[actor, 'time_quarterly']).dropna(); tm = m.time_quarterly.mean(); m['time_centered'] = m.time_quarterly - tm
    for k, bq in BP.items(): m[f'slope_{k}'] = np.maximum(0, m.time_centered - (bq - tm)); m[f'level_{k}'] = (m.time_quarterly >= bq).astype(float)
    m['leaning'] = pd.Categorical(m.leaning, categories=['middle', 'left', 'right']); m = m.rename(columns={y: 'yv'}).reset_index(drop=True)
    f = 'yv ~ time_centered + leaning + time_centered:leaning + ' + ' + '.join(f'slope_{k} + level_{k}' for k in BP) + ' + lag1' + (f' + {extra}' if extra else '')
    mod = MixedLM.from_formula(f, groups=m[actor], data=m)
    try: r = mod.fit(reml=False)
    except Exception: r = mod.fit(reml=False, method='powell')
    out = dict(n=int(r.nobs), sd=float(m.yv.std()))
    for k in BP: out[f'{k}_level'] = float(r.params[f'level_{k}']); out[f'{k}_se'] = float(r.bse[f'level_{k}'])
    if extra: out['control'] = float(r.params[extra]); out['control_se'] = float(r.bse[extra])
    return out


def share_figure(Q, QL, title, fname):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for lean, c in LC.items():
        if lean in QL.index.get_level_values(0):
            q = QL.loc[lean]; ax.plot(q.index.to_timestamp() + pd.Timedelta(days=45), q.values, color=c, lw=1.8, label=lean)
    ax.plot(Q.index.to_timestamp() + pd.Timedelta(days=45), Q.values, color='black', lw=2.4, ls='--', label='all')
    for dte in ['2017-10-24', '2021-09-26', '2025-02-23']: ax.axvline(pd.Timestamp(dte), color='#D62728' if dte.startswith('2017') else 'grey', lw=1.0 if dte.startswith('2017') else 0.7, ls='-' if dte.startswith('2017') else ':')
    ax.set_ylabel('share other-directed'); ax.set_title(title, loc='left', fontsize=10); ax.legend(frameon=False, ncol=4, fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUTD, fname), dpi=200, bbox_inches='tight'); plt.close(fig)


def run_twitter():
    M = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'llm', 'inputs', 'twitter_meta.csv'), dtype={'id': str, 'author_id': str}, parse_dates=['date']).merge(load_cat('twitter'), on='id')
    M['date'] = M.date.dt.tz_localize(None) if M.date.dt.tz is not None else M.date; M['q'] = M.date.dt.to_period('Q')
    M['od_L'] = (M.Llama >= 2).astype(int); M['od_Q'] = (M.Qwen >= 2).astype(int); M['od'] = ((M.od_L == 1) & (M.od_Q == 1)).astype(int); M['od_any'] = ((M.od_L == 1) | (M.od_Q == 1)).astype(int)
    print(f'tweets {len(M):,}; other-directed share: both {M.od.mean():.3f}, Llama {M.od_L.mean():.3f}, Qwen {M.od_Q.mean():.3f}; EMI other-directed {M[M.od==1][EMI].mean():+.3f} vs rest {M[M.od==0][EMI].mean():+.3f}')
    Q = M.groupby('q').od.mean(); QL = M.groupby(['leaning', 'q']).od.mean(); Q.to_csv(os.path.join(OUTD, 'twitter_share_quarterly.csv')); QL.to_csv(os.path.join(OUTD, 'twitter_share_quarterly_leaning.csv'))
    yr = M.groupby(M.date.dt.year)[['od', 'od_L', 'od_Q', EMI]].mean().round(3); print(yr.to_string())
    res = {'all tweets': breakpoint_fit(M, EMI, 'author_id'), 'without other-directed tweets': breakpoint_fit(M[M.od_any == 0], EMI, 'author_id'), 'with flag as control': breakpoint_fit(M, EMI, 'author_id', extra='od')}
    for k, v in res.items(): print(f"   {k:32s}: 2017 {v['elec2017_level']:+.3f} (se {v['elec2017_se']:.3f}) | 2021 {v['elec2021_level']:+.3f} | 2025 {v['elec2025_level']:+.3f}" + (f" | control {v['control']:+.3f} (se {v['control_se']:.3f})" if 'control' in v else '') + f" | n={v['n']:,}")
    pd.DataFrame(res).T.to_csv(os.path.join(OUTD, 'twitter_breakpoints.csv')); NUM['twitter'] = dict(share_both=float(M.od.mean()), by_year=yr.to_dict(), breakpoints=res)
    share_figure(Q, QL, 'Twitter: share of tweets both models code as other-directed', 'twitter_share.png')


def run_parliament():
    S = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'llm', 'inputs', 'sentences_meta.csv'), dtype={'id': str, 'speech_id': str}, parse_dates=['date'])
    C = load_cat('sentences'); S = S.merge(C, on='id', how='left'); lab = 'Llama' if 'Llama' in S else 'Qwen'
    S['od'] = (S[lab] >= 2).astype(float); S.loc[S[lab].isna(), 'od'] = np.nan; S['own_int'] = (S[lab] == 1).astype(float)
    print(f'sentences {len(S):,}, labelled {S[lab].notna().mean():.3f}; label shares {S[lab].value_counts(normalize=True).sort_index().round(3).to_dict()}')
    sp = S.groupby('speech_id').agg(n_sent=('id', 'size'), od_share=('od', 'mean'), own_int_share=('own_int', 'mean'), n_od=('od', 'sum')).reset_index()
    M = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'llm', 'inputs', 'corpus_meta.csv'), dtype={'id': str}, parse_dates=['date']).rename(columns={'id': 'speech_id'}).merge(sp, on='speech_id')
    M['date'] = M.date.dt.tz_localize(None) if M.date.dt.tz is not None else M.date; M['q'] = M.date.dt.to_period('Q'); M['politicianId'] = M.politicianId.astype(str).str.replace(r'\.0$', '', regex=True)
    print(f'speeches {len(M):,}; mean share other-directed {M.od_share.mean():.3f}; speeches with any {(M.n_od > 0).mean():.3f}')
    Q = M.groupby('q').od_share.mean(); QL = M.groupby(['leaning', 'q']).od_share.mean(); Q.to_csv(os.path.join(OUTD, 'parliament_share_quarterly.csv')); QL.to_csv(os.path.join(OUTD, 'parliament_share_quarterly_leaning.csv'))
    yr = M.groupby(M.date.dt.year)[['od_share', 'own_int_share', EMI]].mean().round(3); print(yr.to_string())
    byp = M.groupby('party')[['od_share', 'own_int_share']].mean().round(3); print(byp.to_string())
    # re-score EMI on the speech text without other-directed sentences
    sys.path.insert(0, os.path.join(REPO_ROOT, 'code', '03_scoring')); import importlib; se = importlib.import_module('0304_score_emi')
    txt = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'llm', 'inputs', 'sentences_input.csv'), dtype={'id': str}).merge(S[['id', 'speech_id', 'sent_idx', 'od']], on='id')
    keep = txt[txt.od != 1].sort_values(['speech_id', 'sent_idx']).groupby('speech_id').text.apply(' '.join).rename('text_clean_od').reset_index()
    kv = se.load_model(os.path.join(REPO_ROOT, 'data', 'parliament_model')); tok_stop = se.nltk_german_stopwords()
    sc = se.score_frame(keep.rename(columns={'text_clean_od': 'Text'}), 'Text', kv, tok_stop); sc = se.length_adjust_and_z(sc)
    M = M.merge(sc[['speech_id', 'emi']].rename(columns={'emi': 'emi_no_od'}), on='speech_id', how='left')
    # z-standardise the original EMI on the same speeches for comparability of level shifts (both z over this set)
    M['emi_z_same'] = (M[EMI] - M[EMI].mean()) / M[EMI].std()
    print(f'rescored speeches {M.emi_no_od.notna().sum():,}; corr with original EMI {M[[EMI, "emi_no_od"]].corr().iloc[0, 1]:.3f}')
    res = {'original EMI': breakpoint_fit(M, EMI, 'politicianId'), 'EMI without other-directed sentences': breakpoint_fit(M, 'emi_no_od', 'politicianId'),
           'original EMI + share as control': breakpoint_fit(M, EMI, 'politicianId', extra='od_share'), 'share other-directed as outcome': breakpoint_fit(M, 'od_share', 'politicianId')}
    for k, v in res.items(): print(f"   {k:38s}: 2017 {v['elec2017_level']:+.3f} (se {v['elec2017_se']:.3f}, {v['elec2017_level']/v['sd']:+.2f} sd) | 2021 {v['elec2021_level']:+.3f} | 2025 {v['elec2025_level']:+.3f}" + (f" | control {v['control']:+.3f} (se {v['control_se']:.3f})" if 'control' in v else '') + f" | n={v['n']:,}")
    pd.DataFrame(res).T.to_csv(os.path.join(OUTD, 'parliament_breakpoints.csv')); M.to_csv(os.path.join(OUTD, 'parliament_speech_level.csv'), index=False)
    NUM['parliament'] = dict(share=float(M.od_share.mean()), by_year=yr.to_dict(), by_party=byp.to_dict(), breakpoints=res)
    share_figure(Q, QL, 'Bundestag: share of sentences coded as other-directed (Llama)', 'parliament_share.png')


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if which in ('twitter', 'all'): run_twitter()
    if which in ('parliament', 'all'): run_parliament()
    json.dump(NUM, open(os.path.join(OUTD, f'numbers_{which}.json'), 'w'), indent=1, default=float)
