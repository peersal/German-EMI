"""Which dimension would improve validity on the Left Party chunks? Re-score the 1,142 validation chunks with the
parliament model under dictionary variants (full / without deception-accusation / without deception+honesty /
intuition core only) and test a numeral-density cue for evidence. AUCs pooled, Left/PDS, other post-1949 parties.
Output: results/revision/07_validation_auc/linke_diagnostics.csv
"""
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))   # German-EMI repository root: data in data/, outputs in results/
import os, re, sys, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = REPO_ROOT
sys.path.insert(0, os.path.join(REPO_ROOT, 'code', '03_scoring')); import importlib; S = importlib.import_module('0304_score_emi')
OUT = os.path.join(REPO_ROOT, 'results', 'revision', '07_validation_auc')
v = pd.read_csv(os.path.join(OUT, 'validation_instances.csv'), dtype={'id': str})
vr = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'validation', 'unzipped', 'validation_results.csv'), dtype={'id': str})[['id', 'original_text', 'text']]
v = v.merge(vr, on='id', how='left').dropna(subset=['text'])
v['grp'] = np.where(v.party.isin(['DIE LINKE', 'PDS']), 'Linke/PDS', np.where(v.year >= 1949, 'other parties', 'pre-1949'))

kv = S.load_model(os.path.join(REPO_ROOT, 'data', 'parliament_model')); tok_stop = S.nltk_german_stopwords()
d = pd.read_csv(S.DICT); EV = d['evidence'].dropna().tolist(); IU = d['intuition'].dropna().tolist()[:38]
GROUPS = {
    'deception': ['lügen', 'Propaganda', 'Fake News', 'Täuschung', 'täuschen', 'irreführen', 'Unehrlichkeit', 'unehrlich'],
    'honesty': ['Ehrlichkeit', 'ehrlich', 'aufrichtig', 'beteuern', 'vertrauen', 'vertrauenswürdig'],
    'scepticism': ['Zweifel', 'Verdacht', 'Misstrauen'],
    'claims': ['Behauptung', 'behaupten'],
}
CORE = ['Bauchgefühl', 'Gefühl', 'Instinkt', 'Intuition', 'glauben', 'Überzeugung', 'Meinung', 'Ansicht', 'Sichtweise', 'Standpunkt', 'vermuten', 'gesunder Menschenverstand']
VARIANTS = {'full': IU,
            'minus deception': [w for w in IU if w not in GROUPS['deception']],
            'minus deception+honesty': [w for w in IU if w not in GROUPS['deception'] + GROUPS['honesty']],
            'minus deception+honesty+scepticism+claims': [w for w in IU if w not in sum(GROUPS.values(), [])],
            'intuition core only': [w for w in IU if w in CORE]}

V, _ = S.doc_vectors(v.text.tolist(), kv, tok_stop)
E, ne = S.doc_vectors(EV, kv, tok_stop); Evec = E[ne > 0].mean(0)
ev = S.cosine(V, Evec)
# numeral density on the ORIGINAL text (numbers are stripped by the preprocessing)
num = v.original_text.fillna('').apply(lambda t: len(re.findall(r'\d+(?:[.,]\d+)?\s*(?:%|Prozent|Prozentpunkte|Milliarden|Millionen|Euro|Mark|DM)?', t)))
v['numerals_per_100tok'] = 100 * num / v.original_text.fillna('').str.split().str.len().clip(lower=1)
z = lambda x: (x - np.nanmean(x)) / np.nanstd(x)

rows = []
def auc(y, s):
    m = ~np.isnan(s); return roc_auc_score(y[m], s[m]) if y[m].min() != y[m].max() else np.nan
for name, words in VARIANTS.items():
    I, ni = S.doc_vectors(words, kv, tok_stop); iu = S.cosine(V, I[ni > 0].mean(0))
    emi = z(ev) - z(iu)
    for g in ['all', 'Linke/PDS', 'other parties']:
        m = np.ones(len(v), bool) if g == 'all' else (v.grp == g).values
        rows.append(dict(variant=name, n_terms=len(words), group=g, n=int(m.sum()),
                         auc_evidence=auc(v.ev_label.values[m], ev[m]), auc_intuition=auc(v.iu_label.values[m], iu[m]), auc_emi=auc(v.emi_label.values[m], emi[m])))
# numeral cue: evidence score plus standardised numeral density, and EMI built from it
iu_full = S.cosine(V, S.doc_vectors(IU, kv, tok_stop)[0][S.doc_vectors(IU, kv, tok_stop)[1] > 0].mean(0))
for w in [0.5, 1.0]:
    ev2 = z(ev) + w * z(v.numerals_per_100tok.values); emi2 = ev2 - z(iu_full)
    for g in ['all', 'Linke/PDS', 'other parties']:
        m = np.ones(len(v), bool) if g == 'all' else (v.grp == g).values
        rows.append(dict(variant=f'full + numeral density (w={w})', n_terms=len(IU), group=g, n=int(m.sum()),
                         auc_evidence=auc(v.ev_label.values[m], ev2[m]), auc_intuition=auc(v.iu_label.values[m], iu_full[m]), auc_emi=auc(v.emi_label.values[m], emi2[m])))
for g in ['all', 'Linke/PDS', 'other parties']:
    m = np.ones(len(v), bool) if g == 'all' else (v.grp == g).values
    rows.append(dict(variant='numeral density alone', n_terms=0, group=g, n=int(m.sum()), auc_evidence=auc(v.ev_label.values[m], v.numerals_per_100tok.values[m]), auc_intuition=np.nan, auc_emi=np.nan))
res = pd.DataFrame(rows); res.to_csv(os.path.join(OUT, 'linke_diagnostics.csv'), index=False)
pd.set_option('display.width', 200)
print(res.pivot(index='variant', columns='group', values='auc_emi').round(2).rename(columns=lambda c: f'EMI AUC {c}').to_string()); print()
print(res.pivot(index='variant', columns='group', values='auc_evidence').round(2).rename(columns=lambda c: f'evidence AUC {c}').to_string()); print()
print(res.pivot(index='variant', columns='group', values='auc_intuition').round(2).rename(columns=lambda c: f'intuition AUC {c}').to_string())
print('\nnumerals per 100 tokens, Linke chunks rated evidence-based vs not:', v[(v.grp == 'Linke/PDS') & (v.ev_label == 1)].numerals_per_100tok.mean().round(2), v[(v.grp == 'Linke/PDS') & (v.ev_label == 0)].numerals_per_100tok.mean().round(2))
