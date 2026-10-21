"""EMI scoring for the pre-2015 extension, replicating the paper's DDR pipeline without the SBERT wrapper.

Pipeline (final_repo/German-EMI/code/01_data/.../0113_preprocess_speeches.ipynb and 03_scoring/0301_sbert_emi_speeches.ipynb):
  1. text cleaning: collapse '..', '--', '__', '**', whitespace; drop characters outside [a-zA-Z0-9äöüÄÖÜß\\s];
     drop NLTK German stop words + custom set, tokens <= 2 chars, mostly-numeric tokens; keep texts > 10 tokens
  2. document vector = mean of word2vec vectors of the remaining in-vocabulary tokens (SBERT WordEmbeddings +
     WhitespaceTokenizer(stop_words=NLTK German) + mean Pooling); dictionary vector = mean over keyword vectors,
     a multi-word keyword being the mean of its components
  3. evidence_score / intuition_score = cosine(document, dictionary)
  4. length-bin adjustment (bins of 10 tokens on cleaned length): score minus bin mean
  5. z-standardise the adjusted scores over the WHOLE scored set; emi = evidence_z - intuition_z

Usage
  verify : .venv/Scripts/python.exe code/03_scoring/0304_score_emi.py --model <w2v> --verify
           scores data/parliament_nhb.csv text_clean and correlates with the stored evidence/intuition scores
  score  : .venv/Scripts/python.exe code/03_scoring/0304_score_emi.py --model <w2v> --input data/revision/bundestag_speeches_2000_2025_raw.csv
           --text Text --id ID --output data/revision/bundestag_2000_2025_emi.csv
<w2v> is the paper's parliament model: gensim .model, KeyedVectors, or the word2vec_new.txt text dump.
"""
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))   # German-EMI repository root: data in data/, outputs in results/
import argparse, os, re, sys, json
import numpy as np, pandas as pd
from scipy.stats import zscore

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = REPO_ROOT
DICT = os.path.join(REPO_ROOT, 'data', 'validation', 'keyword', 'PRODEMINFO_German_keywords.csv')
CUSTOM_STOP = {"ja", "nur", "uns", "ist", "tun", "sei", "wollen", "aber", "machen", "könnte", "dm", "dr", "dem", "vor", "aus"}


class SbertWrapper:
    """The paper's model as saved by 0204_w2v_to_sbert: WordEmbeddings (whitespace tokenizer with its own
    stop-word list and vocabulary) + mean Pooling. Exposes the same interface the gensim path uses."""
    def __init__(self, path):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(path, device='cpu')
        emb = self.model[0]
        self.vector_size = emb.get_word_embedding_dimension()
        self.tokenizer = emb.tokenizer
        self.vocab = len(self.tokenizer.get_vocab()) if hasattr(self.tokenizer, 'get_vocab') else len(emb.tokenizer.vocab)
        self.stop_words = set(getattr(self.tokenizer, 'stop_words', []))

    def __len__(self):
        return self.vocab

    def encode(self, texts, batch_size=1024):
        return self.model.encode(list(texts), batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)


def load_model(path):
    if os.path.isdir(path):
        if os.path.exists(os.path.join(path, 'modules.json')):
            return SbertWrapper(path)
        if os.path.exists(os.path.join(path, 'model', 'modules.json')):
            return SbertWrapper(os.path.join(path, 'model'))
    from gensim.models import Word2Vec, KeyedVectors
    if path.endswith('.txt') or path.endswith('.vec'):
        return KeyedVectors.load_word2vec_format(path, binary=False)
    if path.endswith('.bin'):
        return KeyedVectors.load_word2vec_format(path, binary=True)
    try:
        return Word2Vec.load(path).wv
    except Exception:
        return KeyedVectors.load(path)


def nltk_german_stopwords():
    """NLTK German stopword list (232 words); the list is stored in code/03_scoring/german_stopwords.txt."""
    import nltk
    try:
        from nltk.corpus import stopwords; return set(stopwords.words('german'))
    except LookupError:
        nltk.download('stopwords', quiet=True); from nltk.corpus import stopwords; return set(stopwords.words('german'))


def preprocess(texts, stop_set, min_len=10):
    """0113 preprocessing. Returns cleaned strings and token counts."""
    numeric = re.compile(r"^\d*[a-zA-Z]?\d*$")
    special = re.compile(r'[^a-zA-Z0-9äöüÄÖÜß\s]')
    out, lens = [], []
    for t in texts:
        t = str(t)
        t = re.sub(r"\.\.+", " ", t); t = re.sub(r"\-\-+", " ", t); t = re.sub(r"__+", " ", t); t = re.sub(r"\*\*+", " ", t)
        t = re.sub(r"\s+", " ", t)
        t = special.sub('', t)
        toks = [w for w in t.split() if w.lower() not in stop_set and len(w) > 2 and not numeric.match(w)]
        out.append(' '.join(toks)); lens.append(len(toks))
    return out, np.array(lens)


def doc_vectors(texts, kv, tok_stop):
    """Mean word vector over whitespace tokens, dropping SBERT-tokenizer stop words and OOV tokens."""
    if isinstance(kv, SbertWrapper):                 # the original wrapper does tokenisation + pooling itself
        texts = list(texts); V = np.zeros((len(texts), kv.vector_size), dtype=np.float32)
        step = 20000
        for i in range(0, len(texts), step):
            V[i:i + step] = kv.encode(texts[i:i + step])
            if len(texts) > step: print(f'  encoded {min(i + step, len(texts)):,}/{len(texts):,}', flush=True)
        try:                                           # in-vocabulary token counts for the OOV diagnostic
            vocab = kv.tokenizer.vocab if isinstance(kv.tokenizer.vocab, dict) else set(kv.tokenizer.vocab)
            n_tok = np.array([sum(1 for w in t.split() if w in vocab and w not in kv.stop_words) for t in texts], dtype=np.int32)
        except Exception:
            n_tok = (np.abs(V).sum(axis=1) > 0).astype(np.int32)
        return V, n_tok
    dim = kv.vector_size
    V = np.zeros((len(texts), dim), dtype=np.float32); n_tok = np.zeros(len(texts), dtype=np.int32)
    key = kv.key_to_index
    for i, t in enumerate(texts):
        idx = [key[w] for w in t.split() if w not in tok_stop and w in key]
        if idx:
            V[i] = kv.vectors[idx].mean(axis=0); n_tok[i] = len(idx)
    return V, n_tok


def dictionary_vectors(kv, tok_stop):
    d = pd.read_csv(DICT)
    ev = d['evidence'].dropna().tolist(); iu = d['intuition'].dropna().tolist()[:38]
    E, ne = doc_vectors(ev, kv, tok_stop); I, ni = doc_vectors(iu, kv, tok_stop)
    miss = [w for w, k in zip(ev, ne) if k == 0] + [w for w, k in zip(iu, ni) if k == 0]
    if miss:
        print('keywords without any in-vocabulary token (ignored in the mean):', miss)
    return E[ne > 0].mean(axis=0), I[ni > 0].mean(axis=0), len(ev), len(iu)


def cosine(V, u):
    nv = np.linalg.norm(V, axis=1); nu = np.linalg.norm(u)
    with np.errstate(divide='ignore', invalid='ignore'):
        s = (V @ u) / (nv * nu)
    s[nv == 0] = np.nan
    return s


def length_adjust_and_z(df, length_col='length'):
    bins = range(0, int(df[length_col].max()) + 10, 10)
    df['length_bin'] = pd.cut(df[length_col], bins=bins)
    for c in ['evidence', 'intuition']:
        df[f'{c}_mean'] = df.groupby('length_bin', observed=True)[f'{c}_score'].transform('mean')
        df[f'{c}_adj'] = df[f'{c}_score'] - df[f'{c}_mean']
        df[f'{c}_z'] = zscore(df[f'{c}_adj'], nan_policy='omit')
    df['emi'] = df['evidence_z'] - df['intuition_z']
    return df


def score_frame(df, text_col, kv, tok_stop, already_clean=False):
    if already_clean:
        clean = df[text_col].fillna('').astype(str).tolist(); lens = np.array([len(t.split()) for t in clean])
    else:
        clean, lens = preprocess(df[text_col].tolist(), nltk_german_stopwords() | CUSTOM_STOP)
    df = df.copy(); df['text_clean'] = clean; df['length'] = lens
    df = df[df['length'] > 10].reset_index(drop=True)
    E, I, ne, ni = dictionary_vectors(kv, tok_stop)
    V, ntok = doc_vectors(df['text_clean'].tolist(), kv, tok_stop)
    df['n_invocab'] = ntok; df['oov_rate'] = 1 - ntok / df['length'].clip(lower=1)
    df['evidence_score'] = cosine(V, E); df['intuition_score'] = cosine(V, I)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--input'); ap.add_argument('--text', default='Text'); ap.add_argument('--id', default='ID')
    ap.add_argument('--output'); ap.add_argument('--verify', action='store_true')
    ap.add_argument('--nrows', type=int)
    ap.add_argument('--already-clean', action='store_true', help='text column is already preprocessed (text_clean)')
    a = ap.parse_args()
    kv = load_model(a.model); print('model:', a.model, 'vocab', len(kv), 'dim', kv.vector_size)
    tok_stop = nltk_german_stopwords()          # the SBERT WhitespaceTokenizer stop-word list

    if a.verify:
        ref = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'parliament_nhb.csv'),
                          usecols=['id', 'date', 'text_clean', 'length', 'evidence_score_w2vparliament',
                                   'intuition_score_w2vparliament', 'emi_w2vparliament'], nrows=a.nrows)
        s = score_frame(ref, 'text_clean', kv, tok_stop, already_clean=True)
        s = length_adjust_and_z(s)
        for c, r in [('evidence_score', 'evidence_score_w2vparliament'), ('intuition_score', 'intuition_score_w2vparliament'),
                     ('emi', 'emi_w2vparliament')]:
            ok = s[[c, r]].dropna()
            print(f'{c:16s} r = {ok[c].corr(ok[r]):.4f}   mean abs diff = {(ok[c]-ok[r]).abs().mean():.4f}   n = {len(ok)}')
        print('mean OOV rate', round(float(s.oov_rate.mean()), 4))
        return

    df = pd.read_csv(a.input, nrows=a.nrows, dtype={a.id: str})
    print('input rows', len(df))
    s = score_frame(df, a.text, kv, tok_stop, already_clean=a.already_clean)
    print('rows after length > 10 filter', len(s))
    s = length_adjust_and_z(s)                   # z over the whole 2000-2025 set
    s.drop(columns=['length_bin']).to_csv(a.output, index=False)
    print('written', a.output); print(s[['evidence_score', 'intuition_score', 'emi']].describe().round(4))
    datecol = 'Date' if 'Date' in s.columns else 'date'
    s.groupby(pd.to_datetime(s[datecol], utc=True).dt.year)[['evidence_score', 'intuition_score', 'emi', 'oov_rate']].mean().round(4).to_csv(a.output.replace('.csv', '_by_year.csv'))


if __name__ == '__main__':
    main()
