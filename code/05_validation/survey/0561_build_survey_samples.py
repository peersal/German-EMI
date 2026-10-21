"""Build the two 300-instance survey samples (Bundestag chunks and tweets, 2015-2025), stratified by party x EMI quintile.

  python docs/survey/revision/build_survey_samples.py

Parliament: a random pool of speeches 2015-2025 is cut into 45-90-word chunks at sentence boundaries (speaker header
stripped, same logic as code/05_validation/llm/0551_build_inputs.py); chunks that look procedural, truncated or contain stage
directions are dropped; the remaining chunks are scored with the parliament embedding model (code/03_scoring/0304_score_emi.py) and
chunks are drawn per party with quotas that oversample the parties the old survey under-covered (AfD 80, Left 75,
Greens 55, FDP 50, CDU/CSU 20, SPD 20), spread evenly over the global quintiles of chunk EMI.
Twitter: original (non-reply, non-retweet) tweets 2015-2025 with >= 12 words after removing links and @mentions, at
most 4 hashtags, no @mentions, German function words present, sentence-final punctuation, no event reports, thanks or
announcements, no duplicates; 50 per party, 10 per quintile of tweet EMI.

Outputs per channel (docs/survey/revision/<parliament|twitter>/data_files/):
  survey_sample.csv       blinded: id, text                (what potato shows)
  survey_sample_key.csv   id, source id, party, leaning, date, EMI, quintile, ... (never shown to annotators)
and docs/survey/revision/sample_summary.md with the counts per party and the filter statistics.
"""
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))   # German-EMI repository root: data in data/, outputs in results/
import os, re, sys, html, numpy as np, pandas as pd
HERE = os.path.join(REPO_ROOT, 'docs', 'survey', 'revision'); ROOT = REPO_ROOT
sys.path.insert(0, os.path.join(REPO_ROOT, 'code', '03_scoring')); import importlib; se = importlib.import_module('0304_score_emi')
rng = np.random.default_rng(20261019)
PARTIES = ['CDU/CSU', 'SPD', 'Alliance 90/The Greens', 'FDP', 'The Left Party', 'AFD']
PER_PARTY, N_Q = 50, 5
# Bundestag quotas oversample the parties the original survey under-covered (old ground truth, all eras:
# SPD 190, CDU/CSU 159, FDP 61, Greens 52, Left 23 + PDS 11, AfD 20). Combined old + new counts then come to
# roughly AfD 100, Left 109, Greens 107, FDP 111, CDU/CSU 179, SPD 210.
QUOTA_PARLIAMENT = {'AFD': 80, 'The Left Party': 75, 'Alliance 90/The Greens': 55, 'FDP': 50, 'CDU/CSU': 20, 'SPD': 20}
QUOTA_TWITTER = {p: PER_PARTY for p in PARTIES}
MIN_WORDS, MAX_WORDS, TARGET = 45, 90, 60
START, END = '2015-01-01', '2025-12-31'
LOG = []


def log(s): print(s); LOG.append(s)


# ---------------------------------------------------------------- parliament ---------------------------------
SENT = re.compile(r'(?<=[.!?])\s+(?=[A-ZÄÖÜ„"(])')
BAD_ANY = re.compile(r'\(Beifall|\(Zuruf|\(Lachen|\(Heiterkeit|\(Widerspruch|\(Unruhe|Zwischenruf|Drucksache|Tagesordnungspunkt|'
                     r'Geschäftsordnung|Redezeit|Ich frage die Bundesregierung|Fragestunde|Zusatzfrage|Nachfrage|'
                     r'das Wort hat|erteile (ich )?(nun |jetzt )?das Wort|hat (nun |jetzt )?das Wort|zur Abstimmung|Beschlussempfehlung|Überweisung|Vielen Dank für (die|Ihre) Aufmerksamkeit|Danke schön|Ich danke Ihnen|Herzlichen Dank\.|\[|\]|\.\.\.$', re.I)
BAD_START = re.compile(r'^(Vielen Dank|Herzlichen Dank|Danke|Herr Präsident|Frau Präsidentin|Sehr geehrte|Liebe Kolleg|Meine (sehr )?(verehrten |geehrten )?Damen|'
                       r'Meine Damen|Werte Kolleg|Verehrte|Guten (Morgen|Tag|Abend)|Hallo|Ja,|Nein,|Also,|Nun,)', re.I)
GREET = re.compile(r'(Herr Präsident|Frau Präsidentin|Damen und Herren|Kolleginnen und Kollegen|Kollegen und Kolleginnen|Liebe Kolleg)', re.I)


def strip_header(text, last):
    m = re.match(r'^([^:]{0,220}?):\s', text)
    if m and isinstance(last, str) and last.split()[-1] in m.group(1): return text[m.end():]
    return text


ABBR_END = re.compile(r'((Dr|Nr|Abs|bzw|ca|usw|etc|vgl|Prof|Art|Mio|Mrd|St|Hr|Fr|Jh|Str|Tsd|sog|evtl|ggf|inkl|zz|Ziff|Bd|Dipl|Ing|Kap|Pkt)\.|z\. ?B\.|u\. ?a\.|d\. ?h\.|i\. ?d\. ?R\.|\d\.)$')
MONTH_START = re.compile(r'^(Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember|Uhr|Prozent|Euro|Milliarden|Millionen)')


def sentences(text):
    raw = SENT.split(text.strip()); out = []
    for s in raw:
        if out and ABBR_END.search(out[-1]): out[-1] = out[-1] + ' ' + s
        else: out.append(s)
    return out


def chunks_of(text):
    sents = sentences(text); out, cur, n = [], [], 0
    for s in sents:
        w = len(s.split())
        if n and n + w > MAX_WORDS: out.append(' '.join(cur)); cur, n = [], 0
        cur.append(s); n += w
        if n >= TARGET: out.append(' '.join(cur)); cur, n = [], 0
    if cur: out.append(' '.join(cur))
    return out


def chunk_ok(c):
    w = c.split()
    if not (MIN_WORDS <= len(w) <= MAX_WORDS): return 'length'
    if not re.match(r'^[A-ZÄÖÜ„"(]', c): return 'start'
    if not re.search(r'[.!?]["“”)]?$', c): return 'end'
    if BAD_ANY.search(c): return 'procedural'
    if BAD_START.match(c): return 'greeting_start'
    if MONTH_START.match(c) or re.match(r'^[A-ZÄÖÜ][a-zäöüß]*\s*[–-]\s', c): return 'fragment_start'
    if len(GREET.findall(c)) > 0: return 'greeting'
    if sum(ch.isupper() for ch in c) / max(1, sum(ch.isalpha() for ch in c)) > 0.12: return 'caps'
    if re.search(r'\d{2}/\d{3,}', c) or c.count('–') > 3 or c.count(' - ') > 3: return 'format'
    if max(len(t) for t in w) > 35: return 'longtoken'
    return 'ok'


def stratified(df, emi_col, label, quota):
    df = df.copy(); df['q'] = pd.qcut(df[emi_col], N_Q, labels=False) + 1
    take = []
    for p in PARTIES:
        g = df[df.party == p]; n_p = quota[p]
        per_q = int(np.ceil(n_p / N_Q)); got = []
        for q in range(1, N_Q + 1):
            gq = g[(g.q == q) & ~g.index.isin(got)]
            k = min(per_q, len(gq), n_p - len(got)); got += list(gq.sample(k, random_state=int(rng.integers(1e9))).index)
        if len(got) < n_p:   # fill from the quintiles with spare instances
            rest = g[~g.index.isin(got)]; got += list(rest.sample(min(n_p - len(got), len(rest)), random_state=int(rng.integers(1e9))).index)
        take += got
    out = df.loc[take]
    log(f'{label}: {len(out)} instances; per party x quintile:\n' + out.groupby(['party', 'q']).size().unstack(fill_value=0).to_string())
    return out


def build_parliament():
    cols = ['id', 'date', 'electoralTerm', 'politicianId', 'lastName', 'party', 'leaning', 'text', 'length', 'emi_w2vparliament']
    d = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'parliament_nhb.csv'), usecols=lambda c: c in cols, dtype={'id': str})
    d = d.dropna(subset=['text', 'party', 'emi_w2vparliament']); d['date'] = pd.to_datetime(d.date, utc=True).dt.tz_localize(None)
    d = d[(d.date >= START) & (d.date <= END) & d.party.isin(PARTIES)]
    log(f'parliament: {len(d):,} speeches 2015-2025 in the six parties')
    pool = d[d.text.str.split().str.len() >= MIN_WORDS].sample(min(24000, len(d)), random_state=20261019)
    rows, reasons = [], {}
    for r in pool.itertuples():
        body = re.sub(r'\s+', ' ', strip_header(r.text, r.lastName))
        for k, c in enumerate(chunks_of(body)):
            why = chunk_ok(c); reasons[why] = reasons.get(why, 0) + 1
            if why == 'ok': rows.append(dict(speech_id=r.id, chunk_idx=k, text=c, party=r.party, leaning=r.leaning, date=r.date, politicianId=r.politicianId, electoralTerm=r.electoralTerm, speech_emi=r.emi_w2vparliament, speech_words=len(r.text.split())))
    C = pd.DataFrame(rows); log(f'parliament chunks from {len(pool):,} speeches: {sum(reasons.values()):,} chunks, kept {len(C):,}; dropped by reason {dict(sorted(reasons.items(), key=lambda x: -x[1]))}')
    C = C.groupby('speech_id').sample(1, random_state=20261019)          # one chunk per speech
    C = C.drop_duplicates('text'); C['n_words'] = C.text.str.split().str.len()
    kv = se.load_model(os.path.join(REPO_ROOT, 'data', 'parliament_model')); tok_stop = se.nltk_german_stopwords()
    sc = se.score_frame(C.rename(columns={'text': 'Text'}), 'Text', kv, tok_stop); sc = se.length_adjust_and_z(sc)
    C['chunk_emi'] = sc['emi'].values
    C = C.dropna(subset=['chunk_emi']); log(f'parliament: {len(C):,} candidate chunks (one per speech) scored; chunk EMI vs speech EMI rho {C[["chunk_emi", "speech_emi"]].corr("spearman").iloc[0, 1]:.2f}')
    S = stratified(C, 'chunk_emi', 'parliament', QUOTA_PARLIAMENT)
    return S.rename(columns={'chunk_emi': 'emi'})


# ---------------------------------------------------------------- twitter ------------------------------------
URL = re.compile(r'https?://\S+|www\.\S+'); MENTION = re.compile(r'@\w+'); HASH = re.compile(r'#\w+')
DE = re.compile(r'\b(und|der|die|das|nicht|ist|wir|für|mit|auf|den|dem|ein|eine|sich|auch|von|zu|im|es|dass|sind|wird|haben|bei|nach|aus|noch|uns|mehr|muss|werden)\b', re.I)
EMOJI = re.compile('[🌀-🫿☀-➿🇦-🇿]')
EN = re.compile(r'\b(the|and|with|for|that|this|have|from|will|are|you|our|not)\b')


def tweet_clean(t):
    t = html.unescape(t); t = URL.sub('', t); t = MENTION.sub('', t); t = re.sub(r'\s+', ' ', t).strip(); t = re.sub(r'^[\s:,.\-–]+', '', t)
    return t


def tweet_ok(raw):
    if raw.startswith('RT ') or raw.startswith('@'): return 'retweet_or_reply'
    if len(HASH.findall(raw)) > 4: return 'hashtags'
    if len(MENTION.findall(raw)) > 0: return 'mentions'
    c = tweet_clean(raw); w = c.split()
    if len(w) < 15: return 'short'
    if len(w) > 70: return 'long'
    if len(DE.findall(c)) < 2 or len(EN.findall(c)) > 3: return 'language'
    if not re.search(r'[.!?]["“”)]?$', c): return 'end'
    if re.search(r'\.\.\.|…|\(\d/\d\)|\d/\d$|via|\s\|\s|^[^:]{1,25}:\s', c): return 'fragment_or_headline'
    if len(EMOJI.findall(c)) > 1: return 'emoji'
    if re.match(r'^(Heute|Gestern|Morgen|Jetzt|Gerade|Gleich|Soeben|Eben|Live|Vorhin)', c): return 'event_report'
    if re.search(r'Herzlichen Dank|Danke|dankbar|lieber !|liebe !|Glück|wünsche|Glückwunsch|Neujahr|neue Jahr|Weihnacht|Geburtstag|Danke an|Vielen Dank|Dankeschön|willkommen|Mutter|Vater|Urlaub|Wochenende|Appetit|Ostern|Advent|Jubiläum|Geschenk|Spaß|Ticket|Anmeldung|anmelden|Einladung|lade .* ein|Einschalten|einschalten|heute Abend|heute um|morgen um|Uhr|Livestream|Podcast|Interview|Sendung|Talk|Zu Gast|zu Gast|Rede im|meine Rede|Pressekonferenz|Pressemitteilung|Lesetipp|Artikel|Bericht in|Beitrag in|Kolumne', c, re.I): return 'announcement'
    if sum(ch.isupper() for ch in c) / max(1, sum(ch.isalpha() for ch in c)) > 0.2: return 'caps'
    if re.search(r'\d{1,2}[:.]\d{2}\s*(Uhr|h)\b|Livestream|live dabei|jetzt live|Link in|Umfrage:|Gewinnspiel|Danke für|Herzlichen Glückwunsch|Alles Gute|Guten Morgen|Gute Nacht|Frohe|Schönes Wochenende', c, re.I): return 'announcement'
    if '\n' in raw and raw.count('\n') > 6: return 'format'
    return 'ok'


def build_twitter():
    cols = ['id', 'author_id', 'created_at', 'party', 'leaning', 'reply', 'retweeted', 'text', 'emi_w2vparliament']
    d = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'twitter_nhb.csv'), usecols=lambda c: c in cols, dtype={'id': str, 'author_id': str}, low_memory=False)
    d = d.dropna(subset=['text', 'party', 'emi_w2vparliament']); d['created_at'] = pd.to_datetime(d.created_at, utc=True, errors='coerce').dt.tz_localize(None)
    d = d[(d.created_at >= START) & (d.created_at <= END) & d.party.isin(PARTIES) & d.reply.astype(str).str.lower().isin(['false', '0', 'nan']) & d.retweeted.astype(str).str.lower().isin(['false', '0', 'nan'])]
    log(f'twitter: {len(d):,} original tweets 2015-2025 in the six parties (no replies, no retweets)')
    pool = d.groupby('party', group_keys=False).apply(lambda g: g.sample(min(6000, len(g)), random_state=20261019))
    pool['why'] = pool.text.map(tweet_ok); reasons = pool.why.value_counts().to_dict()
    log(f'twitter pool {len(pool):,} tweets: kept {reasons.get("ok", 0):,}; dropped by reason {reasons}')
    T = pool[pool.why == 'ok'].copy(); T['text_shown'] = T.text.map(tweet_clean); T = T.drop_duplicates('text_shown')
    T = T.rename(columns={'id': 'tweet_id', 'created_at': 'date', 'emi_w2vparliament': 'emi'})
    T = T.groupby('author_id', group_keys=False).apply(lambda g: g.sample(min(8, len(g)), random_state=20261019))   # cap per account
    S = stratified(T, 'emi', 'twitter', QUOTA_TWITTER); S['n_words'] = S.text_shown.str.split().str.len()
    return S


def write(S, folder, prefix, text_col):
    S = S.sample(frac=1, random_state=20261019).reset_index(drop=True); S.insert(0, 'id', [f'{prefix}{i + 1:03d}' for i in range(len(S))])
    out = os.path.join(HERE, folder, 'data_files'); os.makedirs(out, exist_ok=True)
    S[['id', text_col]].rename(columns={text_col: 'text'}).to_csv(os.path.join(out, 'survey_sample.csv'), index=False)
    S.drop(columns=[c for c in ['why'] if c in S]).to_csv(os.path.join(out, 'survey_sample_key.csv'), index=False)
    log(f'{folder}: wrote {len(S)} instances to {out} (words median {S.n_words.median():.0f}, min {S.n_words.min()}, max {S.n_words.max()})')
    return S


if __name__ == '__main__':
    P = write(build_parliament(), 'parliament', 'p', 'text')
    T = write(build_twitter(), 'twitter', 't', 'text_shown')
    with open(os.path.join(HERE, 'sample_summary.md'), 'w', encoding='utf-8', newline='\n') as f:
        f.write('# Survey samples (built %s)\n\n' % pd.Timestamp.today().date())
        for name, S in [('Bundestag chunks', P), ('Tweets', T)]:
            f.write(f'## {name}\n\n| Party | n | EMI mean | EMI min | EMI max | words median |\n| --- | --- | --- | --- | --- | --- |\n')
            for p, g in S.groupby('party'): f.write(f'| {p} | {len(g)} | {g.emi.mean():+.2f} | {g.emi.min():+.2f} | {g.emi.max():+.2f} | {g.n_words.median():.0f} |\n')
            f.write(f'| **All** | {len(S)} | {S.emi.mean():+.2f} | {S.emi.min():+.2f} | {S.emi.max():+.2f} | {S.n_words.median():.0f} |\n\n')
        f.write('## Build log\n\n```\n' + '\n'.join(LOG) + '\n```\n')
    print('summary written to', os.path.join(HERE, 'sample_summary.md'))
