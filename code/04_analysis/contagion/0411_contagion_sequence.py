"""Within-sitting contagion analysis (cell 24 of 0403_analysis.ipynb): speech sequences per sitting, contagion effect (next minus previous speech by another party), k-windows, permutation test, within-speaker models, dose / year / naming splits, party reply network, MdB mention network, example sitting. Run with `python code/04_analysis/0406_run_figure_cells.py contagion` from the repository root.
"""
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))   # German-EMI repository root: data in data/, outputs in results/
# ======================================================================
# Within-sitting reaction to AfD speeches ("contagion" analysis, revision).
# Unit: the ordered sequence of plenary speeches within a sitting. For a
# speech by party X, the contagion effect is the EMI of the next speech by
# another party minus the EMI of the previous speech by another party.
#   A  reply network since 24 Oct 2017: arrow = speaker -> next speaker,
#      width = number of transitions, colour = mean contagion effect
#   B  MdB-level mention network, same period: MdBs clustered by party, an
#      edge when a speech names another MdB (title + surname) or a party,
#      colour = mean EMI of those speeches, width = number of speeches
#   C  one sitting: every speech's EMI in speech order, AfD speeches
#      shaded, arrows = reply around each AfD speech
#      (example_sitting, chosen as a long sitting with many AfD speeches,
#      consistent drops after them and a flat placebo for other parties)
# The dose-response (by EMI quartile), the yearly effect and the split by
# whether the reply names the AfD are computed and saved as tables only.
# The distance profile (all parties) is saved as contagion_sequence_profile.csv.
# Inputs: data/revision/bundestag_speeches_2000_2025_raw.csv (sitting order),
#         data/parliament_nhb.csv (EMI, text). Outputs: results/
#         contagion_sequence.{pdf,png}, results/parliament/tables/
#         contagion_sequence_*.csv
# ======================================================================
import re
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib import colors as mcolors

_SEQ_PARTIES = ['AFD', 'CDU/CSU', 'FDP', 'SPD', 'Alliance 90/The Greens', 'The Left Party']
_SEQ_SHORT = {'AFD': 'AfD', 'CDU/CSU': 'CDU/CSU', 'FDP': 'FDP', 'SPD': 'SPD',
              'Alliance 90/The Greens': 'Greens', 'The Left Party': 'Left'}
_SEQ_RAW_TO_NHB = {'AfD': 'AFD', 'CDU/CSU': 'CDU/CSU', 'FDP': 'FDP', 'SPD': 'SPD',
                   'BÜNDNIS 90/DIE GRÜNEN': 'Alliance 90/The Greens', 'DIE LINKE': 'The Left Party'}
_SEQ_AFD_ENTRY = pd.Timestamp('2017-10-24')
_SEQ_EMI_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'reply_effect', ['#D62728', '#D9D9D9', '#2196F3'])   # more intuitive <- 0 -> more evidence-based


def load_speech_sequences(emi_col='emi_w2vparliament', start='2015-01-01'):
    """Ordered regular speeches per sitting (raw Bundestag file) joined with the
    paper's EMI scores and an AfD-mention flag from the speech text."""
    raw = pd.read_csv('data/revision/bundestag_speeches_2000_2025_raw.csv',
                      usecols=['Source_ID', 'Date', 'Sitting', 'Party', 'Role'], dtype=str)
    raw = raw[(raw.Role == 'Regular') & (raw.Date >= start)]
    sc = pd.read_csv('data/parliament_nhb.csv', usecols=['id', emi_col, 'text', 'politicianId'], dtype={'id': str})
    sc['mentions_afd'] = sc.text.fillna('').str.lower().str.contains(r'\bafd\b|alternative für deutschland', regex=True).astype(int)
    d = raw.merge(sc[['id', emi_col, 'mentions_afd', 'politicianId']], left_on='Source_ID', right_on='id').dropna(subset=[emi_col])
    d['party'] = d.Party.map(_SEQ_RAW_TO_NHB)
    d['order'] = pd.to_numeric(d.Source_ID.str.extract(r'ID(\d+)')[0], errors='coerce')
    d['date'] = pd.to_datetime(d.Date)
    return d.sort_values(['Sitting', 'order']).reset_index(drop=True)


def reply_rows(d, emi_col='emi_w2vparliament'):
    """One row per speech: party, EMI, previous/next speech by another party
    (same sitting), contagion effect = next minus previous, reply names the AfD."""
    rows = []
    for sit, g in d.groupby('Sitting', sort=False):
        party, emi, men, dates = g.party.values, g[emi_col].values, g.mentions_afd.values, g.date.values
        n = len(g)
        for i in range(n):
            X = party[i]
            if X not in _SEQ_PARTIES:
                continue
            b = next((j for j in range(i - 1, -1, -1) if party[j] != X), None)
            a = next((j for j in range(i + 1, n) if party[j] != X), None)
            if b is None or a is None or party[a] not in _SEQ_PARTIES:
                continue
            rows.append(dict(sitting=sit, date=dates[i], src=X, dst=party[a], emi=emi[i],
                             diff=emi[a] - emi[b], reply_mentions_afd=men[a]))
    return pd.DataFrame(rows)


def distance_profile(d, emi_col='emi_w2vparliament', offsets=range(-6, 7)):
    """Mean EMI (and s.e.) of other parties' speeches at each offset from a
    speech by party X, within sittings."""
    out = {}
    for sit, g in d.groupby('Sitting', sort=False):
        party, emi = g.party.values, g[emi_col].values; n = len(g)
        for i in range(n):
            X = party[i]
            if X not in _SEQ_PARTIES:
                continue
            for o in offsets:
                j = i + o
                if o == 0 or j < 0 or j >= n or party[j] == X:
                    continue
                out.setdefault((X, o), []).append(emi[j])
    rec = [dict(party=X, offset=o, mean=np.mean(v), se=np.std(v, ddof=1) / np.sqrt(len(v)), n=len(v))
           for (X, o), v in out.items()]
    return pd.DataFrame(rec)


def _cluster_se(x, groups):
    """Standard error of a mean with clustering by sitting."""
    x = np.asarray(x, float); m = x.mean(); g = pd.Series(x - m).groupby(np.asarray(groups)).sum()
    return np.sqrt((g ** 2).sum()) / len(x)


def _draw_reply_network(ax, edges, node_n, title, min_n=60, label_src=None, norm=None, show_labels=False):
    order = _SEQ_PARTIES
    present = [p for p in order if p in node_n.index and node_n[p] > 0]
    ang = {p: np.pi / 2 - 2 * np.pi * i / len(order) for i, p in enumerate(order)}
    pos = {p: (np.cos(a), np.sin(a)) for p, a in ang.items()}
    nmax = node_n.max()
    rad = {p: 0.08 + 0.13 * np.sqrt(node_n[p] / nmax) for p in present}
    eb = edges[edges.src.isin(present) & edges.dst.isin(present) & (edges.n >= min_n)].copy()
    wmax = eb.n.max()
    for r in sorted(eb.itertuples(), key=lambda r: (r.src == label_src, r.n)):
        (x0, y0), (x1, y1) = pos[r.src], pos[r.dst]
        vx, vy = x1 - x0, y1 - y0; L = np.hypot(vx, vy); ux, uy = vx / L, vy / L
        start = (x0 + ux * rad[r.src] * 1.05, y0 + uy * rad[r.src] * 1.05)
        end = (x1 - ux * rad[r.dst] * 1.08, y1 - uy * rad[r.dst] * 1.08)
        lw = 1.0 + 9.0 * np.sqrt(r.n / wmax)
        emph = (label_src is not None and r.src == label_src)
        ax.add_patch(FancyArrowPatch(start, end, connectionstyle='arc3,rad=0.22', arrowstyle='-|>',
                                     mutation_scale=18, lw=lw, color=_SEQ_EMI_CMAP(norm(r.diff)),
                                     alpha=0.95 if emph or label_src is None else 0.45,
                                     zorder=3 if emph else 2, shrinkA=0, shrinkB=0))
    for p in present:
        ax.add_patch(Circle(pos[p], rad[p], facecolor=party_colors.get(p, 'gray'), edgecolor='white', lw=2, zorder=4))
        tx, ty = pos[p][0] * (1 + rad[p] + 0.26), pos[p][1] * (1 + rad[p] + 0.26)
        ax.text(tx, ty, _SEQ_SHORT[p], ha='center', va='center',
                fontsize=PLOT_TICK_SIZE + 2, color='black', zorder=5)
    if label_src is not None and show_labels:
        for r in eb[eb.src == label_src].itertuples():
            (x0, y0), (x1, y1) = pos[label_src], pos[r.dst]
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2; vx, vy = x1 - x0, y1 - y0; L = np.hypot(vx, vy)
            ax.text(mx - vy / L * 0.2, my + vx / L * 0.2, f'{r.diff:+.2f}', fontsize=PLOT_TICK_SIZE - 2,
                    color=_SEQ_EMI_CMAP(norm(r.diff)), ha='center', va='center', fontweight='bold', zorder=6,
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.9))
    ax.set_xlim(-1.7, 1.7); ax.set_ylim(-1.62, 1.62); ax.set_aspect('equal'); ax.axis('off')
    if title:
        ax.set_title(title, fontsize=PLOT_LEGEND_SIZE + 2, loc='left', pad=4)


def permutation_null_afd(d, emi_col='emi_w2vparliament', n_perm=2000, seed=0):
    """Null distribution of the AfD contagion effect: within each sitting the AfD
    label is reassigned at random to the same number of speeches, keeping the
    speech order and all EMI values; the contagion effect (next minus previous
    speech by a non-AfD speaker) is recomputed for the relabelled speeches."""
    rng = np.random.default_rng(seed)
    sittings = []
    for sit, g in d.groupby('Sitting', sort=False):
        party = g.party.values; emi = g[emi_col].values.astype(float); k = int((party == 'AFD').sum())
        if k == 0 or len(g) < 3:
            continue
        sittings.append((emi, k))

    def effect(emi, afd_mask):
        non = np.where(~afd_mask)[0]
        if len(non) < 2:
            return np.array([])
        pos = np.where(afd_mask)[0]
        prev_i = np.searchsorted(non, pos, side='left') - 1; next_i = np.searchsorted(non, pos, side='right')
        ok = (prev_i >= 0) & (next_i < len(non))
        return emi[non[next_i[ok]]] - emi[non[prev_i[ok]]]
    null = np.empty(n_perm)
    for p in range(n_perm):
        acc, cnt = 0.0, 0
        for emi, k in sittings:
            mask = np.zeros(len(emi), bool); mask[rng.choice(len(emi), k, replace=False)] = True
            e = effect(emi, mask); acc += e.sum(); cnt += len(e)
        null[p] = acc / cnt
    return null


def within_speaker_afd_effect(d, emi_col='emi_w2vparliament'):
    """EMI of a non-AfD speech regressed on whether the previous speech by
    another party was an AfD speech, with speaker fixed effects (within-MdB
    demeaning) and, in a second model, sitting fixed effects as well;
    standard errors clustered by sitting."""
    rows = []
    for sit, g in d.groupby('Sitting', sort=False):
        party, emi, pid = g.party.values, g[emi_col].values, g.politicianId.values; n = len(g)
        for j in range(n):
            Y = party[j]
            if Y not in _SEQ_PARTIES or Y == 'AFD':
                continue
            b = next((i for i in range(j - 1, -1, -1) if party[i] != Y), None)
            if b is None or party[b] not in _SEQ_PARTIES:
                continue
            rows.append(dict(sitting=sit, pid=pid[j], y=emi[j], prev_afd=int(party[b] == 'AFD'), prev_emi=emi[b]))
    W = pd.DataFrame(rows).dropna(subset=['pid'])
    W = W[W.groupby('pid').pid.transform('size') >= 5]
    out = []
    for lab, fe in [('speaker FE', ['pid']), ('speaker + sitting FE', ['pid', 'sitting'])]:
        yv = W.y - W.groupby('pid').y.transform('mean'); xv = W.prev_afd - W.groupby('pid').prev_afd.transform('mean')
        if 'sitting' in fe:   # two-way demeaning by alternating projections
            for _ in range(8):
                yv = yv - yv.groupby(W.sitting).transform('mean'); yv = yv - yv.groupby(W.pid).transform('mean')
                xv = xv - xv.groupby(W.sitting).transform('mean'); xv = xv - xv.groupby(W.pid).transform('mean')
        r = sm.OLS(yv, xv).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(W.sitting)[0]})
        out.append(dict(model=lab, coef=r.params.iloc[0], se=r.bse.iloc[0], p=r.pvalues.iloc[0], n_speeches=int(len(W)), n_speakers=int(W.pid.nunique())))
    return pd.DataFrame(out), W


def reply_effect_k(d, emi_col='emi_w2vparliament', ks=(1, 2, 3, 5)):
    """Contagion effect with k speeches on each side: mean EMI of the next k speeches
    by other parties minus the mean of the previous k (same sitting, both sides
    complete). Returns per-k estimates for the AfD, for all other parties pooled
    and the AfD-minus-others contrast, s.e. clustered by sitting."""
    rows = []
    for sit, g in d.groupby('Sitting', sort=False):
        party, emi = g.party.values, g[emi_col].values; n = len(g)
        for i in range(n):
            X = party[i]
            if X not in _SEQ_PARTIES:
                continue
            before = [j for j in range(i - 1, -1, -1) if party[j] != X]
            after = [j for j in range(i + 1, n) if party[j] != X]
            rec = dict(sitting=sit, src=X)
            for k in ks:
                rec[f'd{k}'] = (emi[after[:k]].mean() - emi[before[:k]].mean()) if len(before) >= k and len(after) >= k else np.nan
            rows.append(rec)
    K = pd.DataFrame(rows); out = []
    for k in ks:
        a = K[(K.src == 'AFD')].dropna(subset=[f'd{k}']); o = K[(K.src != 'AFD')].dropna(subset=[f'd{k}'])
        ea, sa = a[f'd{k}'].mean(), _cluster_se(a[f'd{k}'], a.sitting); eo, so = o[f'd{k}'].mean(), _cluster_se(o[f'd{k}'], o.sitting)
        out.append(dict(k=k, afd=ea, afd_se=sa, n_afd=len(a), others=eo, others_se=so, n_others=len(o), contrast=ea - eo, contrast_se=np.sqrt(sa ** 2 + so ** 2)))
    return pd.DataFrame(out)


def mdb_mention_network(emi_col='emi_w2vparliament', start='2017-10-24', min_speeches=30, min_pair=3, min_party=25):
    """MdB-level communication network since `start`: an edge from MdB i to MdB j
    when a speech by i names j (title + surname, unambiguous surnames only), and
    from MdB i to party P when a speech by i names the party. Edge weight = number
    of such speeches, edge value = their mean EMI. Nodes = MdBs with at least
    `min_speeches` speeches; MdB-MdB edges with at least `min_pair` mentions,
    MdB-party edges with at least `min_party` mentions."""
    sc = pd.read_csv('data/parliament_nhb.csv', usecols=['id', 'date', 'politicianId', 'firstName', 'lastName', 'party', 'text', emi_col], dtype={'id': str})
    sc = sc.dropna(subset=['politicianId', emi_col, 'text']); sc['date'] = pd.to_datetime(sc.date, utc=True).dt.tz_localize(None)
    P = sc[sc.date >= start].copy(); P['politicianId'] = P.politicianId.astype(int)
    mdb = P.groupby('politicianId').agg(last=('lastName', 'first'), party=('party', lambda s: s.mode().iloc[0]), n=('id', 'size'), emi=(emi_col, 'mean')).reset_index()
    mdb['last'] = mdb['last'].str.strip(); cnt = mdb['last'].value_counts(); uniq = mdb[~mdb['last'].isin(cnt[cnt > 1].index)]
    name2id = dict(zip(uniq['last'], uniq.politicianId))
    pat = re.compile(r'(?:Herr|Frau|Kolleg(?:e|in|en)|Abgeordnete[rn]?|Minister(?:in)?|Staatssekretär(?:in)?|Dr\.|Professor)\s+(?:(?:Dr\.|Professor|von|de|van|zu)\s+)?([A-ZÄÖÜ][\wäöüß\-]+)')
    ppat = {'AFD': r'\bafd\b|alternative für deutschland', 'CDU/CSU': r'\bcdu\b|\bcsu\b', 'SPD': r'\bspd\b|sozialdemokrat', 'FDP': r'\bfdp\b',
            'Alliance 90/The Greens': r'\bgrünen?\b|bündnis 90', 'The Left Party': r'\blinke\b|\blinkspartei\b'}
    t = P.text.fillna(''); tl = t.str.lower()
    rows = []
    for sid, pid, own_party, txt, low, e in zip(P.id, P.politicianId, P.party, t, tl, P[emi_col]):
        tg = {name2id[m.group(1)] for m in pat.finditer(txt) if m.group(1) in name2id and name2id[m.group(1)] != pid}
        for j in tg: rows.append(dict(src=pid, dst=str(j), kind='mdb', emi=e))
        for p_, rp in ppat.items():
            if p_ != own_party and re.search(rp, low): rows.append(dict(src=pid, dst='party:' + p_, kind='party', emi=e))
    M = pd.DataFrame(rows); E = M.groupby(['src', 'dst', 'kind']).agg(n=('emi', 'size'), emi=('emi', 'mean')).reset_index()
    nodes = mdb[(mdb.n >= min_speeches) & mdb.party.isin(_SEQ_PARTIES)].copy(); keep = set(nodes.politicianId.astype(str))
    E = E.merge(mdb[['politicianId', 'emi']].rename(columns={'politicianId': 'src', 'emi': 'src_mean'}), on='src'); E['emi_dev'] = E.emi - E.src_mean   # relative to the speaker's own mean EMI
    E_mdb = E[(E.kind == 'mdb') & (E.n >= min_pair) & E.src.astype(str).isin(keep) & E.dst.isin(keep)].copy()
    E_party = E[(E.kind == 'party') & (E.n >= min_party) & E.src.astype(str).isin(keep)].copy()
    nodes['in_deg'] = nodes.politicianId.astype(str).map(E_mdb.groupby('dst').n.sum()).fillna(0)
    return nodes, E_mdb, E_party, M


def _cluster_layout(nodes, E_mdb, order=_SEQ_PARTIES, R=1.75, r_cluster=0.55, seed=0):
    """Party clusters on a circle (same angular positions as the party network). Inside a cluster MdBs sit on
    concentric rings: the most-addressed MdBs in the centre, and each MdB on the side of its cluster that faces
    the party it exchanges most mentions with (angle = mention-weighted direction to the other clusters)."""
    ang = {p: np.pi / 2 - 2 * np.pi * i / len(order) for i, p in enumerate(order)}
    centre = {p: (R * np.cos(a), R * np.sin(a)) for p, a in ang.items()}
    ids = nodes.politicianId.astype(str).tolist(); party_of = dict(zip(ids, nodes.party)); indeg = dict(zip(ids, nodes.in_deg))
    vec = {i: np.zeros(2) for i in ids}
    for r in E_mdb.itertuples():
        s_, d_ = str(r.src), str(r.dst)
        if s_ in party_of and d_ in party_of:
            for a_, b_ in ((s_, d_), (d_, s_)):
                if party_of[a_] != party_of[b_]:
                    c = np.array(centre[party_of[b_]]) - np.array(centre[party_of[a_]]); vec[a_] += r.n * c / np.linalg.norm(c)
    rng = np.random.default_rng(seed); pos = {}
    for p in order:
        members = [i for i in ids if party_of[i] == p]
        if not members: continue
        members.sort(key=lambda i: -indeg[i])
        pref = {i: (np.arctan2(vec[i][1], vec[i][0]) if np.linalg.norm(vec[i]) > 0 else rng.uniform(-np.pi, np.pi)) for i in members}
        # rings: ring k (k = 0 centre) holds about 6k nodes
        k, start = 0, 0
        while start < len(members):
            cap = 1 if k == 0 else 6 * k; ring = members[start:start + cap]; start += cap
            rad = 0 if k == 0 else r_cluster * k / max(1, int(np.ceil((-3 + np.sqrt(9 + 12 * (len(members) - 1))) / 6)) + 1)
            ring_sorted = sorted(ring, key=lambda i: pref[i])
            for m_, i in enumerate(ring_sorted):
                th = pref[ring_sorted[0]] + 2 * np.pi * m_ / len(ring_sorted) if len(ring_sorted) > 1 else pref[i]
                # anchor the ring so that the member with the most extreme preference keeps its direction
                pos[i] = (centre[p][0] + rad * np.cos(th), centre[p][1] + rad * np.sin(th))
            k += 1
    return pos, centre


def draw_mdb_mention_network(ax, nodes, E_mdb, E_party, norm, label_parties=True):
    pos, centre = _cluster_layout(nodes, E_mdb)
    nmax = nodes.n.max()
    # soft disc behind each party cluster
    for p_, (cx, cy) in centre.items():
        if (nodes.party == p_).sum() == 0: continue
        ax.add_patch(Circle((cx, cy), 0.64, facecolor=party_colors.get(p_, 'gray'), edgecolor='none', alpha=0.06, zorder=0))
    # party-directed edges: MdB -> the named party's cluster, faint background layer
    wmax_p = E_party.n.max()
    for r in E_party.sort_values('n').itertuples():
        p_ = r.dst.split(':', 1)[1]
        if str(r.src) not in pos or p_ not in centre: continue
        (x0, y0), (x1, y1) = pos[str(r.src)], centre[p_]
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), connectionstyle='arc3,rad=0.3', arrowstyle='-', lw=0.2 + 0.9 * np.sqrt(r.n / wmax_p),
                                     color=_SEQ_EMI_CMAP(norm(r.emi_dev)), alpha=0.07, zorder=1))
    # MdB -> MdB edges: thin bezier arcs, width and opacity by weight, strongest on top
    wmax = E_mdb.n.max()
    for r in E_mdb.sort_values('n').itertuples():
        if str(r.src) not in pos or str(r.dst) not in pos: continue
        (x0, y0), (x1, y1) = pos[str(r.src)], pos[str(r.dst)]
        w = np.sqrt(r.n / wmax)
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), connectionstyle='arc3,rad=0.25', arrowstyle='-|>', mutation_scale=4 + 4 * w,
                                     lw=0.25 + 1.6 * w, color=_SEQ_EMI_CMAP(norm(r.emi_dev)), alpha=0.25 + 0.6 * w, zorder=2, shrinkA=1.5, shrinkB=1.5))
    # nodes: small, size by speeches; glow halo on the ten per cent most-addressed MdBs
    q90 = nodes.in_deg.quantile(0.9)
    for r in nodes.itertuples():
        pid = str(r.politicianId)
        if pid not in pos: continue
        rad = 0.005 + 0.018 * np.sqrt(r.n / nmax); col = party_colors.get(r.party, 'gray')
        if q90 > 0 and r.in_deg >= q90:
            for k, a in [(3.2, 0.10), (2.2, 0.18), (1.5, 0.30)]:
                ax.add_patch(Circle(pos[pid], rad * k, facecolor=col, edgecolor='none', alpha=a, zorder=3))
        ax.add_patch(Circle(pos[pid], rad, facecolor=col, edgecolor='white', lw=0.5, zorder=4))
    if label_parties:
        for p_, (cx, cy) in centre.items():
            if (nodes.party == p_).sum() == 0: continue
            ax.text(cx * 1.42, cy * 1.42, _SEQ_SHORT[p_], ha='center', va='center', fontsize=PLOT_TICK_SIZE + 2, color='black', zorder=6)
    ax.set_xlim(-2.75, 2.75); ax.set_ylim(-2.62, 2.62); ax.set_aspect('equal'); ax.axis('off')


def plot_contagion_sequence(emi_col='emi_w2vparliament', save_path=None, tables_dir=None, example_sitting='19/133', n_perm=2000):
    _white_style = {
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "text.color": "black", "axes.edgecolor": "black", "axes.labelcolor": "black",
        "xtick.color": "black", "ytick.color": "black", "xtick.labelcolor": "black", "ytick.labelcolor": "black",
    }
    with plt.rc_context(_white_style):
        d = load_speech_sequences(emi_col)
        pre = d[(d.date < _SEQ_AFD_ENTRY) & (d.date < '2017-09-06')]
        post = d[d.date >= _SEQ_AFD_ENTRY]
        R_pre, R_post = reply_rows(pre, emi_col), reply_rows(post, emi_col)
        agg = lambda R: R.groupby(['src', 'dst']).agg(n=('diff', 'size'), diff=('diff', 'mean')).reset_index()
        E_pre, E_post = agg(R_pre), agg(R_post)
        N_pre = pre[pre.party.isin(_SEQ_PARTIES)].groupby('party').size()
        N_post = post[post.party.isin(_SEQ_PARTIES)].groupby('party').size()
        prof = distance_profile(post, emi_col)

        # ---- headline statistics (clustered by sitting) --------------------
        stats_rows = []
        for X in _SEQ_PARTIES:
            r = R_post[R_post.src == X]
            stats_rows.append(dict(period='since Oct 2017', party=X, n=len(r), reply_effect=r['diff'].mean(),
                                   se=_cluster_se(r['diff'], r.sitting)))
        for X in [p for p in _SEQ_PARTIES if p != 'AFD']:
            r = R_pre[R_pre.src == X]
            stats_rows.append(dict(period='2015-Sep 2017', party=X, n=len(r), reply_effect=r['diff'].mean(),
                                   se=_cluster_se(r['diff'], r.sitting)))
        ST = pd.DataFrame(stats_rows); ST = ST[ST.n >= 50]
        afd = R_post[R_post.src == 'AFD']; oth = R_post[R_post.src != 'AFD']
        contrast = afd['diff'].mean() - oth['diff'].mean()
        contrast_se = np.sqrt(_cluster_se(afd['diff'], afd.sitting) ** 2 + _cluster_se(oth['diff'], oth.sitting) ** 2)
        print(ST.round(3).to_string(index=False))
        print(f"AfD minus other parties (since Oct 2017): {contrast:+.3f} (s.e. {contrast_se:.3f})")

        # ---- dose: by the preceding speech's EMI quartile ------------------
        afd = afd.assign(q=pd.qcut(afd.emi, 4, labels=False))
        oth = oth.assign(q=oth.groupby('src').emi.transform(lambda s: pd.qcut(s, 4, labels=False, duplicates='drop')))
        dose = []
        for lab, R in [('AfD', afd), ('other parties', oth)]:
            for q in range(4):
                r = R[R.q == q]; dose.append(dict(speaker=lab, quartile=q, n=len(r), effect=r['diff'].mean(), se=_cluster_se(r['diff'], r.sitting)))
        DO = pd.DataFrame(dose)

        # ---- by year -------------------------------------------------------
        yr = []
        for lab, R in [('AfD', afd), ('other parties', oth)]:
            for y, r in R.groupby(pd.to_datetime(R.date).dt.year):
                yr.append(dict(speaker=lab, year=int(y), n=len(r), effect=r['diff'].mean(), se=_cluster_se(r['diff'], r.sitting)))
        YR = pd.DataFrame(yr)

        # ---- reply names the AfD ------------------------------------------
        nm = []
        for lab, m in [('names the AfD', 1), ('does not name the AfD', 0)]:
            r = afd[afd.reply_mentions_afd == m]; nm.append(dict(reply=lab, n=len(r), effect=r['diff'].mean(), se=_cluster_se(r['diff'], r.sitting)))
        NM = pd.DataFrame(nm); print(NM.round(3).to_string(index=False))

        # ---- permutation null and within-speaker effect --------------------
        obs = afd['diff'].mean()
        null = permutation_null_afd(post, emi_col, n_perm=n_perm)
        p_perm = (np.sum(null <= obs) + 1) / (len(null) + 1)
        PERM = pd.DataFrame([dict(observed=obs, null_mean=null.mean(), null_sd=null.std(ddof=1), null_min=null.min(), null_max=null.max(),
                                  z=(obs - null.mean()) / null.std(ddof=1), p_one_sided=p_perm, n_perm=len(null))])
        print(f"permutation null ({len(null)} shuffles of the AfD label within sittings): mean {null.mean():+.4f}, sd {null.std(ddof=1):.4f}, "
              f"min {null.min():+.4f}; observed {obs:+.4f} = {(obs - null.mean()) / null.std(ddof=1):+.1f} sd, one-sided p = {p_perm:.4f}")
        WS, W = within_speaker_afd_effect(post, emi_col); print(WS.round(4).to_string(index=False))
        KK = reply_effect_k(post, emi_col); print('contagion effect by window size k:'); print(KK.round(3).to_string(index=False))

        if tables_dir:
            os.makedirs(tables_dir, exist_ok=True)
            ST.to_csv(os.path.join(tables_dir, 'contagion_sequence_effects.csv'), index=False)
            E_post.to_csv(os.path.join(tables_dir, 'contagion_sequence_edges_post.csv'), index=False)
            E_pre.to_csv(os.path.join(tables_dir, 'contagion_sequence_edges_pre.csv'), index=False)
            prof.to_csv(os.path.join(tables_dir, 'contagion_sequence_profile.csv'), index=False)
            DO.to_csv(os.path.join(tables_dir, 'contagion_sequence_dose.csv'), index=False)
            YR.to_csv(os.path.join(tables_dir, 'contagion_sequence_by_year.csv'), index=False)
            NM.to_csv(os.path.join(tables_dir, 'contagion_sequence_naming.csv'), index=False)
            PERM.to_csv(os.path.join(tables_dir, 'contagion_sequence_permutation.csv'), index=False)
            WS.to_csv(os.path.join(tables_dir, 'contagion_sequence_within_speaker.csv'), index=False)
            KK.to_csv(os.path.join(tables_dir, 'contagion_sequence_k.csv'), index=False)
            pd.DataFrame({'null': null}).to_csv(os.path.join(tables_dir, 'contagion_sequence_permutation_null.csv'), index=False)

        # ---- figure ---------------------------------------------------------
        fig = plt.figure(figsize=(24, 17))
        gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1.45, 0.8], hspace=0.22, wspace=0.04)
        axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1]); axS = fig.add_subplot(gs[1, :])
        norm = mcolors.Normalize(-0.3, 0.3)
        _draw_reply_network(axA, E_post, N_post, '', label_src='AFD', norm=norm)
        # B: MdB-level mention network, edge colour = speaker's EMI when naming the target minus the speaker's own mean
        nodes_m, E_mdb, E_party, M_all = mdb_mention_network(emi_col)
        draw_mdb_mention_network(axB, nodes_m, E_mdb, E_party, norm)
        print("MdB-MdB edge EMI deviation from the speaker's mean: median {:+.2f}, IQR {:+.2f} to {:+.2f}".format(E_mdb.emi_dev.median(), E_mdb.emi_dev.quantile(.25), E_mdb.emi_dev.quantile(.75)))
        sm = plt.cm.ScalarMappable(cmap=_SEQ_EMI_CMAP, norm=norm)
        cb = fig.colorbar(sm, ax=[axA, axB], orientation='horizontal', fraction=0.025, pad=0.01, aspect=60, shrink=0.55)
        cb.set_label("Change in EMI: next minus previous speech by another party (A); speech naming the target minus the speaker's own mean (B)", fontsize=PLOT_LEGEND_SIZE); cb.ax.tick_params(labelsize=PLOT_TICK_SIZE)
        print(f"MdB network: {len(nodes_m)} MdBs with >= 30 speeches, {len(E_mdb)} MdB-MdB edges (>= 3 mentions, {int(E_mdb.n.sum())} mentions), {len(E_party)} MdB-party edges (>= 25 mentions); "
              f"cross-party share of MdB-MdB mentions {(E_mdb.merge(nodes_m[['politicianId','party']].astype({'politicianId':str}).rename(columns={'politicianId':'dst','party':'party_dst'}), on='dst').pipe(lambda x: (x.party_dst != x.src.map(nodes_m.set_index('politicianId').party)).mean())):.2f}")
        if tables_dir:
            E_mdb.to_csv(os.path.join(tables_dir, 'contagion_mention_edges_mdb.csv'), index=False); E_party.to_csv(os.path.join(tables_dir, 'contagion_mention_edges_party.csv'), index=False)
            nodes_m.to_csv(os.path.join(tables_dir, 'contagion_mention_nodes.csv'), index=False)
        for ax, letter in [(axA, 'A'), (axB, 'B')]:
            x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
            ax.text(x0 - 0.08 * (x1 - x0), y1, letter, fontsize=PLOT_PANEL_SIZE, fontweight='bold', va='top', ha='left', clip_on=False)

        # C: one sitting as a strip -------------------------------------------
        ex = post[post.Sitting == example_sitting].reset_index(drop=True)
        ex['seq'] = np.arange(1, len(ex) + 1)
        is_afd = (ex.party == 'AFD').to_numpy(); non = ex[~is_afd].copy()
        non['run'] = non[emi_col].rolling(3, center=True, min_periods=2).mean()
        # contagion effect around each AfD speech (next minus previous speech by another party)
        reply, seen = [], set()
        for i in np.where(is_afd)[0]:
            b = next((j for j in range(i - 1, -1, -1) if not is_afd[j]), None); a = next((j for j in range(i + 1, len(ex)) if not is_afd[j]), None)
            if b is not None and a is not None and (b, a) not in seen:   # consecutive AfD speeches share one reply
                seen.add((b, a)); reply.append((i, b, a, ex[emi_col].iloc[a] - ex[emi_col].iloc[b]))
        norm_s = mcolors.Normalize(-1.5, 1.5)
        ymin, ymax = ex[emi_col].min(), ex[emi_col].max(); pad = 0.12 * (ymax - ymin)
        for i in np.where(is_afd)[0]:
            axS.axvspan(i + 0.5, i + 1.5, color=party_colors['AFD'], alpha=0.10, lw=0, zorder=1)
        for i, b, a, dv in reply:
            col = _SEQ_EMI_CMAP(norm_s(dv))
            axS.annotate('', xy=(a + 1, ex[emi_col].iloc[a]), xytext=(b + 1, ex[emi_col].iloc[b]),
                         arrowprops=dict(arrowstyle='-|>', color=col, lw=2.2, mutation_scale=18, connectionstyle='arc3,rad=-0.25', alpha=0.9), zorder=4)
        axS.plot(non.seq, non.run, color='black', lw=2.0, alpha=0.75, zorder=2, label='other parties, running mean (3 speeches)')
        for X in _SEQ_PARTIES:
            gx = ex[ex.party == X]
            if len(gx) == 0:
                continue
            axS.scatter(gx.seq, gx[emi_col], s=150 if X == 'AFD' else 70, marker='D' if X == 'AFD' else 'o',
                        color=party_colors[X], edgecolor='white', linewidth=1.0, zorder=5, label=_SEQ_SHORT[X])
        mm = non[non.mentions_afd == 1]
        axS.scatter(mm.seq, mm[emi_col], s=330, facecolor='none', edgecolor='#D62728', linewidth=2.0, zorder=6, label='speech names the AfD')
        axS.axhline(0, color='black', lw=0.8, alpha=0.6); axS.grid(True, axis='y', alpha=0.25)
        axS.set_xlim(0.3, len(ex) + 0.7); axS.set_ylim(ymin - pad * 0.6, ymax + pad * 0.6)
        axS.set_yticks(np.arange(np.ceil(ymin), np.floor(ymax) + 1))
        axS.set_xlabel('Speech order within the sitting', fontsize=PLOT_LABEL_SIZE - 4, fontweight='bold')
        axS.set_ylabel('EMI of the speech', fontsize=PLOT_LABEL_SIZE - 4, fontweight='bold')
        axS.tick_params(labelsize=PLOT_TICK_SIZE)
        rr = np.array([dv for *_, dv in reply])
        axS.set_title(f"Sitting {example_sitting}, {ex.date.iloc[0].strftime('%d %B %Y')}", fontsize=PLOT_LEGEND_SIZE + 2, loc='left', pad=8)
        print(f"example sitting {example_sitting}: {len(ex)} speeches, {int(is_afd.sum())} by the AfD; after {int((rr < 0).sum())} of {len(rr)} AfD interventions the next speech is more intuition-based (mean contagion effect {rr.mean():+.2f})")
        axS.legend(frameon=False, fontsize=PLOT_LEGEND_SIZE, loc='upper left', bbox_to_anchor=(0.0, -0.14), ncol=8, handletextpad=0.4, columnspacing=1.2)
        axS.text(-0.05, 1.03, 'C', transform=axS.transAxes, fontsize=PLOT_PANEL_SIZE, fontweight='bold', va='bottom', ha='left')

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white'); print(f'Saved to {save_path}')
            fig.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        # supplementary: permutation null distribution
        figp, axp = plt.subplots(figsize=(10, 6))
        axp.hist(null, bins=60, color='#BBBBBB', edgecolor='white')
        axp.axvline(obs, color=party_colors['AFD'], lw=3)
        axp.text(obs + 0.004, axp.get_ylim()[1] * 0.95, 'observed AfD contagion effect {:+.3f}, {:+.1f} s.d. below the null (p < {:.4f})'.format(obs, (obs - null.mean()) / null.std(ddof=1), max(p_perm, 1 / (len(null) + 1))),
                 ha='left', va='top', fontsize=PLOT_TICK_SIZE, color=party_colors['AFD'], wrap=True)
        axp.set_xlabel('Contagion effect after AfD-labelled speeches (EMI)', fontsize=PLOT_LABEL_SIZE - 4, fontweight='bold')
        axp.set_ylabel('Permutations', fontsize=PLOT_LABEL_SIZE - 4, fontweight='bold'); axp.tick_params(labelsize=PLOT_TICK_SIZE); axp.grid(True, alpha=0.25)
        axp.set_title('AfD label reassigned at random within each sitting ({:,} permutations)'.format(len(null)), fontsize=PLOT_LEGEND_SIZE + 1, loc='left')
        if save_path:
            pp = save_path.replace('.pdf', '_permutation.pdf'); figp.savefig(pp, dpi=300, bbox_inches='tight', facecolor='white'); figp.savefig(pp.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', facecolor='white'); print('Saved to ' + pp)
        plt.show()
        return fig, dict(effects=ST, permutation=PERM, within_speaker=WS, by_k=KK, edges_pre=E_pre, edges_post=E_post, profile=prof, dose=DO, by_year=YR, naming=NM, example=ex)


contagion_fig, contagion_results = plot_contagion_sequence(
    save_path='results/contagion_sequence.pdf',
    tables_dir='results/parliament/tables')
