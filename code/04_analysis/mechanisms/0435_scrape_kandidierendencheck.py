"""Collect Kandidierenden-Check thesis answers of Bundestag mandate holders from abgeordnetenwatch.de.

Source: the per-thesis modal https://www.abgeordnetenwatch.de/cc/teaser/<politician_id>/<check_id>/<n>?_wrapper_format=drupal_modal
(CC0 content of abgeordnetenwatch.de). Polite: one request per ~1.3 s, backoff on 429, resumable through a JSON cache
per politician. Usage:
  python code/04_analysis/mechanisms/0435_scrape_kandidierendencheck.py --period 111 --check 14     # 19th Bundestag MdBs, 2017 check
  python code/04_analysis/mechanisms/0435_scrape_kandidierendencheck.py --period 132 --check 58     # 20th Bundestag MdBs, 2021 check
Output: data/revision/abgeordnetenwatch/answers_<period>_<check>.csv (one row per politician x thesis) + cache/.
"""
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))   # German-EMI repository root: data in data/, outputs in results/
import os, sys, re, json, time, html, argparse, subprocess, random
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(REPO_ROOT, 'data', 'revision', 'abgeordnetenwatch')
UA = 'Mozilla/5.0 (academic research; contact peer.saleth@uni-konstanz.de)'
ap = argparse.ArgumentParser(); ap.add_argument('--period', type=int, required=True); ap.add_argument('--check', type=int, required=True)
ap.add_argument('--sleep', type=float, default=1.3); ap.add_argument('--limit', type=int, default=None)
a = ap.parse_args()
CACHE = os.path.join(D, 'cache', f'{a.period}_{a.check}'); os.makedirs(CACHE, exist_ok=True)
LOG = open(os.path.join(D, f'scrape_{a.period}_{a.check}.log'), 'a', encoding='utf-8')
def log(s): print(s, flush=True); LOG.write(s + '\n'); LOG.flush()

def fetch(url, tries=6):
    for k in range(tries):
        r = subprocess.run(['curl', '-sL', '--max-time', '60', '-A', UA, '-H', 'X-Requested-With: XMLHttpRequest', '-w', '\n%{http_code}', url],
                           capture_output=True, text=True, encoding='utf-8', errors='replace')
        body, _, code = r.stdout.rpartition('\n')
        if code == '200': return body
        if code in ('429', '503', '502', '000'):
            wait = 15 * (k + 1) + random.uniform(0, 5); log(f'  http {code}, waiting {wait:.0f}s'); time.sleep(wait); continue
        return None
    return None

def modal_html(pid, n):
    t = fetch(f'https://www.abgeordnetenwatch.de/cc/teaser/{pid}/{a.check}/{n}?_wrapper_format=drupal_modal')
    if t is None: return None
    try: cmds = json.loads(t[:t.rfind(']') + 1])
    except Exception: return None
    for c in cmds:
        if c.get('command') == 'openDialog': return c.get('data', '')
    return ''

def parse(h):
    g = lambda cls: [re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', html.unescape(m))).strip() for m in re.findall(r'class="' + cls + r'[^"]*"[^>]*>(.*?)</(?:div|p|span|h\d)>', h, re.S)]
    pag = re.search(r'(\d+)\s*</[^>]+>\s*<[^>]+>\s*/\s*(\d+)', re.sub(r'\s+', ' ', h)) or re.search(r'(\d+)\s*/\s*(\d+)', re.sub(r'<[^>]+>', ' ', h))
    total = int(pag.group(2)) if pag else None
    topic = (g('assumption-detail__sidebar__quote__topic') or [''])[0]
    quote = g('assumption-detail__sidebar__quote__inner'); thesis = quote[0] if quote else ''
    thesis = re.sub(r'^' + re.escape(topic) + r'\s*', '', thesis).strip()
    rating = (g('assumption-detail__content__rating') or [''])[0]
    reason = (g('assumption-detail__content__reason') or [''])[0]
    return dict(total=total, topic=topic, thesis=thesis, rating=rating, reason=reason)

mand = json.load(open(os.path.join(D, f'mandates_{a.period}.json'), encoding='utf-8'))
pols = {}
for c in mand:
    p = c['politician']; fr = (c.get('fraction_membership') or [{}]); frac = fr[0].get('fraction', {}).get('label') if fr and fr[0] else None
    pols[p['id']] = dict(aw_id=p['id'], name=p['label'], party=(c.get('party') or {}).get('label'), fraction=frac,
                         constituency=((c.get('electoral_data') or {}).get('constituency') or {}).get('label'))
log(f'period {a.period} check {a.check}: {len(pols)} mandate holders')
rows = []; done = 0
for i, (pid, info) in enumerate(list(pols.items())[:a.limit]):
    cf = os.path.join(CACHE, f'{pid}.json')
    if os.path.exists(cf):
        rec = json.load(open(cf, encoding='utf-8'))
    else:
        rec = dict(info, participated=False, answers=[])
        h = modal_html(pid, 1); time.sleep(a.sleep)
        if h:
            first = parse(h); rec['participated'] = True; rec['answers'].append(dict(n=1, **first))
            total = first['total'] or 24
            for n in range(2, total + 1):
                hn = modal_html(pid, n); time.sleep(a.sleep)
                if not hn: break
                rec['answers'].append(dict(n=n, **parse(hn)))
        json.dump(rec, open(cf, 'w', encoding='utf-8'), ensure_ascii=False)
        done += 1
    for ans in rec['answers']:
        rows.append(dict(aw_id=pid, name=info['name'], party=info['party'], fraction=info['fraction'], constituency=info['constituency'], **{k: v for k, v in ans.items() if k != 'total'}))
    if (i + 1) % 25 == 0:
        log(f'  {i + 1}/{len(pols)} politicians, participants so far {sum(1 for f in os.listdir(CACHE))}')
out = pd.DataFrame(rows); out.to_csv(os.path.join(D, f'answers_{a.period}_{a.check}.csv'), index=False)
part = sum(1 for f in os.listdir(CACHE) if json.load(open(os.path.join(CACHE, f), encoding='utf-8'))['participated'])
log(f'finished: {len(pols)} MdBs, {part} participated, {len(out)} answers -> answers_{a.period}_{a.check}.csv')
