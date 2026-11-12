"""Reproduce every number on the poster and in the paper "What Recall Hides".

    python reproduce.py                 # measurements -> all 223 values (seconds); corpora if already downloaded
    python reproduce.py --download      # also download the four corpora (8.7 GB) and check them
    python reproduce.py --figures       # also redraw figures/recall.png and figures/occupancy.png
    python reproduce.py -v              # print every value, not only the summary

Step 1 recomputes each value from the measurements in data/ and compares it with the number as printed.
Step 2 checks the corpora themselves (via download.py): checksum, size, dimension, metric and query count, and an
exact top-10 search for a sample of queries against the ground truth shipped with each corpus.
Exit status: 0 if everything matches, 1 if any value differs, 2 on a usage or setup problem.
"""
import argparse
import csv
import sys
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError as e:
    sys.exit(f'missing package: {e.name}. Install the requirements first:  pip install -r requirements.txt')

HERE = Path(__file__).resolve().parent
DATA, OUT = HERE / 'data', HERE / 'out'
CORPORA = ['Deep10M', 'GloVe', 'SIFT', 'GIST']
METHODS = ['HNSW', 'IVF', 'IVFPQ+R', 'BLISS', 'LIRA']
TARGET = 0.90          # recall@10 at which recall@1 and recall@100 are compared
PROBES = 40            # BLISS probe depth for the initialization comparison
BLISS_TABLES, BLISS_MIN_REPS = 4, 2


def r2(v):   # .89 style, as printed on the poster
    return f'{v:.2f}'.lstrip('0') if v >= 0 else '-' + f'{-v:.2f}'.lstrip('0')


def r3(v):
    return f'{v:.3f}'.lstrip('0') if v >= 0 else '-' + f'{-v:.3f}'.lstrip('0')


def signed3(v):
    s = f'{v:+.3f}'.replace('+0.', '+.').replace('-0.', '-.')
    return '+.000' if s in ('+.000', '-.000') else s


def rng(lo, hi, fmt):
    return f'{fmt(lo)}–{fmt(hi)}'


# ------------------------------------------------------------------------------------------------ inputs
corp = pd.read_csv(DATA / 'corpora.csv').set_index('corpus')
op = pd.read_csv(DATA / 'operating_points.csv')
rep = pd.read_csv(DATA / 'operating_points_repeat.csv')
ini = pd.read_csv(DATA / 'bliss_initialization.csv')
lists = pd.read_csv(DATA / 'list_sizes.csv.gz')

# recall per operating point; BLISS is the mean over its three seeds
points = op.groupby(['corpus', 'method', 'operating_point'])[['recall_at_1', 'recall_at_10', 'recall_at_100']].mean().reset_index()

checks = []


def check(section, item, printed, computed):
    checks.append((section, item, printed, computed, printed == computed))


# ------------------------------------------------------------------------------------------------ corpora
for c, (n, d) in {'Deep10M': ('9.99 M', 96), 'GloVe': ('1.18 M', 100), 'SIFT': ('1 M', 128), 'GIST': ('1 M', 960)}.items():
    v = int(corp.loc[c, 'vectors'])
    millions = f'{v // 1_000_000}' if v % 1_000_000 == 0 else f'{v / 1e6:.2f}'
    check('Poster Corpora', f'{c} size', f'{n} x {d}-d', f"{millions} M x {corp.loc[c, 'dim']}-d")
check('Poster Corpora', 'queries (Deep10M, SIFT / GloVe, GIST)', '10,000 / 1,000',
      f"{corp.loc['Deep10M', 'queries']:,} / {corp.loc['GloVe', 'queries']:,}")

# ------------------------------------------------------------------------------------------------ RQ1: coverage
shared, at90 = {}, {}
for c in CORPORA:
    g = points[points.corpus == c]
    r = {m: (g[g.method == m].recall_at_10.min(), g[g.method == m].recall_at_10.max()) for m in METHODS}
    lo, hi = max(v[0] for v in r.values()), min(v[1] for v in r.values())
    shared[c] = (lo, hi) if hi >= lo else None
    at90[c] = {}
    for m in METHODS:
        s = g[g.method == m].sort_values('recall_at_10')
        x = s.recall_at_10.values
        if x.min() <= TARGET <= x.max():
            at90[c][m] = {k: float(np.interp(TARGET, x, s[k].values)) for k in ('recall_at_1', 'recall_at_100')}
printed_shared = {'Deep10M': '.89–.98', 'GloVe': '.83–.87', 'SIFT': 'none', 'GIST': 'none'}
for c in CORPORA:
    check('Poster RQ1', f'{c} shared recall@10 range', printed_shared[c], rng(*shared[c], r2) if shared[c] else 'none')
check('Poster Key results', 'corpora where the recall@10 ranges never overlap', '2 of 4', f'{sum(v is None for v in shared.values())} of 4')

printed_r1 = {'Deep10M': ['.940', '.924', '.928', '.923', '.917'], 'GloVe': ['.935', '.934', '–', '.926', '.923'],
              'SIFT': ['–', '.923', '.924', '–', '–'], 'GIST': ['.916', '.914', '–', '.915', '.930']}
printed_d1 = {'Deep10M': '.02', 'GloVe': '.01', 'SIFT': '<.01', 'GIST': '.02'}
printed_d100 = {'Deep10M': '.11', 'GloVe': '.05', 'SIFT': '.04', 'GIST': '.05'}
printed_n = {'Deep10M': '5 of 5', 'GloVe': '4 of 5', 'SIFT': '2 of 5', 'GIST': '4 of 5'}
for c in CORPORA:
    for m, p in zip(METHODS, printed_r1[c]):
        check('Poster RQ1', f'{c} {m} recall@1 at recall@10 = .90', p, r3(at90[c][m]['recall_at_1']) if m in at90[c] else '–')
    v1 = [v['recall_at_1'] for v in at90[c].values()]
    v100 = [v['recall_at_100'] for v in at90[c].values()]
    d1 = max(v1) - min(v1)
    check('Poster RQ1', f'{c} spread of recall@1 at .90', printed_d1[c], '<.01' if d1 < 0.005 else r2(d1))
    check('Poster RQ1', f'{c} spread of recall@100 at .90', printed_d100[c], r2(max(v100) - min(v100)))
    check('Poster RQ1', f'{c} methods whose sweep crosses .90', printed_n[c], f'{len(at90[c])} of 5')
check('Poster RQ1', 'largest recall@100 spread at .90 ("up to")', '.11', r2(max(max(v['recall_at_100'] for v in at90[c].values()) - min(v['recall_at_100'] for v in at90[c].values()) for c in CORPORA)))
check('Poster RQ1', 'largest recall@1 spread at .90 ("within")', '.02', r2(max(max(v['recall_at_1'] for v in at90[c].values()) - min(v['recall_at_1'] for v in at90[c].values()) for c in CORPORA)))
b = points[(points.corpus == 'SIFT') & (points.method == 'BLISS')].set_index('operating_point').recall_at_10
check('Poster RQ1', 'SIFT BLISS: 20 probes worse than 10', 'True', str(bool(b['probe-20'] < b['probe-10'])))

# ------------------------------------------------------------------------------------------------ RQ2: initialization and occupancy
at = ini[ini.probe_depth == PROBES]
printed_gain = {'Deep10M': '+.000', 'GloVe': '-.007', 'SIFT': '+.209', 'GIST': '-.031'}
printed_ratio = {'Deep10M': '1.25', 'GloVe': '0.84', 'SIFT': '0.47', 'GIST': '0.99'}
for c in CORPORA:
    h = at[(at.corpus == c) & (at.start == 'hash')]                       # mean of the three hash seeds
    k = at[(at.corpus == c) & (at.start == 'k-means') & (at.seed == 0)]   # one k-means seed
    check('Poster RQ2', f'{c} recall@10 gain, k-means - hash', printed_gain[c], signed3(k.recall_at_10.mean() - h.recall_at_10.mean()))
    check('Poster RQ2', f'{c} vectors scored, k-means / hash', printed_ratio[c], f'{k.avg_candidates.mean() / h.avg_candidates.mean():.2f}')


def layout(c, name):
    return lists[(lists.corpus == c) & (lists.layout == name)].entries.values


def top_share(s, frac=0.10):  # share of entries in the largest frac of lists, interpolated as on the plotted curve
    s = np.sort(s.astype(float))[::-1]
    cum = np.concatenate([[0.0], np.cumsum(s) / s.sum()])
    return float(np.interp(frac * len(s), np.arange(len(s) + 1), cum))


printed_lm = {'BLISS hash': ['20.4', '15.7', '82.0', '42.1'], 'BLISS k-means': ['14.4', '10.2', '15.5', '10.8'], 'IVF': ['4.0', '2.8', '3.5', '5.3']}
for name, vals in printed_lm.items():
    for c, p in zip(CORPORA, vals):
        s = layout(c, name)
        check('Poster RQ2', f'{c} {name} largest / mean list', p, f'{s.max() / s.mean():.1f}')
sh, sk = layout('SIFT', 'BLISS hash'), layout('SIFT', 'BLISS k-means')
check('Poster RQ2', 'SIFT BLISS hash empty lists', '2,419 of 4,096', f'{(sh == 0).sum():,} of {len(sh):,}')
check('Poster RQ2', 'SIFT BLISS k-means empty lists', '20', f'{(sk == 0).sum()}')
n_sift = corp.loc['SIFT', 'vectors']
raw = 2 * BLISS_TABLES * n_sift * PROBES / (corp.loc['SIFT', 'ivf_lists'] * BLISS_MIN_REPS)
cap = 1024 * int(raw // 1024)   # BLISS candidate buffer: 2 R N m / (B min_reps), rounded down to a multiple of 1024
hs0 = at[(at.corpus == 'SIFT') & (at.start == 'hash') & (at.seed == 0)].avg_candidates.iloc[0]
check('Poster RQ2', 'SIFT BLISS hash candidates scored, seed 0, of the buffer', '155,647 of 155,648', f'{int(hs0):,} of {cap:,}')
check('Poster RQ2', 'SIFT entries in the largest 10% of lists, BLISS hash', '95.5%', f'{100 * top_share(sh):.1f}%')
check('Poster RQ2', 'SIFT entries in the largest 10% of lists, IVF', '17.0%', f"{100 * top_share(layout('SIFT', 'IVF')):.1f}%")

# ------------------------------------------------------------------------------------------------ RQ3: costs and variability
share = {}
for c in CORPORA:
    g = op[(op.corpus == c) & (op.method == 'IVFPQ+R')]
    share[c] = g.faiss_rerank_s.sum() / (g.faiss_rerank_s.sum() + g.faiss_search_s.sum())
for c, p in zip(CORPORA, ['54%', '67%', '81%', '91%']):
    check('Poster RQ3', f'{c} IVFPQ+R re-rank / (search + re-rank)', p, f'{100 * share[c]:.0f}%')
check('Poster Key results', 'IVFPQ+R time in exact re-rank', '54–91%', f'{100 * min(share.values()):.0f}–{100 * max(share.values()):.0f}%')

build = op.groupby(['corpus', 'method']).build_total_s.mean()
ratio = {m: [build[c, m] / build[c, 'IVF'] for c in CORPORA] for m in ['HNSW', 'IVFPQ+R', 'BLISS', 'LIRA']}
for m, p, f in (('HNSW', '1.2–3.0×', '.1f'), ('IVFPQ+R', '1.0–1.4×', '.1f'), ('BLISS', '15–61×', '.0f'), ('LIRA', '224–724×', '.0f')):
    check('Poster RQ3', f'{m} build time / IVF build', p, f'{min(ratio[m]):{f}}–{max(ratio[m]):{f}}×')
learned = ratio['BLISS'] + ratio['LIRA']
check('Poster Key results', 'BLISS and LIRA build time / IVF build', '15–724×', f'{min(learned):.0f}–{max(learned):.0f}×')

spread = 0.0
for c in CORPORA:
    g = op[(op.corpus == c) & (op.method == 'BLISS')].pivot_table(index='operating_point', columns='seed', values='recall_at_10')
    spread = max(spread, float((g.max(axis=1) - g.min(axis=1)).max()))
check('Poster RQ3', 'BLISS: 3 hash seeds move recall@10 by up to', '.04', r2(spread))
for m in ('IVF', 'IVFPQ+R'):
    d = 0.0
    for c in CORPORA:
        x = op[(op.corpus == c) & (op.method == m)].groupby('operating_point').recall_at_10.mean()
        y = rep[(rep.corpus == c) & (rep.method == m)].groupby('operating_point').recall_at_10.mean()
        j = x.index.intersection(y.index)
        assert len(j) == len(x) == len(y)
        d = max(d, float((x[j] - y[j]).abs().max()))
    check('Poster RQ3', f'{m} repeats recall@10 exactly across two runs', 'True', str(d == 0.0))


# ================================================================================================ paper (3-page version)
# Tables II-VI and every number in the text, as printed.
f2 = lambda v: f'{v:.2f}'
f3 = lambda v: f'{v:.3f}'
T2 = {'Deep10M': ['0.89–0.99', '0.86–0.99', '0.85–0.98', '0.83–0.99', '0.89–1.00', '0.89–0.98', '0.94/0.76', '0.92/0.86', '0.93/0.76', '0.92/0.86', '0.92/0.87'],
      'GloVe': ['0.76–0.94', '0.75–0.94', '0.74–0.87', '0.71–0.95', '0.83–0.98', '0.83–0.87', '0.93/0.80', '0.93/0.84', '↓0.87', '0.93/0.86', '0.92/0.85'],
      'SIFT': ['0.94–1.00', '0.87–1.00', '0.87–1.00', '0.70–0.79', '0.92–1.00', 'none', '↑0.94', '0.92/0.84', '0.92/0.80', '↓0.79', '↑0.92'],
      'GIST': ['0.72–0.96', '0.70–0.99', '0.59–0.68', '0.67–0.99', '0.87–1.00', 'none', '0.92/0.81', '0.91/0.85', '↓0.68', '0.92/0.86', '0.93/0.84']}
for c in CORPORA:
    g = points[points.corpus == c]
    for m, p in zip(METHODS, T2[c][:5]):
        x = g[g.method == m].recall_at_10
        check('Paper T2', f'{c} {m} recall@10 range', p, f'{f2(x.min())}–{f2(x.max())}')
    check('Paper T2', f'{c} shared range', T2[c][5], f'{f2(shared[c][0])}–{f2(shared[c][1])}' if shared[c] else 'none')
    for m, p in zip(METHODS, T2[c][6:]):
        if m in at90[c]:
            got = f"{f2(at90[c][m]['recall_at_1'])}/{f2(at90[c][m]['recall_at_100'])}"
        else:
            x = g[g.method == m].recall_at_10
            got = f'↑{f2(x.min())}' if x.min() > TARGET else f'↓{f2(x.max())}'
        check('Paper T2', f'{c} {m} recall@1/recall@100 at .90', p, got)

T3 = {'Deep10M': ['0.989', '0.990', '173.4', '216.2'], 'GloVe': ['0.953', '0.944', '119.5', '99.9'],
      'SIFT': ['0.803', '0.995', '155.6', '73.1'], 'GIST': ['0.984', '0.956', '144.5', '142.6']}
s0 = at[at.seed == 0].set_index(['corpus', 'start'])
for c in CORPORA:
    h, k = s0.loc[(c, 'hash')], s0.loc[(c, 'k-means')]
    for item, p, v in (('hash recall@10', T3[c][0], f3(h.recall_at_10)), ('k-means recall@10', T3[c][1], f3(k.recall_at_10)),
                       ('hash candidates (thousands)', T3[c][2], f'{h.avg_candidates / 1e3:.1f}'), ('k-means candidates (thousands)', T3[c][3], f'{k.avg_candidates / 1e3:.1f}')):
        check('Paper T3', f'{c} {item}, seed 0', p, v)

T4 = {'Deep10M': ['20.4', '14.4', '4.0', '7.5', '293', '1259'], 'GloVe': ['15.7', '10.2', '2.8', '10.2', '175', '205'],
      'SIFT': ['82.0', '15.5', '3.5', '5.8', '2419', '20'], 'GIST': ['42.1', '10.8', '5.3', '10.7', '1490', '1319']}
for c in CORPORA:
    for name, p in zip(('BLISS hash', 'BLISS k-means', 'IVF', 'LIRA'), T4[c][:4]):
        s_ = layout(c, name)
        check('Paper T4', f'{c} {name} largest / mean list', p, f'{s_.max() / s_.mean():.1f}')
    for name, p in zip(('BLISS hash', 'BLISS k-means'), T4[c][4:]):
        check('Paper T4', f'{c} {name} empty lists', p, f'{(layout(c, name) == 0).sum()}')

T5 = {'Deep10M': ['96', '0.027–0.153', '0.092', '54'], 'GloVe': ['100', '0.022–0.095', '0.111', '67'],
      'SIFT': ['128', '0.015–0.065', '0.153', '81'], 'GIST': ['960', '0.034–0.113', '0.675', '91']}
for c in CORPORA:
    g = op[(op.corpus == c) & (op.method == 'IVFPQ+R')]
    q = corp.loc[c, 'queries']
    se, rr = g.faiss_search_s / q * 1e3, g.faiss_rerank_s / q * 1e3
    check('Paper T5', f'{c} dimension', T5[c][0], f"{corp.loc[c, 'dim']}")
    check('Paper T5', f'{c} search ms/query, range', T5[c][1], f'{se.min():.3f}–{se.max():.3f}')
    check('Paper T5', f'{c} re-rank ms/query, mean', T5[c][2], f'{rr.mean():.3f}')
    check('Paper T5', f'{c} re-rank share (%)', T5[c][3], f'{100 * share[c]:.0f}')

T6 = {'Deep10M': ['181', '2.2', '1.1', '26', '330', '4.4'], 'GloVe': ['21', '3.0', '1.2', '61', '330', '4.3'],
      'SIFT': ['24', '1.2', '1.0', '53', '724', '4.0'], 'GIST': ['112', '1.2', '1.4', '15', '224', '3.1']}
fp = op.groupby(['corpus', 'method']).serving_footprint_mb.mean()
for c in CORPORA:
    check('Paper T6', f'{c} IVF build (s)', T6[c][0], f"{build[c, 'IVF']:.0f}")
    for m, p, f in zip(('HNSW', 'IVFPQ+R', 'BLISS', 'LIRA'), T6[c][1:5], ('.1f', '.1f', '.0f', '.0f')):
        check('Paper T6', f'{c} {m} build / IVF build', p, f'{build[c, m] / build[c, "IVF"]:{f}}')
    check('Paper T6', f'{c} LIRA footprint / IVF footprint', T6[c][5], f"{fp[c, 'LIRA'] / fp[c, 'IVF']:.1f}")

# numbers in the text
bl = points[(points.corpus == 'SIFT') & (points.method == 'BLISS')].recall_at_10.max()
others = min(points[(points.corpus == 'SIFT') & (points.method == m)].recall_at_10.min() for m in METHODS if m != 'BLISS')
check('Paper text', 'SIFT: BLISS maximum below lowest other minimum', '0.79 < 0.87', f'{f2(bl)} < {f2(others)}')
pq_max = points[(points.corpus == 'GIST') & (points.method == 'IVFPQ+R')].recall_at_10.max()
overlap = [m for m in METHODS if m != 'IVFPQ+R' and points[(points.corpus == 'GIST') & (points.method == m)].recall_at_10.min() <= pq_max]
check('Paper text', 'GIST: IVFPQ+R maximum, and the methods it overlaps', '0.68, BLISS', f"{f2(pq_max)}, {', '.join(overlap)}")
hs, ks = s0.loc[('SIFT', 'hash')], s0.loc[('SIFT', 'k-means')]
check('Paper text', 'SIFT k-means gain at 40 probes, seed 0', '0.192', f3(ks.recall_at_10 - hs.recall_at_10))
check('Paper text', 'SIFT fewer scored candidates', '53%', f'{100 * (1 - ks.avg_candidates / hs.avg_candidates):.0f}%')
sg = ini[(ini.corpus == 'SIFT') & (ini.seed == 0)].pivot_table(index='probe_depth', columns='start', values='recall_at_10')
gains = sg['k-means'] - sg['hash']
check('Paper text', 'SIFT gain range over 5/10/20/40 probes', '0.119–0.221', f'{f3(gains.min())}–{f3(gains.max())}')
check('Paper text', 'SIFT hash candidates and buffer, k-means candidates', '155,647 / 155,648 / 73,146', f'{int(hs.avg_candidates):,} / {cap:,} / {int(ks.avg_candidates):,}')
check('Paper text', 'SIFT hash empty lists', '59%', f'{100 * (sh == 0).mean():.0f}%')
check('Paper text', 'SIFT largest 10% of lists: hash, k-means, IVF', '95.5% / 33.8% / 17.0%',
      ' / '.join(f'{100 * top_share(layout("SIFT", n)):.1f}%' for n in ('BLISS hash', 'BLISS k-means', 'IVF')))
lt = {c: {n: top_share(layout(c, n)) for n in ('IVF', 'BLISS hash', 'BLISS k-means', 'LIRA')} for c in CORPORA}
check('Paper text', 'all learned layouts more concentrated than IVF', 'True',
      str(all(lt[c][n] > lt[c]['IVF'] for c in CORPORA for n in ('BLISS hash', 'BLISS k-means', 'LIRA'))))
grow, s10, s100 = [], [], []
for c in CORPORA:
    g = op[(op.corpus == c) & (op.method == 'IVFPQ+R')].copy()
    g['nprobe'] = g.operating_point.str.extract(r'(\d+)').astype(int)
    g = g.set_index('nprobe')
    grow.append(g.faiss_search_s[100] / g.faiss_search_s[10])
    s10.append(g.faiss_rerank_s[10] / (g.faiss_rerank_s[10] + g.faiss_search_s[10]))
    s100.append(g.faiss_rerank_s[100] / (g.faiss_rerank_s[100] + g.faiss_search_s[100]))
check('Paper text', 'IVFPQ+R search time growth from 10 to 100 probes', '3.4–5.6×', f'{min(grow):.1f}–{max(grow):.1f}×')
check('Paper text', 'IVFPQ+R re-rank share at 10 probes', '77–95%', f'{100 * min(s10):.0f}–{100 * max(s10):.0f}%')
check('Paper text', 'IVFPQ+R re-rank share at 100 probes', '37–85%', f'{100 * min(s100):.0f}–{100 * max(s100):.0f}%')
bt = [op[(op.corpus == c) & (op.method == m)].batch1_wall_s_median.mean() / op[(op.corpus == c) & (op.method == m)].batch256_wall_s_median.mean()
      for c in CORPORA for m in ('HNSW', 'IVF', 'IVFPQ+R')]
check('Paper text', 'batch-1 over batch-256 wall time per query', '1.9–19.9×', f'{min(bt):.1f}–{max(bt):.1f}×')
check('Paper text', 'HNSW and IVFPQ+R build over IVF', '1.0–3.0×', f"{min(ratio['HNSW'] + ratio['IVFPQ+R']):.1f}–{max(ratio['HNSW'] + ratio['IVFPQ+R']):.1f}×")
hn = 0.0
for c in CORPORA:
    x = op[(op.corpus == c) & (op.method == 'HNSW')].groupby('operating_point').recall_at_10.mean()
    y = rep[(rep.corpus == c) & (rep.method == 'HNSW')].groupby('operating_point').recall_at_10.mean()
    hn = max(hn, float((x - y.reindex(x.index)).abs().max()))
check('Paper text', 'HNSW recall@10 difference between the two runs, at most', '0.0065', f'{hn:.4f}')
sp = {}
for c in ('SIFT', 'GIST'):
    g = op[(op.corpus == c) & (op.method == 'BLISS')].pivot_table(index='operating_point', columns='seed', values='recall_at_10')
    sp[c] = float((g.max(axis=1) - g.min(axis=1)).max())
check('Paper text', 'BLISS seed spread of recall@10, SIFT and GIST', '0.039 / 0.027', f"{f3(sp['SIFT'])} / {f3(sp['GIST'])}")

# ------------------------------------------------------------------------------------------------ report
OUT.mkdir(exist_ok=True)
bad = sum(not ok for *_, ok in checks)


def report(verbose):
    width = max(len(i) for _, i, *_ in checks)
    for sec, item, p, got, ok in checks:
        if verbose or not ok:
            print(f"  {'ok  ' if ok else 'FAIL'} {sec:<12} {item:<{width}}  printed {p:<26} computed {got}")
    sections = list(dict.fromkeys(c[0] for c in checks))
    for sec in sections:
        n, k = sum(c[0] == sec for c in checks), sum(c[4] for c in checks if c[0] == sec)
        print(f"  {sec:<20} {k:>3} of {n:>3}{'' if k == n else '   <- differs'}")
    print(f'  {len(checks) - bad} of {len(checks)} values reproduced')


# ------------------------------------------------------------------------------------------------ figures
def figures():
    """The two plots in the README: recall@100 against recall@10, and the share of entries in the largest lists."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    style = {'HNSW': ('#005A9B', 'o'), 'IVF': ('#2B2B2B', 's'), 'IVFPQ+R': ('#8A8A8A', 'D'), 'BLISS': ('#E6320F', '^'), 'LIRA': ('#86AA00', 'v')}
    FIG = HERE / 'figures'
    FIG.mkdir(exist_ok=True)

    fig, axs = plt.subplots(1, 4, figsize=(11, 3.0), sharey=True)
    for ax, c in zip(axs, CORPORA):
        if shared[c]:
            ax.axvspan(*shared[c], color='#E3ECF5', zorder=0, label='range all five reach')
        ax.axvline(TARGET, color='.65', lw=.8, zorder=0)
        for m in METHODS:
            s_ = points[(points.corpus == c) & (points.method == m)].sort_values('recall_at_10')
            cl, mk = style[m]
            ax.plot(s_.recall_at_10, s_.recall_at_100, color=cl, marker=mk, ms=4, lw=1.4, label=m)
        v = [x['recall_at_100'] for x in at90[c].values()]
        ax.annotate(f'Δ{max(v) - min(v):.2f} at .90\n{len(at90[c])} of 5 cross .90', xy=(.03, .97), xycoords='axes fraction', va='top', fontsize=8)
        off = [m for m in METHODS if points[(points.corpus == c) & (points.method == m)].recall_at_10.max() < .7]
        for m in off:
            hi = points[(points.corpus == c) & (points.method == m)].recall_at_10.max()
            ax.annotate(f'{m} off scale (recall@10 ≤ {hi:.2f})', xy=(.97, .04), xycoords='axes fraction', ha='right', fontsize=8, color=style[m][0])
        ax.set(title=c + ('' if shared[c] else '  (no shared range)'), xlabel='recall@10', xlim=(.7, 1.0), ylim=(.5, 1.0))
    axs[0].set_ylabel('recall@100')
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=6, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, .08, 1, 1))
    fig.savefig(FIG / 'recall.png', dpi=200)

    lc = {'IVF': ('#2B2B2B', '-'), 'BLISS hash': ('#E6320F', '-'), 'BLISS k-means': ('#005A9B', '--'), 'LIRA': ('#86AA00', '-.')}
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.9), sharey=True)
    for ax, c in zip(axs, ('SIFT', 'GIST')):
        for name, (cl, ls) in lc.items():
            s_ = np.sort(layout(c, name).astype(float))[::-1]
            ax.plot(100 * np.arange(len(s_) + 1) / len(s_), 100 * np.concatenate([[0], np.cumsum(s_) / s_.sum()]), color=cl, ls=ls, lw=1.5, label=name)
        ax.plot([0, 100], [0, 100], color='.75', lw=.7, ls=':')
        ax.axvline(10, color='.75', lw=.7)
        ax.set(title=c, xlabel='lists, largest first (%)', xlim=(0, 100), ylim=(0, 101))
    axs[0].set_ylabel('stored entries (%)')
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, .1, 1, 1))
    fig.savefig(FIG / 'occupancy.png', dpi=200)
    print('figures: figures/recall.png, figures/occupancy.png')


# ================================================================================================ step 2: the corpora
SAMPLE_QUERIES = 100


def check_corpora(paths):
    """Facts read from each corpus file, and an exact top-10 search for a sample of the evaluated queries."""
    import h5py
    results = []
    for name, path in paths.items():
        row = corp.loc[name]
        with h5py.File(path, 'r') as f:
            train, test, gt, gt_d = f['train'], f['test'], f['neighbors'], f['distances']
            metric = f.attrs['distance']
            metric = metric.decode() if isinstance(metric, bytes) else str(metric)
            used = int(row.queries)
            results.append(('Corpus file', f'{name} vectors', f'{row.vectors:,}', f'{train.shape[0]:,}'))
            results.append(('Corpus file', f'{name} dimension', f'{row.dim}', f'{train.shape[1]}'))
            results.append(('Corpus file', f'{name} metric', {'cosine': 'angular', 'l2': 'euclidean'}[row.distance], metric))
            results.append(('Corpus file', f'{name} queries (the study evaluates the first ones)',
                            f'first {used:,} of {int(row.queries_in_file):,}', f'first {min(used, test.shape[0]):,} of {test.shape[0]:,}'))
            # exact search in float64 for queries the study evaluated; a tie with the 10th true neighbor counts as a match
            rng = np.random.default_rng(0)
            qi = np.sort(rng.choice(used, min(SAMPLE_QUERIES, used), replace=False))
            q = test[qi].astype(np.float64)
            if metric == 'angular':
                q /= np.linalg.norm(q, axis=1, keepdims=True)
            best = np.full((len(q), 10), np.inf)
            step = 100_000
            for start in range(0, train.shape[0], step):
                x = train[start:start + step].astype(np.float64)
                if metric == 'angular':
                    d = 1.0 - q @ (x / np.linalg.norm(x, axis=1, keepdims=True)).T
                else:
                    d = np.sqrt(np.maximum((q * q).sum(1)[:, None] - 2 * q @ x.T + (x * x).sum(1)[None, :], 0))
                best = np.sort(np.concatenate([best, np.partition(d, 9, axis=1)[:, :10]], axis=1), axis=1)[:, :10]
                print(f'\r    {name}: exact search over {min(start + step, train.shape[0]):,} of {train.shape[0]:,} vectors', end='', flush=True)
            print()
            # compare the 10 exact distances with the 10 distances shipped with the corpus
            ref = gt_d[qi, :10].astype(np.float64)
            match = np.mean(np.isclose(best, ref, rtol=1e-4, atol=1e-5))
            results.append(('Corpus file', f'{name} exact top-10 distances equal the shipped ground truth ({len(qi)} queries)', '1.000', f'{match:.3f}'))
    return results


def main():
    ap = argparse.ArgumentParser(description='Reproduce every number on the poster and in the paper from the measurements in data/.',
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog='examples:\n  python reproduce.py\n  python reproduce.py --download --figures\n  python reproduce.py -v')
    ap.add_argument('-v', '--verbose', action='store_true', help='print every value, not only mismatches and the summary')
    ap.add_argument('--figures', action='store_true', help='redraw figures/recall.png and figures/occupancy.png')
    ap.add_argument('--download', action='store_true', help='download missing corpora (8.7 GB) before checking them')
    ap.add_argument('--no-corpora', action='store_true', help='skip step 2 even if the corpora are present')
    ap.add_argument('--corpora-dir', type=Path, default=None, help='where the corpora are or should go (default: data/corpora)')
    args = ap.parse_args()

    print('Step 1  values from the measurements in data/')
    report(args.verbose)
    failed = bad

    print('\nStep 2  the corpora')
    if args.no_corpora:
        print('  skipped (--no-corpora)')
    else:
        import download
        directory = args.corpora_dir or download.DEFAULT_DIR
        present = {n for n in download.CORPORA if download.path_of(n, directory).exists()}
        if args.download or present == set(download.CORPORA):
            try:
                paths = download.ensure(directory=directory)
            except (ValueError, OSError) as e:
                print(f'  error: {e}')
                return 2
            rows = check_corpora(paths)
            for sec, item, p, got in rows:
                ok = p == got
                failed += not ok
                checks.append((sec, item, p, got, ok))
                if args.verbose or not ok:
                    print(f"  {'ok  ' if ok else 'FAIL'} {item:<70} expected {p:<12} found {got}")
            print(f'  {sum(p == g for *_, p, g in rows)} of {len(rows)} corpus checks passed')
        else:
            missing = ', '.join(n for n in download.CORPORA if n not in present)
            print(f'  not checked: {missing} not downloaded. Run with --download (8.7 GB) or `python download.py`.')

    if args.figures:
        print('\nStep 3  figures')
        figures()

    with open(OUT / 'values.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['section', 'value', 'printed', 'computed', 'match'])
        w.writerows(checks)
    print('\n' + ('All values match.' if not failed else f'{failed} value(s) differ; see out/values.csv.'))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
