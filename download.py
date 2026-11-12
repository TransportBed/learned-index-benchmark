"""Download the four ANN-Benchmarks corpora used in this study and verify them.

    python download.py                 # all four corpora into data/corpora/ (about 8.7 GB)
    python download.py sift glove      # only some of them
    python download.py --list          # show what is available and what is already here

Downloads resume where they stopped, and every file is checked against its size and SHA-256 before use.
reproduce.py --corpora calls the same functions, so you rarely need to run this file by hand.
"""
import argparse
import hashlib
import sys
import time
import urllib.request
from pathlib import Path

BASE_URL = 'https://ann-benchmarks.com'
DEFAULT_DIR = Path(__file__).resolve().parent / 'data' / 'corpora'

# name used in the study -> (file on ann-benchmarks.com, bytes, SHA-256 of the file the measurements were made on)
CORPORA = {
    'Deep10M': ('deep-image-96-angular.hdf5', 3_848_008_288, 'a0a44dfe80c58e63860862eeea2d34e62cff958dc3aec14fd65263d1d3c751f8'),
    'GloVe': ('glove-100-angular.hdf5', 485_413_888, '544af1d5e84e112cd4749571dcfd8ca109818a572f850af75a3a09e093a953c4'),
    'SIFT': ('sift-128-euclidean.hdf5', 525_128_288, 'dd6f0a6ed6b7ebb8934680f861a33ed01ff33991eaee4fd60914d854a0ca5984'),
    'GIST': ('gist-960-euclidean.hdf5', 3_844_648_288, '8e95831936bfdbfa0a56086942e2cf98cd703517c67f985914183eb4cdbf026a'),
}
ALIASES = {k.lower(): k for k in CORPORA} | {'deep': 'Deep10M', 'deep10m': 'Deep10M'}


def _human(n):
    for unit in ('B', 'KB', 'MB', 'GB'):
        if n < 1024 or unit == 'GB':
            return f'{n:.1f} {unit}' if unit != 'B' else f'{n} B'
        n /= 1024


def sha256(path, chunk=8 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def path_of(name, directory=DEFAULT_DIR):
    return Path(directory) / CORPORA[name][0]


def is_verified(name, directory=DEFAULT_DIR):
    """True if the file is complete and its checksum was verified before (the result is cached next to it)."""
    path = path_of(name, directory)
    stamp = path.with_suffix('.verified')
    return path.exists() and stamp.exists() and stamp.read_text().strip() == f'{path.stat().st_size} {CORPORA[name][2]}'


def verify(name, directory=DEFAULT_DIR, quiet=False):
    """Check size and SHA-256 of a downloaded corpus. Raises ValueError on a mismatch."""
    path = path_of(name, directory)
    _, size, digest = CORPORA[name]
    if is_verified(name, directory):
        return path
    if not path.exists():
        raise FileNotFoundError(path)
    if path.stat().st_size != size:
        raise ValueError(f'{path.name}: {path.stat().st_size:,} bytes, expected {size:,}')
    if not quiet:
        print(f'  checking {path.name} ...', end='', flush=True)
    got = sha256(path)
    if got != digest:
        raise ValueError(f'{path.name}: SHA-256 {got}, expected {digest}')
    path.with_suffix('.verified').write_text(f'{size} {digest}\n')
    if not quiet:
        print(' ok')
    return path


def download(name, directory=DEFAULT_DIR):
    """Download one corpus (resuming a partial file), verify it, and return its path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fname, size, _ = CORPORA[name]
    path = directory / fname
    if path.exists() and path.stat().st_size == size:
        return verify(name, directory)
    part = path.with_suffix('.part')
    done = part.stat().st_size if part.exists() else 0
    request = urllib.request.Request(f'{BASE_URL}/{fname}', headers={'User-Agent': 'learned-index-benchmark/1.0'})
    if done:
        request.add_header('Range', f'bytes={done}-')
    print(f'  {name}: downloading {fname} ({_human(size)})' + (f', resuming at {_human(done)}' if done else ''))
    start, last = time.time(), 0.0
    with urllib.request.urlopen(request, timeout=60) as response, open(part, 'ab' if done else 'wb') as out:
        if done and response.status != 206:  # server ignored the range request: start over
            out.truncate(0)
            done = 0
        while block := response.read(1 << 20):
            out.write(block)
            done += len(block)
            now = time.time()
            if now - last > 0.5 or done == size:
                rate = done / max(now - start, 1e-6)
                bar = '#' * int(30 * done / size)
                print(f'\r    [{bar:<30}] {100 * done / size:5.1f}%  {_human(done)} of {_human(size)}  {_human(rate)}/s',
                      end='', flush=True)
                last = now
    print()
    part.rename(path)
    return verify(name, directory)


def ensure(names=None, directory=DEFAULT_DIR):
    """Make sure the given corpora (default: all) are downloaded and verified. Returns {name: path}."""
    out = {}
    for name in names or CORPORA:
        out[name] = verify(name, directory) if path_of(name, directory).exists() and \
            path_of(name, directory).stat().st_size == CORPORA[name][1] else download(name, directory)
    return out


def main():
    ap = argparse.ArgumentParser(description='Download and verify the ANN-Benchmarks corpora used in this study.')
    ap.add_argument('corpora', nargs='*', help='deep10m, glove, sift, gist (default: all four)')
    ap.add_argument('--dir', type=Path, default=DEFAULT_DIR, help=f'where to store the files (default: {DEFAULT_DIR})')
    ap.add_argument('--list', action='store_true', help='list the corpora and their local state, then exit')
    args = ap.parse_args()
    if args.list:
        for name, (fname, size, _) in CORPORA.items():
            path = path_of(name, args.dir)
            state = 'verified' if is_verified(name, args.dir) else 'downloaded' if path.exists() else \
                'partial' if path.with_suffix('.part').exists() else 'missing'
            print(f'  {name:<8} {fname:<28} {_human(size):>9}  {state}')
        return 0
    try:
        names = [ALIASES[n.lower()] for n in args.corpora] or list(CORPORA)
    except KeyError as e:
        sys.exit(f'unknown corpus {e}; choose from: deep10m, glove, sift, gist')
    total = sum(CORPORA[n][1] for n in names if not path_of(n, args.dir).exists())
    if total:
        print(f'Downloading {_human(total)} into {args.dir}')
    try:
        ensure(names, args.dir)
    except (ValueError, OSError) as e:
        sys.exit(f'error: {e}')
    print('All requested corpora are present and verified.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
