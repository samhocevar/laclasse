#!/usr/bin/env python

import click
from collections import deque, Counter
import getopt
from itertools import islice
from math import ceil
from statistics import median
import subprocess
import sys

#
# Utility functions
#

def prettify_dar(w, h):
    from fractions import Fraction
    n = Fraction(w, h)
    if n in (Fraction(4, 3), Fraction(16, 9)):
        return f'{n.numerator}:{n.denominator}'
    return f'{n:.2f}:1'

def ff_filter(file, filter, pre_args=[]):
    proc = subprocess.Popen(['ffmpeg', *pre_args, '-i', file, '-vf', filter, '-f', 'null', '-'],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             bufsize=1, text=True)
    for line in proc.stdout:
        yield line.strip()
    proc.poll()

def get_quickhash(file: str) -> str:
    from hashlib import md5
    from os import SEEK_END
    ret = md5()
    with open(file, 'rb') as f:
        # Hash 10 MiB of data
        data = f.read(10 * 1024 * 1024)
        ret.update(data)
        # Add the file size to the hash
        f.seek(0, SEEK_END)
        ret.update(str(f.tell()).encode())
    return ret.hexdigest()

#
# Fingerprinting
#

# Threshold for scene detection
CUTOFF = 10
# Sliding window width
NEIGHBOURHOOD = 10

def sliding_window(seq, n):
    it = iter(seq)
    win = tuple(islice(it, n))
    if len(win) == n:
        yield win
    for x in it:
        win = win[1:] + (x,)
        yield win

def get_cuts(file):
    cuts = []
    last_cut, frame = 0, 0
    for line in ff_filter(file, 'scdet,metadata=mode=print:key=lavfi.scd.score:file=-'):
        if line.startswith("frame:"):
            frame = int(line.split(':')[1].split(' ')[0])
        elif line.startswith("lavfi.scd.score="):
            score = float(line.split('=')[1].split(' ')[0])
            if score > CUTOFF:
                cuts.append(frame - last_cut)
                last_cut = frame
    if not cuts:
        return []
    #m = median(cuts)
    #cuts = list(x / m for x in cuts)
    windows = sliding_window(("{:.2f}".format(x) for x in cuts), NEIGHBOURHOOD)
    hashes = set('_'.join(s) for w in windows for s in w)
    return hashes 

known = {}

def fingerprint(file):
    print(f'Fingerprinting {file}...')
    cuts = get_cuts(file)
    if cuts:
        if known:
            print(f'  Similarities:')
        for k, v in known.items():
            counts = Counter(c in cuts for c in v)
            print(f'    {k}: yes={counts.get(True, 0)} no={counts.get(False, 0)}')
        known[file] = cuts
    else:
        print('  Error while computing fingerprint')

#
# Probing metadata
#

def get_file_info(file):
    print(f'# Getting information for {file}...')
    entries = 'stream=height,width,avg_frame_rate,sample_aspect_ratio,display_aspect_ratio,duration'
    proc = subprocess.Popen(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                             '-show_entries', entries, '-of', 'default=nw=1', file],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             bufsize=1, text=True)
    data = { t[0]: '='.join(t[1:]) for t in (l.strip().split('=') for l in proc.stdout) }
    proc.poll()

    if not data:
        print('# Error parsing file!')
        return

    def get(k):
        v = data.get(k)
        if not v or v == 'N/A':
            return None
        if ':' in v or '/' in v:
            return tuple(map(int, v.replace('/', ':').split(':')))
        if '.' in v:
            return float(v)
        return int(v)

    dur = get('duration')
    print(f'    duration: {dur}')
    fps = get('avg_frame_rate')
    print(f'    framerate: {fps[0]/fps[1]:.4f}')
    w, h = get('width'), get('height')
    print(f'    encoded_size: {w}x{h}')
    sar, dar = get('sample_aspect_ratio'), get('display_aspect_ratio')
    if not sar: sar = (1, 1)
    if not dar: dar = (w * sar[0], h * sar[1])
    sw, sh = int(ceil(w * sar[0] / sar[1])), h
    print(f'    display_size: {sw}x{sh} # {dar[0]/dar[1]:.2f}:1')

#
# Letterboxing detection
#

def get_letterbox_info(file):
    print(f'# Finding letterboxing for {file}...')
    last = None
    for l in ff_filter(file, "select='between(mod(t\\,600)\\,10\\,20)',cropdetect=limit=24:round=2",
                       pre_args="-ss 00:10:00 -skip_frame nokey".split()):
        if 'Parsed_cropdetect' in l:
            last = l
    if not last:
        print('# Error detecting letterboxing!')
        return
    crop = tuple(map(int, last.strip().split(' ')[-1].split('=')[1].split(':')))
    print(f'    letterboxing: {crop[0]}x{crop[1]}+{crop[2]}+{crop[3]} # {crop[0]/crop[1]:.2f}:1')

# -f null - 2>&1 | sed -ne 's/.*cropdetect.*\]//p' | sed 's/pts.* limit/limit/' | uniq -c
#$ ffmpeg -ss 00:10:00 -skip_frame nokey -i La.Classe.Américaine.1993.Betacam.To.DVD.mkv -vf select='between(mod(t\,600)\,10\,20)',cropdetect=limit=24:round=2 -f null - 2>&1 | sed -ne 's/.*cropdetect.*\]//p' | sed 's/pts.* limit/limit/' | uniq -c
    #ffmpeg -skip_frame nokey -i La.Classe.Américaine.1993.Betacam.To.DVD.mkv -^C select='between(mod(t\,300)\,100\,110)',cropdetect=limit=24:round=2 -f null -

#
# Main
#

@click.command()
@click.argument('files', nargs=-1)
def info(files: tuple[str, ...]):
    ''' Get information about movie files. '''
    for f in files:
        print(f'# Analysing {f}...')
        h = get_quickhash(f)
        print(f'  - hash: {h}')
        get_file_info(f)
        get_letterbox_info(f)
        #fingerprint(f)
        print()

