#!/usr/bin/env python

import click
from collections import deque, Counter
from itertools import islice
from math import ceil
from statistics import median
from bisect import bisect_right
from platformdirs import PlatformDirs
import os
import sys
import pickle
import subprocess
import re

CACHE_PATH = PlatformDirs('ouiche').user_cache_path

_sourcedb = {}


class Source:
    def __init__(self, filename: str):
        if os.name == 'nt':
            if m := re.match('^/([a-z])/', filename):
                filename = m.group(1).upper() + ':' + filename[3:]
        self.filename = filename
        self.hash = None
        self.info = None


    def dump(self, start: int, count: int, output: str):

        skip = start
        flags_jump, flags_count = '', ''

        if count > 0:
            flags_count = f'-vframes {count}'

        # Optimisation: we can use our database of keyframes to seek to a safe timestamp
        # in the stream and only start decoding from that position. One caveat: if the
        # stream has a start_time value, we need to subtract it from the keyframe's PTS.
        if start > 0:
            from decimal import Decimal
            stream_start = self.get_metadata().get('start_time') or '0'
            keyframes = self.get_keyframes()
            i = bisect_right(keyframes, (start, chr(sys.maxunicode)))
            k = keyframes[max(0, i - 1)]
            flags_jump = f'-ss {Decimal(k[1]) - Decimal(stream_start):.6f}'
            skip -= k[0]

        p = subprocess.Popen([*re.split(' +', f'ffmpeg -v error {flags_jump} -i'),
                              self.filename,
                              *re.split(' +', f'-vf select=gte(n\\,{skip}),setpts=PTS-STARTPTS {flags_count}'),
                              *re.split(' +', f'-start_number {start} -flags2 +icc_profiles -strict unofficial -q:v 2'),
                              output], 
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             bufsize=1, text=True)
        p.wait()
        if p.returncode != 0:
            print('Error dumping images!')
            print(p.stderr)


    def get_metadata(self):
        info = self._get_info()
        if 'metadata' not in info:
            self._read_stream_metadata()
        return info['metadata']


    def get_keyframes(self):
        info = self._get_info()
        if 'keyframes' not in info:
            self._read_stream_keyframes()
        return info['keyframes']


    def _get_info(self):
        if self.hash is None:
            self.hash = self.quickhash(self.filename)
        if self.hash not in _sourcedb:
            cache_file = CACHE_PATH / self.hash
            try:
                with cache_file.open('rb') as f:
                    cache_data = pickle.load(f)
                print(f'# Found cached data in {cache_file}')
            except:
                cache_data = {}
            _sourcedb[self.hash] = cache_data
        return _sourcedb[self.hash]


    def _read_stream_metadata(self):
        ''' Get video stream information such as width, height, framerate, colorspace, etc. '''

        p = subprocess.Popen([*'ffprobe -v error -select_streams v:0 -show_streams'.split(' '), self.filename],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             bufsize=1, text=True)

        r_entry = re.compile(r'^([a-z_]+)=')
        metadata = {}
        for l in p.stdout:
            if m := r_entry.match(l):
                metadata[m.group(1)] = l.strip()[m.span()[1]:]
        p.wait()
        self._get_info()['metadata'] = metadata
        self._save_info()


    def _read_stream_keyframes(self):
        ''' Analyse video stream for keyframes to allow for very fast lookups afterwards '''

        p = subprocess.Popen([*'ffprobe -v error -select_streams v:0 -show_entries packet=pts_time,flags -of csv=p=0'.split(' '), self.filename],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             bufsize=1, text=True)

        # Output will be in the form:
        # 0.000000,K__
        # 0.334000,___
        # 0.167000,___
        # ...
        r_entry = re.compile(r'^([\d.]+),(.)')
        n, keyframes = 0, []
        for l in p.stdout:
            if m := r_entry.match(l):
                if m.group(2) == 'K':
                    keyframes.append((n, m.group(1)))
                n += 1
        p.wait()
        self._get_info()['keyframes'] = keyframes
        self._save_info()


    def _save_info(self):
        cache_file = CACHE_PATH / self.hash
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with cache_file.open('wb+') as f:
            pickle.dump(self._get_info(), f)

    @staticmethod
    def quickhash(file: str) -> str:
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
# Commands
#

@click.command()
@click.argument('files', nargs=-1)
def info(files: tuple[str, ...]):
    ''' Get information about movie files. '''
    for f in files:
        source = Source(f)
        print(f'# Metadata: {len(source.get_metadata())} entries')
        print(f'# Keyframes: {len(source.get_keyframes())} entries')


@click.command()
@click.option('--start', '--skip', default=0, help='Start dumping frames at this offset.')
@click.option('--count', default=-1, help='Number of frames to dump (-1 for maximum count).')
@click.argument('file', nargs=1)
@click.argument('output', nargs=1, default='fast%06d.jpeg')
def dump(file: str, output: str, start: int, count: int):
    source = Source(file)
    source.dump(start, count, output)
