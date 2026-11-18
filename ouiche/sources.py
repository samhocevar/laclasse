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


    def dump(self, start: int, count: int, output: str, quality: str):

        print(f'# Dumping movie frames to {output}... (quality: {quality})')

        def split(s): return re.split(' +', s.strip())

        input_options, filters, output_options = '', [], ''

        # Try to reduce ffmpeg stderr noise and amount of processed information
        input_options += ' -nostdin -nostats -an -sn'

        # Optimisation: we can use a database of keyframes to seek to a safe timestamp
        # in the stream and only start decoding from that position. One caveat: if the
        # stream has a start_time value, we need to subtract it from the keyframe's PTS.
        # XXX: after a lot of tinkering with this, I decided it was not worth the hassle,
        # because indexing a movie can take as much as 1 hour for huge Blu-rays. We might
        # get the occasional off-by-one error on some MKV files, but we can just
        # manually fix it, it's not the end of the world.
        #if start > 0:
        #    from decimal import Decimal
        #    stream_start = self.get_metadata().get('start_time') or '0'
        #    keyframes = self.get_keyframes()
        #    i = bisect_right(keyframes, (start, chr(sys.maxunicode)))
        #    k = keyframes[max(0, i - 1)]
        #    input_options += f' -ss {Decimal(k[1]) - Decimal(stream_start):.6f}'
        #    filters.append(f'select=gte(n\\,{start - k[0]}')

        # If an initial offset is required, use the pts of the first frame and the
        # stream framerate to decide exactly where to seek.
        if start > 0:
            sinfo = self.get_section('stream')
            finfo = self.get_section('frame')

            fps = sinfo.get('r_frame_rate')
            CAN_SEEK = True
            if CAN_SEEK and fps is not None:
                from decimal import Decimal
                delta = float(finfo.get('pts_time', 0))
                a, b = map(int, fps.split('/'))
                input_options += f' -ss {(1000 * start * b // a - 0.5) / 1000 + delta:.6f}'
            else:
                # This is very slow, as it requires decoding every frame, so only
                # keep it as a last resort.
                filters.append(f'select=gte(n\\,{start})')

        # Limit the number of frames. "-vframes {count}" would work, too, but the
        # decoder would have to decode an extra frame before it is triggered.
        if count > 0:
            filters.append(f'trim=end_frame={count}')

        # Ensure correct numbering of output files
        output_options += f' -start_number {start}'

        if len(filters):
            output_options += ' -vf ' + ','.join(filters)

        # It is important to lock the quality, otherwise identical frames may not
        # get the same checksum due to the dynamic quantisation table, making things
        # non-reproducible and harder to debug.
        if output.endswith('jpeg') or output.endswith('jpg'):
            # Quality options have really no impact on the mjpeg encoder performance
            # but the resulting file size is 1:3 between fast and best here.
            output_options += ' -c:v mjpeg'
            match quality:
                case 'fast': output_options += ' -global_quality 50 -q:v 15 -huffman default'
                case 'good': output_options += ' -global_quality 80 -q:v 5 -huffman default'
                case 'best': output_options += ' -global_quality 99 -q:v 2 -huffman optimal'

        # This is very important, it avoids dropping or duplicating frames when the
        # first pts is not what the demuxer expects.
        output_options += ' -fps_mode passthrough'

        # Other options of variable but minor importance
        output_options += ' -strict unofficial'
        output_options += ' -map_chapters -1'
        # Not sure about this yet, it makes decoding non-reproducible
        # TODO: investigate and report to FFmpeg upstream when I have the time
        #output_options += ' -flags2 +icc_profiles'

        p = subprocess.Popen([*split(f'ffmpeg {input_options} -i'), self.filename, *split(output_options), output], 
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             bufsize=1, text=True)
        print(' '.join(f'"{x}"' if ' ' in x else x for x in p.args))
        errors = []
        written = 0
        for l in p.stderr:
            if m := re.match('.*Parsed_showinfo.*pts:', l):
                print(l.strip()[39:190])
            elif m := re.match(rf'^frame=\s*(\d+)', l):
                written = int(m.group(1))
            else:
                errors.append(l.strip())
        p.wait()
        if p.returncode != 0:
            print('Error dumping images!')
            print('\n'.join(errors))
        else:
            print(f'# Wrote {written} frames!')


    def get_section(self, section):
        cache_data = self._ensure_cache()
        if section not in cache_data:
            match section:
                case 'stream': self._read_stream_metadata(cache_data)
                case 'frame':  self._read_stream_metadata(cache_data)
                #case 'keyframes': self._read_keyframes(cache_data)
            self._write_cache()
        return cache_data.get(section, {})


    #def get_keyframes(self, cache_data):
    #    return [] # FIXME
    #    if info.get('keyframes-version', 0) < KEYFRAMES_VERSION:
    #        print(f'Retrieving keyframe info for {self.filename}')
    #        frames = self._read_stream_keyframes()
    #        info['keyframes'] = frames
    #        info['keyframes-version'] = KEYFRAMES_VERSION
    #        self._write_cache()
    #
    #    return info['keyframes']


    def invalidate_cache(self):
        cache_data = self._ensure_cache()
        print(f'# Invalidating cache for {self.filename}')
        cache_data.clear()
        self._write_cache()


    def _ensure_cache(self):
        if self.hash is None:
            self.hash = self.quickhash(self.filename)
        cache_data = _sourcedb.get(self.hash)
        if cache_data is None:
            cache_file = CACHE_PATH / self.hash
            try:
                with cache_file.open('rb') as f:
                    cache_data = pickle.load(f)
                print(f'# Found cached data in {cache_file}')
            except:
                cache_data = {}
            _sourcedb[self.hash] = cache_data
        return cache_data


    def _read_stream_metadata(self, cache_data):
        ''' Get video stream information such as width, height, framerate, colorspace, etc. '''

        def super_escape(f):
            f = re.sub(r"(['\\])", r'\\\1', f)
            f = f"'{f}'"
            f = re.sub(r"(['\\])", r'\\\1', f)
            return f

        cmd = 'ffprobe -v error -select_streams v:0 -show_streams -show_frames -f lavfi -i'
        flt = f'movie={super_escape(self.filename)},trim=end_frame=1,showinfo'
        p = subprocess.Popen([*cmd.split(' '), flt],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             bufsize=1, text=True)

        r_entry = re.compile(r'^([a-z_]+)=')
        section = {}
        for l in p.stdout:
            if l.startswith('[/'):
                match l.strip():
                    case '[/FRAME]':  cache_data['frame'] = section
                    case '[/STREAM]': cache_data['stream'] = section
                section = {}
            elif m := r_entry.match(l):
                section[m.group(1)] = l.strip()[m.span()[1]:]

        p.wait()
        if p.returncode != 0:
            print('Error probing file!')
            if p.stderr:
                print(''.join(p.stderr))


    #def _read_stream_keyframes(self):
    #    ''' Analyse video stream for keyframes to allow for very fast lookups afterwards '''
    #
    #    from decimal import Decimal
    #
    #    p = subprocess.Popen([*'ffprobe -v error -select_streams v:0 -skip_frame discard -show_frames -of compact'.split(' '), self.filename],
    #                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    #                         bufsize=1, text=True)
    #
    #    # Output will be in the form:
    #    # 0.000000,K__
    #    # 0.334000,___
    #    # 0.167000,___
    #    # ...
    #    #r_entry = re.compile(r'^(N/A|[0-9.]+),(N/A|[0-9.]+),(.*)')
    #    #r_entry = re.compile(r'^([^,]+),(.)')
    #    r_entry = re.compile(r'^frame\|(.*)')
    #    n, frames = 0, []
    #    for l in p.stdout:
    #        if m := r_entry.match(l.strip()):
    #            key, pts = False, None
    #            for kv in m.group(1).split('|'):
    #                if kv == 'key_frame=1':
    #                    key = True
    #                if kv.startswith('best_effort_timestamp_time='):
    #                    pts = kv.split('=')[1]
    #            if key and pts is not None:
    #                frames.append((pts, n))
    #            n += 1
    #    p.wait()
    #    if p.returncode != 0:
    #        print('Error parsing file!')
    #        if p.stderr:
    #            print(''.join(p.stderr))
    #    print(f'Found {len(frames)} entries.')
    #
    #    return frames


    def _write_cache(self):
        cache_file = CACHE_PATH / self.hash
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with cache_file.open('wb+') as f:
            pickle.dump(self._ensure_cache(), f)

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

    #def get_letterbox_info(file):
    #    print(f'# Finding letterboxing for {file}...')
    #    last = None
    #    for l in ff_filter(file, "select='between(mod(t\\,600)\\,10\\,20)',cropdetect=limit=24:round=2",
    #                       pre_args="-ss 00:10:00 -skip_frame nokey".split()):
    #        if 'Parsed_cropdetect' in l:
    #            last = l
    #    if not last:
    #        print('# Error detecting letterboxing!')
    #        return
    #    crop = tuple(map(int, last.strip().split(' ')[-1].split('=')[1].split(':')))
    #    print(f'    letterboxing: {crop[0]}x{crop[1]}+{crop[2]}+{crop[3]} # {crop[0]/crop[1]:.2f}:1')

# -f null - 2>&1 | sed -ne 's/.*cropdetect.*\]//p' | sed 's/pts.* limit/limit/' | uniq -c
#$ ffmpeg -ss 00:10:00 -skip_frame nokey -i La.Classe.Américaine.1993.Betacam.To.DVD.mkv -vf select='between(mod(t\,600)\,10\,20)',cropdetect=limit=24:round=2 -f null - 2>&1 | sed -ne 's/.*cropdetect.*\]//p' | sed 's/pts.* limit/limit/' | uniq -c
    #ffmpeg -skip_frame nokey -i La.Classe.Américaine.1993.Betacam.To.DVD.mkv -^C select='between(mod(t\,300)\,100\,110)',cropdetect=limit=24:round=2 -f null -

#
# Commands
#

@click.command()
@click.argument('files', nargs=-1)
@click.option('--invalidate', default=False, is_flag=True, help='Invalidate cache before processing.')
def info(files: tuple[str, ...], invalidate: bool):
    ''' Get information about movie files. '''
    for f in files:
        source = Source(f)
        if invalidate:
            source.invalidate_cache()
        stream = source.get_section('stream')
        frame = source.get_section('frame')
        print(f'# Stream info: {' '.join(f'{k}: {v}' for k, v in stream.items())}')
        print(f'# Initial frame: {' '.join(f'{k}: {v}' for k, v in frame.items())}')


@click.command()
@click.option('--start', '--skip', default=0, help='Start dumping frames at this offset.')
@click.option('--count', default=-1, help='Number of frames to dump (-1 for maximum count).')
@click.option('--invalidate', default=False, is_flag=True, help='Invalidate cache before processing.')
@click.option('--quality', type=click.Choice(['fast', 'good', 'best']), default='good', is_flag=True, help='Set the image quality.')
@click.option('--fast', 'quality', flag_value='fast', help='Alias for --quality=fast')
@click.option('--good', 'quality', flag_value='good', help='Alias for --quality=good')
@click.option('--best', 'quality', flag_value='best', help='Alias for --quality=best')
@click.argument('file', nargs=1)
@click.argument('output', nargs=1, default='fast%06d.jpeg')
def dump(file: str, output: str, start: int, count: int, invalidate: bool, quality: str):
    source = Source(file)
    if invalidate:
        source.invalidate_cache()
    source.dump(start, count, output, quality)
