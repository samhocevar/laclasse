#!/usr/bin/env python

import click
import yaml
import glob
import cv2
import numpy as np
import re
import signal
import sys
from PIL import Image, ImageDraw, ImageFont, ImageText
from itertools import accumulate
from math import sin, cos, radians, sqrt, ceil
import concurrent.futures as futures
from pathlib import Path

# To generate a movie after this script has run:
'''
ffmpeg -safe 0 -f concat -i list.txt \
       -i ../../sources/dvd/la_classe*/*mkv \
       -map 0:v:0 -r 25 -c:v libx264 -pix_fmt yuv420p \
       -map 1:a:0 -c:a aac \
       output.mkv
'''

# At 24 fps with audio resampling and libx265 max quality:
'''
ffmpeg -safe 0 -f concat -r 24 -i list.txt \
       -i ../../sources/dvd/la_classe*/*mkv \
       -map 0:v:0 -c:v libx265 -b:v 8M -maxrate 10M -bufsize 16M \
       -map 1:a:0 -af "rubberband=tempo=24/25:pitch=1:formant=preserved:transients=crisp" -c:a aac \
       output.mkv
'''

# This is the size of the original movie (letterbox removed)
ORIGINAL_SIZE = (750, 358)

# Choose the aspect ratio
#TARGET_RATIO = 16 / 9
#TARGET_RATIO = 2.39
#TARGET_RATIO = 4 / 3
#TARGET_RATIO = 1.85
TARGET_RATIO = ORIGINAL_SIZE[0] / ORIGINAL_SIZE[1] # Original ~2.10

# Choose the mode here
mode = 'dcp4k'
match mode:
    case 'dcp4k':
        # Pillarboxed Scope 4K (better than Flat which would be letterboxed)
        MEDIA_SIZE = (4096, 1716)
        TARGET_SIZE = (int(MEDIA_SIZE[1] * TARGET_RATIO), MEDIA_SIZE[1])
    case 'dcp2k':
        # Pillarboxed Scope 2K (better than Flat which would be letterboxed)
        MEDIA_SIZE = (2048, 858)
        TARGET_SIZE = (int(MEDIA_SIZE[1] * TARGET_RATIO), MEDIA_SIZE[1])
    case 'bluray':
        # Letterboxed Blu-ray resolution
        MEDIA_SIZE = (1920, 1080)
        TARGET_SIZE = (MEDIA_SIZE[0], int(MEDIA_SIZE[0] / TARGET_RATIO))

# This is for the debug canvas
SCALED_SIZE = (1280, int(1280 / TARGET_RATIO))
THUMBNAIL_SIZE = (320, 240)
INFO_WIDTH = 600
MARGIN = 20

# 16:9 misses a lot of information, try 1.85:1
#TARGET_SIZE = (ORIGINAL_SIZE[0], int(ORIGINAL_SIZE[0] / 1.85))
#TARGET_SIZE = (ORIGINAL_SIZE[0], int(ORIGINAL_SIZE[0] / 1.77))
#TARGET_SIZE = ORIGINAL_SIZE
#TARGET_SIZE = (1280, 620)
#TARGET_SIZE = (1920, 1080) # 16:9 (1.77:1)
#TARGET_SIZE = (1920, 1038) # 1.85:1 (widescreen flat)
#TARGET_SIZE = (1920, 800) # 2.39:1 (scope)
#TARGET_SIZE = (1920, int(1920 * ORIGINAL_SIZE[1] / ORIGINAL_SIZE[0])) # original ~2.10 ratio

# The debug canvas; make sure dimensions are multiples of 4
DEBUG_SIZE = (
    (MARGIN + SCALED_SIZE[0] + MARGIN + INFO_WIDTH + MARGIN) // 4 * 4,
    (MARGIN + SCALED_SIZE[1] + MARGIN + ORIGINAL_SIZE[1] + MARGIN) // 4 * 4
)

DESC = Path.home() / 'laclasse/data/laclasse.yaml'
DATA_ROOT = Path.home() / 'laclasse/data'
DUMP_ROOT = Path.home() / 'laclasse/dump-dvd'

SEARCH_ROOT = (
    'D:/clips/jpeg-best',
    'D:/clips/png-good',
    'D:/clips/jpeg-good',
    'D:/clips/jpeg-fast',
    #'D:/clips/new',
    #'D:/clips/good',
)

DEBUG_ROOT = 'D:/final/debug'
OUTPUT_ROOT = f'D:/final/{mode}'
DCP_ROOT = 'D:/final/dcp'

# Some convenient constants
FONT = cv2.FONT_HERSHEY_SIMPLEX
LARGE = THUMBNAIL_SIZE[1] / 240
SMALL = THUMBNAIL_SIZE[1] / 320
NEWLINE = int(THUMBNAIL_SIZE[1] / 8)

WHITE = (255, 255, 255)
GRAY = (127, 127, 127)
RED = (127, 127, 255)
PURPLE = (255, 0, 255)
GREEN = (127, 255, 127)
YELLOW = (127, 255, 255)

INTRO_COLOR = (251, 238, 98)
INTRO_SHADOW = (34, 28, 6)

def cv2_dither_matrix(width, height):
    bayer = np.array([[ 0, 32,  8, 40,  2, 34, 10, 42],
                      [48, 16, 56, 24, 50, 18, 58, 26],
                      [12, 44,  4, 36, 14, 46,  6, 38],
                      [60, 28, 52, 20, 62, 30, 54, 22],
                      [ 3, 35, 11, 43,  1, 33,  9, 41],
                      [51, 19, 59, 27, 49, 17, 57, 25],
                      [15, 47,  7, 39, 13, 45,  5, 37],
                      [63, 31, 55, 23, 61, 29, 53, 21]]).astype(np.float32) / 64.0
    return np.tile(bayer, (int(ceil(height / bayer.shape[0])),
                           int(ceil(width / bayer.shape[1]))))[:height,:width]


def safe_pooled_map(num, fn, it):
    with futures.ProcessPoolExecutor(max_workers=num) as e:
        for f in futures.as_completed([e.submit(fn, arg) for arg in it]):
            if f.done():
                yield f.result()
            else:
                raise f.exception()


def cv2_blit(bg, fg, x, y, opacity=1.0):
    h, w, _ = bg.shape
    sh, sw, _ = fg.shape

    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + sw, w), min(y + sh, h)
    if x1 >= x2 or y1 >= y2:
        return

    sx1, sy1 = x1 - x, y1 - y
    sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)

    dst = bg[y1:y2, x1:x2]
    src = fg[sy1:sy2, sx1:sx2]

    if src.shape[2] == 4:
        dst += src[...,3:4] / 255.0 * (src[:,:,:3] - dst)
    else:
        dst += src if opacity == 1.0 else (src * opacity)


def cv2_lift_gain_gamma(im, lift, gain, gamma):
    ret = im.copy()
    if lift is not None:
        ret[...,:3] += np.array(lift[::-1]) * (255.0 - ret[...,:3])

    if gain is not None:
        ret[...,:3] *= np.array(gain[::-1])

    if gamma is not None:
        tmp = np.clip(ret[...,:3], 0.0, None) / 255.0
        ret[...,:3] = 255.0 * np.power(tmp, 1.0 / np.maximum(np.array(gamma[::-1]), 1e-5))

    return np.clip(ret, 0.0, 255.0)


def cv2_tint_midtones(im, colorshift):
    # Compute luma in the 0..1 range
    gray = np.array([0.0722, 0.7152, 0.2126]) / 255.0
    luma = im[...,:3] @ gray

    # Only work in the midtones
    mask = np.exp(-(((luma - 0.5) / 0.35) ** 4))

    ret = im.copy()
    ret[...,:3] += mask[...,np.newaxis] * colorshift[::-1]
    return np.clip(ret[...,:3], 0.0, 255.0)


def cv2_tint_dark(im, color):
    # Compute darkness weight
    gray = np.array([0.0722, 0.7152, 0.2126]) / 255.0
    luma = im[...,:3] @ gray

    w = (1.0 - luma) ** 2.0
    w = w[..., None]
    # Return blend but don't touch alpha if it is there)
    ret = im.copy()
    ret[...,:3] *= (1.0 - w)
    ret[...,:3] += np.array(color) * w
    return ret


def cv2_emboss(im, shift, strength):
    ''' Add highlights and shadows in an image along the alpha channel border. '''
    m1 = np.array([[1.0, 0.0, shift],
                   [0.0, 1.0, shift]])
    emboss = cv2.warpAffine(im[...,3], m1, (im.shape[1], im.shape[0]), flags=cv2.INTER_LINEAR)
    m2 = np.array([[1.0, 0.0, -shift],
                   [0.0, 1.0, -shift]])
    shadow = cv2.warpAffine(im[...,3], m2, (im.shape[1], im.shape[0]), flags=cv2.INTER_LINEAR)
    a = np.maximum(0, im[...,3] - emboss)
    b = np.maximum(0, im[...,3] - shadow)
    bevel = (a - b) * im[...,3] * strength

    ret = im.copy()
    #ret[...,:3] = bevel[...,np.newaxis]
    #return ret
    #ret[...,:3] = 127.0
    ret[...,:3] = np.clip(ret[...,:3] + bevel[...,np.newaxis], 0, 255)
    return ret


def parse_playback(playback):
    sequence = []
    for s in playback.split(','):
        l = [x.strip() for x in s.split('-')]
        match len(l):
            case 1: sequence.append(int(l[0], 10))
            case 2:
                a, b = int(l[0], 10), int(l[1], 10)
                sequence.extend(range(a, b + 1) if a <= b else range(a, b - 1, -1))
            case _: return None
    return sequence


def render_text(text: str, font: str, size: float, border: int=0):
    # FIXME: this is ridiculous, PIL doesn't do antialiasing?!
    SCALE = 4
    image_font = ImageFont.truetype(DATA_ROOT / font, size * SCALE)
    image_text = ImageText.Text(text, image_font)

    left, top, right, bottom = image_text.get_bbox()
    anchor_x, anchor_y = image_text.get_bbox(anchor='ls')[:2]

    im = Image.new('F', (right - left + 2 * border * SCALE, bottom - top + 2 * border * SCALE))
    draw = ImageDraw.Draw(im)
    draw.text((border * SCALE - left, border * SCALE - top), image_text, 1.0)
    im = im.resize((im.width // SCALE, im.height // SCALE), Image.LANCZOS)

    # Use rendered text image as an alpha layer for a white image
    return (np.dstack((np.full((im.height, im.width, 3), 255.0),
                       np.array(im)[:,:,np.newaxis])),
            (anchor_x // SCALE - border, anchor_y // SCALE - border))


class RenderConfig:
    def __init__(self, force, verbose, dry_run):
        self.force = force
        self.verbose = verbose
        self.dry_run = dry_run


class Renderer:
    def __init__(self, config):
        with open(DESC, encoding='utf-8') as f:
            data = yaml.safe_load(f)

        self.config = config
        self.presets = data['presets']
        self.segments = data['segments']
        self.movies = data['movies']


class Segment:
    def __init__(self, config, renderer, start, desc):
        self.config = config
        self.presets = renderer.presets
        self.start = start
        self.desc = desc
        self.id = desc.get('id', 'unknown')
        self.count = desc.get('count', 0)
        self.clips = desc.get('clips', [])

        self.output_dir = Path(f'{OUTPUT_ROOT}/{self.start:06d}')
        self.debug_dir = Path(f'{DEBUG_ROOT}/{self.start:06d}')
        self._orig_files = None # Lazily loaded

    def has_movie(self, movie):
        return any(m in c.get('movie', '') for c in self.clips for m in movie.split(','))

    def has_property(self, prop):
        return any(prop in c for c in self.clips)

    def orig_file(self, n):
        if not self._orig_files:
            # Find original image
            dumpdir = glob.glob(f'{DUMP_ROOT}/{self.start:06d}-*')
            if not dumpdir:
                raise f'{self.start:06d}: cannot find source directory in {DUMP_ROOT}'
            self._orig_files = glob.glob(f'{dumpdir[0]}/*')
            if len(self._orig_files) != self.count:
                raise BaseException(f'{self.start:06d}: need {self.count} frames but dump directory has {len(dumpfiles)}')
        return self._orig_files[n]


    def render_all(self):

        print(f'{self.start:06d}: ', end='')
        sys.stdout.flush()

        if not self.config.dry_run:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.debug_dir.mkdir(parents=True, exist_ok=True)

        # XXX: Ensure the list of original files is cached, otherwise each
        # pool worker will do it again and again. Not very elegant.
        self.orig_file(0)

        for n, ok in enumerate(safe_pooled_map(8, self.render_frame, range(self.count))):
            if not self.config.verbose:
                print('.' if not n or n % 10 else str(n), end='')
                sys.stdout.flush()


    def render_frame(self, n: int):

        f = Frame(self.config, self, n)

        # Skip existing files
        if not self.config.force and f.is_uptodate():
            return

        f.render()

        if not self.config.dry_run:
            f.save()

        return True


class Frame:
    def __init__(self, config, segment, n):
        self.n = n
        self.config = config
        self.segment = segment
        self.output_file = segment.output_dir / f'img{segment.start+n:06d}.jpeg'
        self.debug_file = segment.debug_dir / f'img{segment.start+n:06d}.jpeg'
        self.debug_x = MARGIN + SCALED_SIZE[0] + MARGIN
        self.debug_y = MARGIN


    def debug_text(self, text, size=SMALL, color=GRAY, dx=0, dy=0, newline=True):
        cv2.putText(self.debug_image, text, (self.debug_x + dx, self.debug_y + dy + 25), FONT, size, color, 3)
        if newline:
            self.debug_y += NEWLINE


    def is_uptodate(self):
        return self.output_file.is_file() and self.debug_file.is_file()


    def render(self):
        self.debug_image = np.zeros((DEBUG_SIZE[1], DEBUG_SIZE[0], 3), np.uint8)

        # Blit original dumped image
        # XXX: imread does not handle UTF-8 paths!
        #src = cv2.imread(self.segment.orig_file(self.n))
        src = cv2.imdecode(np.fromfile(self.segment.orig_file(self.n), dtype=np.uint8), cv2.IMREAD_COLOR)
        h, w, _ = src.shape
        x, y = MARGIN, MARGIN + SCALED_SIZE[1] + MARGIN
        x += max(0, SCALED_SIZE[0] - ORIGINAL_SIZE[0]) // 2 # Center image for aesthetic purposes
        cv2_blit(self.debug_image, src, x, y)
        cv2.rectangle(self.debug_image, (x-1, y-1), (x+w, y+h), YELLOW, 1)
        cv2.putText(self.debug_image, f'Original: {ORIGINAL_SIZE[0]}x{ORIGINAL_SIZE[1]} ({ORIGINAL_SIZE[0]/ORIGINAL_SIZE[1]:.2f}:1)',
                    (x + ORIGINAL_SIZE[0] + MARGIN, y + ORIGINAL_SIZE[1]), FONT, SMALL, GRAY, 3)

        # Text information
        self.debug_text(f'S.{self.segment.start:06d} ({self.segment.count})', size=LARGE, color=WHITE, newline=False)
        self.debug_text(f'F.{self.segment.start+self.n:06d} (+{self.n})', dx=300, size=LARGE, color=GREEN)

        self.debug_text(f'Target: {TARGET_SIZE[0]}x{TARGET_SIZE[1]} ({TARGET_SIZE[0]/TARGET_SIZE[1]:.2f}:1)')
        if scenes := self.segment.desc.get('scenes'):
            self.debug_text(f'Scenes: {" ".join(scenes)}')

        errors = []
        clips = []

        # Enumerate clips
        for i, c in enumerate(self.segment.clips):

            # Compute opacity early on; if it is zero, we can bail out quickly
            opacity = 1.0
            if l := c.get('fade-in'):
                opacity = min(opacity, max(0.0, (self.n - l[0]) / (l[1] - l[0])))
            if l := c.get('fade-out'):
                opacity = min(opacity, max(0.0, (l[1] - self.n) / (l[1] - l[0])))
            if opacity <= 0.0:
                continue

            transforms = tuple(filter(None, re.split('\\s+', c.get('transform', ''))))

            # If it is a text clip, handle it here
            if text := c.get('text'):
                font, size, pos, frames = c.get('font'), c.get('size'), c.get('pos'), c.get('frames')
                if frames is not None and (self.n < frames[0] or self.n > frames[1]):
                    continue

                self.debug_text(f'Clip {i}: "{text}"', color=WHITE)

                # XXX: adjust size in case the aspect ratio is narrower than the original!
                if (λ := (TARGET_SIZE[0] / TARGET_SIZE[1]) / (ORIGINAL_SIZE[0] / ORIGINAL_SIZE[1])) < 1.0:
                    size *= λ

                border = 5
                data, anchor = render_text(text, font, size * TARGET_SIZE[1] / 100, border=border)
                tx, ty = pos[0] * TARGET_SIZE[0] / 100, pos[1] * TARGET_SIZE[1] / 100

                # XXX: debug baseline
                #tmp_image = np.zeros((TARGET_SIZE[1], TARGET_SIZE[0], 4), float)
                #cv2.line(tmp_image, (int(tx), 0), (int(tx), TARGET_SIZE[1]), (*RED, 255.0), 1)
                #cv2.line(tmp_image, (0, int(ty)), (TARGET_SIZE[0], int(ty)), (*RED, 255.0), 1)
                #clips.append((tmp_image, 0, 0, 1.0))

                # We do not use the anchor when padding to the right, so we need to manually add
                # the border value. Not very elegant.
                if tx == 0:   x = (TARGET_SIZE[0] - data.shape[1]) // 2
                elif tx >= 0: x = int(tx + anchor[0])
                else:         x = int(TARGET_SIZE[0] - data.shape[1] + tx - 1 + border)

                y = anchor[1] + int(ty if ty >= 0 else TARGET_SIZE[1] - data.shape[0] + ty - 1)

                # Pre-apply opacity because opacity is actually some kind of alpha blend
                # FIXME: we need to merge everything _before_ adding a single clip to the
                # list, otherwise opacity will accumulate and will be wrong
                data[:,:,3] *= opacity

                # Add a little bit of blur and print a shadow
                data = cv2.blur(data, (3, 3))
                shadow = data.copy()
                shadow[...,:3] = INTRO_SHADOW[::-1]
                shadow[...,:3] += np.random.normal(0, 6, data[...,:3].shape).astype(float)
                step = 0.4 * TARGET_SIZE[1] / 100
                for k in range(1, 5):
                    clips.append((shadow, int(x + step * k), int(y + step * k), 1.0))

                #data[...,3] = cv2.blur(data[...,3], (5, 5))

                # XXX: just a little test, like embossing
                #emboss = data.copy()
                #emboss[...,3] = cv2.blur(emboss[...,3], (3, 3))
                #emboss[...,:3] = np.array(INTRO_COLOR[::-1]) * 0.25
                #clips.append((emboss.copy(), int(x + step / 2), int(y + step / 2), 1.0))
                #emboss[...,:3] = 255.0 - (255.0 - np.array(INTRO_COLOR[::-1])) * 0.25
                #clips.append((emboss.copy(), int(x - step / 2), int(y - step / 2), 1.0))

                data[...,:3] = INTRO_COLOR[::-1]
                data = cv2_emboss(data, step * 0.1, 500.0)

                data[...,:3] += np.random.normal(0, 10, data[...,:3].shape).astype(float)
                data[...,:3] = cv2.blur(data[...,:3], (3, 3))
                data[...,:3] += np.random.normal(0, 6, data[...,:3].shape).astype(float)
                #data = cv2.blur(data, (3, 3))

                clips.append((data, x, y, 1.0))
                continue

            # Otherwise it must be a movie clip
            if clip_movie := c.get('movie'):
                clip_segment, clip_offset = c.get('segment'), c.get('offset')
            else:
                errors.append(f'no movie associated with clip {i}')
                continue

            clip_presets = self.segment.presets.get(clip_movie) or {}

            self.debug_text(f'Clip {i}: {clip_movie}', color=WHITE)

            clipinfo = []
            fxinfo = []
            todo = []

            if clip_segment is None or clip_offset is None:
                self.debug_text(f'Todo: missing segment or offset information', dx=20, color=RED)
                continue

            # Compute source offset
            current_offset = self.n
            sequence = None
            match playback := c.get('playback'):
                case None: pass
                case 'reverse': current_offset = self.segment.count - 1 - self.n
                case _:
                    seq = parse_playback(playback)
                    if seq is None:
                        errors.append(f'cannot parse playback {playback}')
                    elif len(seq) != self.segment.count:
                        errors.append(f'playback length {len(seq)}, expected {self.segment.count}')
                    else:
                        current_offset = seq[self.n]
            if playback is not None:
                fxinfo.append(f'Playback: {re.sub(r'\s+', ' ', playback)}')

            src_offset = clip_offset + current_offset

            clipinfo.append(f'S.{clip_segment} F.{src_offset}') # use computed values from clip if available

            # Find and load image
            filename = None
            src = None
            srcglob = f'{clip_movie}/{clip_segment}/*{int(clip_segment, 10)+src_offset:06d}.*'
            for rootdir in SEARCH_ROOT:
                m = glob.glob(f'{rootdir}/{srcglob}')
                if m and Path(m[0]).is_file():
                    filename = m[0]
                    src = cv2.imread(filename).astype(float)
                    break

            if src is None:
                errors.append(f'cannot find image for {srcglob} in {', '.join(SEARCH_ROOT)}')
                continue

            if 'woodpanel' in transforms:
                stencil = cv2.imread(DATA_ROOT / 'woodstencil.png', cv2.IMREAD_UNCHANGED).astype(float)
                src = np.dstack((src, 255.0 - stencil[...,3:]))

            # Apply source letterbox cropping if the value is known to us
            if crop := clip_presets.get(f'letterbox-{src.shape[0]}'):
                if crop > 0:
                    src = src[crop:-crop,:]
            if crop := clip_presets.get(f'letterbox-{src.shape[1]}'):
                if crop > 0:
                    src = src[:,crop:-crop]

            # Apply horizontal flip early, it's easier to think with the flipped image
            if 'hflip' in transforms:
                src = cv2.flip(src, 1)

            # encoded_h, encoded_w: the size of the source image on disk, _after_ letterboxing
            # display_h, display_w: the original display size of the source image, after anamorphic stretching
            encoded_h, encoded_w, _ = src.shape
            display_h, display_w = encoded_h, encoded_w

            if self.config.verbose:
                print('  Encoded size:', encoded_w, encoded_h)

            # XXX: apply stretch to DVD images, this is for debugging purposes
            if 'revolution' in clip_movie and display_w == 720:
                display_w = 875
            if 'texas' in clip_movie and display_w == 720:
                display_w = 853
            if 'firecreek' in clip_movie and display_w == 1920:
                display_w = 2543
            if 'arrangement' in clip_movie and display_w == 1280:
                todo.append(f'Must unscale to 960')

            if self.config.verbose:
                print('  Display size:', display_w, display_h)

            clipinfo.append(f'DAR: {display_w/display_h:.2f}:1')

            # The authorised bounds where we can pick our target image
            bounds_x, bounds_y, bounds_w, bounds_h = 0, 0, encoded_w, encoded_h

            # Apply cropping by simply reducing the bounds
            if crop := c.get('crop'):
                values = tuple(map(float, crop.strip().split(' ')))
                if len(values) == 4:
                    left = int(values[0] * bounds_w / 100)
                    right = int(values[1] * bounds_w / 100)
                    top = int(values[2] * bounds_h / 100)
                    bottom = int(values[3] * bounds_h / 100)
                    bounds_x, bounds_w = bounds_x + left, bounds_w - left - right
                    bounds_y, bounds_h = bounds_y + top, bounds_h - top - bottom
                elif len(values) in (1, 2):
                    dx = int(values[0] * bounds_w / 100)
                    dy = int(values[-1] * bounds_h / 100)
                    bounds_x, bounds_w = bounds_x + dx, bounds_w - 2 * dx
                    bounds_y, bounds_h = bounds_y + dy, bounds_h - 2 * dy
                else:
                    errors.append(f'invalid crop {crop}')
                fxinfo.append(f'Crop: {crop}')

            if self.config.verbose:
                print('  Bounds:', bounds_x, bounds_y, bounds_w, bounds_h)

            # Compute pick rectangle aspect ratio (might be different from target aspect
            # ratio because we apply stretching, both from anamorphic encoding and from
            # the clip properties)
            rect_ratio = TARGET_SIZE[0] / TARGET_SIZE[1]
            rect_ratio *= encoded_w / display_w
            match stretch := c.get('stretch') or clip_presets.get('stretch'):
                case None: pass
                case _: rect_ratio /= stretch
            if stretch is not None:
                fxinfo.append(f'Stretch: {stretch}')

            if self.config.verbose:
                print('  Pick ratio:', rect_ratio)

            # rect_x, rect_y, rect_w, rect_h: the coordinates where we will actually
            # pick our pixels, preserving desired aspect ratio
            if bounds_w / bounds_h > rect_ratio:
                rect_w, rect_h = int(bounds_h * rect_ratio), bounds_h
            else:
                rect_w, rect_h = bounds_w, int(bounds_w / rect_ratio)
            rect_x, rect_y = bounds_x + (bounds_w - rect_w) // 2, bounds_y + (bounds_h - rect_h) // 2

            if self.config.verbose:
                print('  Initial pick area', rect_x, rect_y, rect_w, rect_h)

            # Apply gravity (when we want to pan to a specific direction)
            match gravity := c.get('gravity') or clip_presets.get('gravity'):
                case None: pass
                case _:
                    for g in gravity.split(' '):
                        match g[0]:
                            case 'r': rect_x += (bounds_w - rect_w) * int(g[1:]) // 200
                            case 'l': rect_x -= (bounds_w - rect_w) * int(g[1:]) // 200
                            case 'd': rect_y += (bounds_h - rect_h) * int(g[1:]) // 200
                            case 'u': rect_y -= (bounds_h - rect_h) * int(g[1:]) // 200
                            case _: errors.append(f'invalid gravity {gravity}')
            if gravity is not None:
                fxinfo.append(f'Gravity: {gravity}')

            # Some parameters cause a zoom effect
            zoom = 1.0

            # Load wave parameters before scaling so that we know how much to zoom in
            wave = 0
            if x := c.get('wave-in'):
                wave = max(wave, x - self.n)
            if x := c.get('wave-out'):
                wave = min(wave, x - self.n)
            if wave != 0:
                # Zoom slightly
                zoom *= 1 + abs(wave) / 400

            # If rotating, we need to scale a bit more!
            rotate = c.get('rotate', 0.0)
            if rotate != 0:
                zoom *= cos(radians(rotate)) + abs(sin(radians(rotate))) * rect_ratio

            # Update pick area according to zoom
            if zoom != 1.0:
                rect_x, rect_w = rect_x + int(rect_w * (1 - 1 / zoom) / 2), int(rect_w / zoom)
                rect_y, rect_h = rect_y + int(rect_h * (1 - 1 / zoom) / 2), int(rect_h / zoom)

            if self.config.verbose:
                print('  Final pick area:', rect_x, rect_y, rect_w, rect_h)

            # Compute the final parameters for scaling. After this step, our frame lies somewhere
            # in there as a rectangle of size TARGET_SIZE, but we can apply more effects.
            scale_x, scale_y = TARGET_SIZE[0] / rect_w, TARGET_SIZE[1] / rect_h

            if self.config.verbose:
                print('  Scale factor:', scale_x, scale_y)

            final_x = int(rect_x * scale_x)
            final_y = int(rect_y * scale_y)
            final_w = TARGET_SIZE[0]
            final_h = TARGET_SIZE[1]
            blit_x = 0
            blit_y = 0

            # Tint effect
            if tint := c.get('tint'):
                src = cv2_tint_midtones(src, tint)
                fxinfo.append(f'Tint: {tint}')

            # Lift/gain/gamma
            lift, gain, gamma = c.get('lift'), c.get('gain'), c.get('gamma')
            if lift is not None or gain is not None or gamma is not None:
                src = cv2_lift_gain_gamma(src, lift, gain, gamma)
                if lift  is not None: fxinfo.append(f'Lift: {lift}')
                if gain  is not None: fxinfo.append(f'Gain: {gain}')
                if gamma is not None: fxinfo.append(f'Gamma: {gamma}')

            # Effet spéciau minable 1
            if 'cheesy-1' in transforms:
                bar_left, bar_right = 0.495, 0.515
                # Transition happens between frames 28 and 50
                if self.n > 28:
                    dt = min(1.0, (self.n - 28) / (50 - 28))
                    dt = 1.0 - pow(1.0 - dt, 2) # smooth transition
                    bar_x = int(final_w * (1 - (1 - bar_left) * dt))
                    bar_w = int(final_w * (bar_right - bar_left))
                    if i == 0:
                        final_x += int((final_w - bar_x) * 0.37)
                        final_w = bar_x
                    if i == 1:
                        blit_x = int(TARGET_SIZE[0] * (1 - (1 - bar_left) * dt + bar_right - bar_left))
                        final_w = final_w // 2
                        final_x += int(final_w * 0.85)

            # Effet spéciau minable 2
            if 'cheesy-2' in transforms:
                tint = (48.0, 0.0, 0.0)
                #lift = None #(0.1, 0.0, 0.0)
                #gain = None #(1.0, 1.0, 1.0)
                #gamma = (1.0, 0.9, 0.9)
                bar_left, bar_right = 0.530, 0.545
                # Need to preserve the left part of the screen
                if self.n > 197:
                    saved = src[:,rect_x:rect_x+int(rect_w*bar_left)]
                    saved = cv2_tint_dark(saved, tint[::-1])
                    #saved = cv2_lift_gain_gamma(saved, lift, gain, gamma)
                if self.n > 136:
                    # FIXME: the black bar keeps readjusting until frame 194!
                    # (the position is non-linear, it follows the scene background)
                    # But this will do for now.
                    dt = min(1.0, (self.n - 136) / (171 - 136))
                    src[:,:int(rect_x+rect_w*dt*bar_right)] = (0, 0, 0)
                if self.n > 197:
                    dt = min(1.0, (self.n - 197) / (222 - 197))
                    dt = 1.0 - pow(1.0 - dt, 2) # smooth transition
                    cv2_blit(src, saved, rect_x, int((rect_y-display_h)*(1-dt)))

            # Effet spéciau minable 3
            if 'cheesy-3' in transforms:
                tint = (64.0, 64.0, 0.0)
                bar_left, bar_right = 0.400, 0.415
                # Transition happens between 52 and 77
                if self.n < 77:
                    dt = min(1.0, (77 - self.n) / (77 - 52))
                    bar_w = int(rect_w*(bar_right-bar_left))
                    bar_x = rect_x + int(rect_w*bar_right*dt) - bar_w
                    saved = src[:,rect_x:bar_x]
                    #saved[...] = cv2_tint_midtones(saved, (100, 80, 0))
                    saved[...] = cv2_tint_dark(saved, tint[::-1])
                    saved[...] = cv2_lift_gain_gamma(saved, None, None, (0.9, 0.8, 0.7))
                    src[:,max(0,bar_x):max(0,bar_x+bar_w)] = (0, 0, 0)

            # Other transforms
            for t in transforms:
                match t:
                    case 'hflip': pass
                    case 'cheesy-1': pass
                    case 'cheesy-2': pass
                    case 'cheesy-3': pass
                    case 'woodpanel': pass
                    case _: errors.append(f'invalid transform "{t}"')
                fxinfo.append(f'Transform: {t}')

            # Apply scaling to the source image, it's now time to get the good pixels
            if scale_x > 10 or scale_y > 10:
                raise Exception('insane scaling, aborting')
            target_w, target_h = int(src.shape[1] * scale_x), int(src.shape[0] * scale_y)
            if rotate != 0.0:
                m_rot = cv2.getRotationMatrix2D((src.shape[1] / 2, src.shape[0] / 2), -rotate, 1.0)
                # FIXME: we could do this in one go!
                src = cv2.warpAffine(src, m_rot, (src.shape[1], src.shape[0]))
                src = cv2.resize(src, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            elif scale_x != 1.0 or scale_y != 1.0:
                src = cv2.resize(src, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

            if self.config.verbose:
                print('  Scaled pick area:', final_x, final_y, final_w, final_h)

            if opacity != 1.0:
                fxinfo.append(f'Opacity: {opacity:.2f}')

            # "Quality" is the density of source pixels in the target image, a rough
            # estimate of the number of pixels lost through scaling in both dimensions.
            quality = min(1, encoded_w / src.shape[1]) * min(1, encoded_h / src.shape[0])
            clipinfo.append(f'Quality: {100 * quality:.1f}%')

            # Add debug info etc.
            xthumb, ythumb = self.debug_x, self.debug_y + 10
            thumbscale = min(THUMBNAIL_SIZE[0] / display_w, THUMBNAIL_SIZE[1] / display_h)
            thumb = cv2.resize(src, (int(display_w * thumbscale), int(display_h * thumbscale)), interpolation=cv2.INTER_LANCZOS4)
            #if transform == 'hflip': thumb = cv2.flip(thumb, 1)
            cv2_blit(self.debug_image, thumb[...,:3].clip(0, 255).astype(np.uint8), xthumb, ythumb)
            cv2.rectangle(self.debug_image, (xthumb-1, ythumb-1), (xthumb+thumb.shape[1], ythumb+thumb.shape[0]), YELLOW, 1)

            # FIXME: somewhere on the way we lost rect_ratio and/or the stretch parameters
            clip_x = xthumb + int(rect_x * thumbscale)
            clip_y = ythumb + int(rect_y * thumbscale)
            clip_x2 = xthumb + int((rect_x + rect_w) * thumbscale)
            clip_y2 = ythumb + int((rect_y + rect_h) * thumbscale)
            cv2.rectangle(self.debug_image, (clip_x - 1, clip_y - 1), (clip_x2, clip_y2), PURPLE, 2)

            self.debug_text(f'File: {Path(filename).name}', dx=THUMBNAIL_SIZE[0] + 10)
            self.debug_text(f'Size: {encoded_w}x{encoded_h}', dx=THUMBNAIL_SIZE[0] + 10)

            # Special effect: wavy deformation + zoom + motion blur
            # Still very WIP; but works rather well on 043887
            if wave != 0:
                # Apply wave; parameters are very guess-and-error
                freq = 1.5 * (1 + abs(wave) / 1000)
                shift = -0.1 * wave
                amplitude = min(0.15 * abs(wave), 20)
                #shift = -pow(0.1 * wave, 2)
                #amplitude = pow(0.1 * wave, 2)

                stride = np.arange(src.shape[1])
                for j in range(final_h):
                    dx = (0.5 + sin(6.28 * (freq * (j / final_h) + shift))) * amplitude * (0.6 * final_h / 100)
                    # Fast path XXX: doesn't work
                    #src[final_y+j,0:final_x-int(dx)] = src[final_y+j,int(dx):final_x]
                    # Slow path, must separate colour components
                    for channel in range(src.shape[2]):
                        tmp1 = np.interp(stride - dx, stride, src[final_y+j,:,channel], left=0, right=0)
                        src[final_y+j,:,channel] = tmp1

                # Apply blur; could be a separate effect; cap at 4%
                blur = min(4, abs(wave) * 0.25)

                # Box blur
                ksize = int(blur * final_h / 100)
                #ksize = int(blur * 2 * final_h / 100)
                if ksize > 1:
                    src = cv2.blur(src, (ksize, ksize))

                # Gaussian blur
                ksize = 1 + 2 * int(blur * final_h / 100)
                src = cv2.GaussianBlur(src, (ksize, ksize), 0)
                #src = cv2.GaussianBlur(src, (ksize, ksize), 0)

                fxinfo.append(f'Wave: f={freq:.2f} s={shift:.2f} a={amplitude:.2f}')
                fxinfo.append(f'Blur: {blur:.2f}%')

            for e in clipinfo:
                self.debug_text(e, dx=THUMBNAIL_SIZE[0] + 10, color=WHITE)

            if e := c.get('todo'):
                todo.append(e)
            for e in todo:
                self.debug_text(f'Todo: {e}', dx=THUMBNAIL_SIZE[0] + 10, color=RED)

            for e in fxinfo:
                self.debug_text(e, dx=THUMBNAIL_SIZE[0] + 10, color=YELLOW)

            self.debug_y = max(self.debug_y, ythumb + thumb.shape[0] + 20)

            if 'woodpanel' in transforms:
                cv2_blit(clips[-1][0], src[final_y:final_y+final_h,final_x:final_x+final_w], 0, 0)
            else:
                clips.append((src[final_y:final_y+final_h,final_x:final_x+final_w], blit_x, blit_y, opacity))

        if e := self.segment.desc.get('todo'):
            self.debug_text(f'Todo: {e}', color=RED)

        for e in errors:
            print(f'{self.segment.start:06d}: [{self.n}/{self.segment.count}: {e}')
            self.debug_text(f'Error: {e}', color=RED)

        # Perform composition
        self.target_image = np.zeros((TARGET_SIZE[1], TARGET_SIZE[0], 3), float)

        # If more than one image has opacity != 1.0, it is actually a blend
        # for cross-fade, so we will want to normalise it.
        total_opacity = 1.0
        if sum(1 for e in clips if e[3] not in [0.0, 1.0]) >= 2:
            total_opacity = sum(np.clip(e[3], 0.0, 1.0) for e in clips)
        self.debug_text(f'Total opacity: {total_opacity}', color=YELLOW)

        for image, x, y, opacity in clips:
            if total_opacity > 0.0:
                opacity /= total_opacity
            if opacity != 1.0:
                # FIXME: do this in linear colorspace one day?
                opacity = pow(opacity, 1.0)
            if opacity > 0.0:
                cv2_blit(self.target_image, image, x, y, opacity)

        # Blit final image to canvas
        x, y = MARGIN, MARGIN
        cv2_blit(self.debug_image, cv2.resize(self.target_image, SCALED_SIZE, interpolation=cv2.INTER_LANCZOS4).clip(0, 255).astype(np.uint8), x, y)
        cv2.rectangle(self.debug_image, (x-1, y-1), (x+SCALED_SIZE[0], y+SCALED_SIZE[1]), YELLOW, 1)

#        x, y = (CANVAS_WIDTH - WIDTH) // 2, (CANVAS_HEIGHT * 3 // 2 - HEIGHT) // 2
#        x2, y2 = x + WIDTH, y + HEIGHT
#
#        # Darken clipped area
#        self.debug_image[0:CANVAS_HEIGHT, 0:x] //= 3
#        self.debug_image[0:CANVAS_HEIGHT, x2:CANVAS_WIDTH] //= 3
#        self.debug_image[0:y, 0:CANVAS_WIDTH] //= 3
#        self.debug_image[y2:CANVAS_HEIGHT, 0:CANVAS_WIDTH] //= 3
#
#        # Draw border for destination frame
#        cv2.rectangle(self.debug_image, (x-2, y-2), (x2+1, y2+1), RED, 4)


    def save(self):
        # FIXME: handle gamma, maybe HDR etc.
        media_image = np.zeros((MEDIA_SIZE[1], MEDIA_SIZE[0], 3), float)
        cv2_blit(media_image, self.target_image, (MEDIA_SIZE[0] - TARGET_SIZE[0]) // 2, (MEDIA_SIZE[1] - TARGET_SIZE[1]) // 2)
        # FIXME: blue-noise dithering on separate colour layers
        dither = cv2_dither_matrix(MEDIA_SIZE[0], MEDIA_SIZE[1])
        cv2.imwrite(self.output_file, (media_image + dither[...,np.newaxis]).clip(0, 255).astype(np.uint8))

        cv2.imwrite(self.debug_file, self.debug_image)
        if self.n == self.segment.count // 2:
            cv2.imwrite(f'{DEBUG_ROOT}/{self.segment.start:06d}.jpeg', self.debug_image)



@click.command()
@click.option('-f', '--force',    is_flag=True, default=False)
@click.option('-n', '--dry-run',  is_flag=True, default=False)
@click.option('-v', '--verbose',  is_flag=True, default=False, help='Verbose output for debugging purposes.')
@click.option(      '--id',       default='', help='Only render frames from this segment (comma-separated).')
@click.option('-m', '--movie',    default='', help='Only render frames from this movie (comma-separated, partial match OK).')
@click.option('-p', '--property', default='', help='Only render frames with this clip property (comma-separated).')
def render(force: bool, dry_run: bool, verbose: bool, id: str, movie: str, property: str):

    """ Render movie frames. """
    config = RenderConfig(force=force, verbose=verbose, dry_run=dry_run)
    r = Renderer(config)

    starts = accumulate((0, *(s['count'] for s in r.segments)))

    for start, desc in zip(starts, r.segments):

        s = Segment(config, r, start, desc)

        # Early checks for requirements so we can skip whole segments
        if movie and not s.has_movie(movie):
            continue
        if property and not s.has_property(property):
            continue
        if id and s.id not in id.split(','):
            continue

        s.render_all()

        print(f' done.')

