#!/usr/bin/env python

import yaml
import glob
import cv2
import numpy as np
import sys
#from math import round
from pathlib import Path

# To generate a movie after this script has run:
# ffmpeg -safe 0 -f concat -i list.txt -i ../../original/*mkv -map 0:v:0 -map 1:a:0 -r 25 -c:v libx264 -pix_fmt yuv420p -c:a aac sidebyside.mkv

ORIGINAL_SIZE = (750, 358)
MINI_SIZE = (320, 240)
INFO_WIDTH = 600
MARGIN = 25

# 16:9 misses a lot of information, try 1.85:1
#TARGET_SIZE = (ORIGINAL_SIZE[0], int(ORIGINAL_SIZE[0] / 1.85))
#TARGET_SIZE = (ORIGINAL_SIZE[0], int(ORIGINAL_SIZE[0] / 1.77))
#TARGET_SIZE = ORIGINAL_SIZE
TARGET_SIZE = (1280, 620)
TARGET_SIZE = (1920, int(1920 * ORIGINAL_SIZE[1] / ORIGINAL_SIZE[0]))

CANVAS_SIZE = (
    (MARGIN + TARGET_SIZE[0] + MARGIN + INFO_WIDTH + MARGIN) // 4 * 4,
    (MARGIN + TARGET_SIZE[1] + MARGIN + ORIGINAL_SIZE[1] + MARGIN) // 4 * 4
)

DESC = 'C:/Users/sam/laclasse/data/laclasse.yaml'
DUMP_ROOT = 'C:/Users/sam/laclasse/dump-dvd'
FAST_ROOT = 'D:/clips/fast'
GOOD_ROOT = 'D:/clips/good'
SEARCH_ROOT = (GOOD_ROOT, FAST_ROOT)

DEBUG_ROOT = 'D:/final/compare'
OUTPUT_ROOT = 'D:/final/movie'

# Some convenient constants
FONT = cv2.FONT_HERSHEY_SIMPLEX
LARGE = INFO_WIDTH / 600
SMALL = INFO_WIDTH / 800
NEWLINE = int(INFO_WIDTH / 20)

WHITE = (255, 255, 255)
GRAY = (127, 127, 127)
RED = (127, 127, 255)
PURPLE = (255, 0, 255)
GREEN = (127, 255, 127)
YELLOW = (127, 255, 255)

def cv2_blit(bg, fg, x, y):
    h, w, _ = bg.shape
    sh, sw, _ = fg.shape

    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + sw, w), min(y + sh, h)
    if x1 < x2 and y1 < y2:
        sx1, sy1 = x1 - x, y1 - y
        sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)
        bg[y1:y2, x1:x2] += fg[sy1:sy2, sx1:sx2]

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

with open(DESC, encoding='utf-8') as f:
    data = yaml.safe_load(f)

presets = data['presets']
segments = data['segments']
movies = data['movies']

segment = 0
for s in data['segments']:

    dumpdir = glob.glob(f'{DUMP_ROOT}/{segment:06d}-*')
    if not dumpdir:
        raise f'{segment:06d}: cannot find source directory in {DUMP_ROOT}'

    count, clip_desc = s['count'], s.get('clips') or []

    dumpfiles = glob.glob(f'{dumpdir[0]}/*')
    if len(dumpfiles) != count:
        raise BaseException(f'{segment:06d}: need {count} frames but dump directory has {len(dumpfiles)}')

    output_dir = f'{OUTPUT_ROOT}/{segment:06d}'
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    debug_dir = f'{DEBUG_ROOT}/{segment:06d}'
    Path(debug_dir).mkdir(parents=True, exist_ok=True)

    force = False
    fatal = None

    # XXX: hacks to quickly force generation
    #force = force or any('searchers' in c.get('movie') or '' for c in clip_desc)
    #force = force or any('judge' in c.get('movie') or '' for c in clip_desc)
    #force = force or len(clip_desc) >= 2
    #force = True

    for n, orig in enumerate(dumpfiles):

        if len(sys.argv) >= 2 and sys.argv[1] != s.get('id'):
            continue

        output_file = f'{output_dir}/img{segment+n:06d}.jpeg'
        debug_file = f'{debug_dir}/img{segment+n:06d}.jpeg'

        # Skip existing files
        if not force and Path(output_file).is_file() and Path(debug_file).is_file():
            continue

        debug_image = np.zeros((CANVAS_SIZE[1], CANVAS_SIZE[0], 3), np.uint8)

        # Blit original dumped image
        # XXX: imread does not handle UTF-8 paths!
        #src = cv2.imread(orig)
        src = cv2.imdecode(np.fromfile(orig, dtype=np.uint8), cv2.IMREAD_COLOR)
        h, w, _ = src.shape
        x, y = MARGIN, MARGIN + TARGET_SIZE[1] + MARGIN
        x += max(0, TARGET_SIZE[0] - ORIGINAL_SIZE[0]) // 2 # Center image for aesthetic purposes
        cv2_blit(debug_image, src, x, y)
        cv2.rectangle(debug_image, (x-1, y-1), (x+w, y+h), YELLOW, 1)
        cv2.putText(debug_image, f'Original: {ORIGINAL_SIZE[0]}x{ORIGINAL_SIZE[1]} ({ORIGINAL_SIZE[0]/ORIGINAL_SIZE[1]:.2f}:1)',
                    (x + ORIGINAL_SIZE[0] + MARGIN, y + ORIGINAL_SIZE[1]), FONT, SMALL, GRAY, 3)

        # Text information
        xinfo, yinfo = MARGIN + TARGET_SIZE[0] + MARGIN, MARGIN
        cv2.putText(debug_image, f'S.{segment:06d} ({count})', (xinfo, yinfo + 25), FONT, LARGE, WHITE, 3)
        cv2.putText(debug_image, f'F.{segment+n:06d} (+{n})', (xinfo + 300, yinfo + 25), FONT, LARGE, GREEN, 3)
        yinfo += NEWLINE

        cv2.putText(debug_image, f'Target: {TARGET_SIZE[0]}x{TARGET_SIZE[1]} ({TARGET_SIZE[0]/TARGET_SIZE[1]:.2f}:1)', (xinfo, yinfo + 25), FONT, SMALL, GRAY, 3)
        yinfo += NEWLINE
        if scenes := s.get('scenes'):
            cv2.putText(debug_image, f'Scenes: {" ".join(scenes)}', (xinfo, yinfo + 25), FONT, SMALL, GRAY, 3)
            yinfo += NEWLINE

        errors = []
        clips = []

        # Enumerate clips
        for i, c in enumerate(clip_desc):
            clip_movie, clip_segment, clip_offset = c.get('movie'), c.get('segment'), c.get('offset')

            # Compute opacity early on; if it is zero, we can bail out quickly
            opacity = 1.0
            if l := c.get('fade-in'):
                opacity = min(opacity, max(0.0, (n - l[0]) / (l[1] - l[0])))
            if l := c.get('fade-out'):
                opacity = min(opacity, max(0.0, (l[1] - n) / (l[1] - l[0])))
            if opacity <= 0.0:
                continue

            if clip_movie is None:
                errors.append(f'no movie associated with clip {i}')
                continue

            clip_presets = presets.get(clip_movie) or {}

            cv2.putText(debug_image, f'Clip {i}: {clip_movie}', (xinfo, yinfo + 25), FONT, SMALL, WHITE, 3)
            yinfo += NEWLINE

            clipinfo = []
            fxinfo = []
            todo = []

            if clip_segment is None or clip_offset is None:
                cv2.putText(debug_image, f'Todo: missing segment or offset information', (xinfo + 20, yinfo + 25), FONT, SMALL, RED, 3)
                yinfo += NEWLINE
                continue

            # Compute source offset
            current_offset = n
            sequence = None
            match playback := c.get('playback'):
                case None: pass
                case 'reverse': current_offset = count - 1 - n
                case _:
                    seq = parse_playback(playback)
                    if seq is None:
                        errors.append(f'cannot parse playback {playback}')
                    elif len(seq) != count:
                        errors.append(f'playback length {len(seq)}, expected {count}')
                    else:
                        current_offset = seq[n]
            if playback is not None:
                fxinfo.append(f'Playback: {playback}')

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
                errors.append(f'[{n}/{count}] cannot find image for {srcglob} in {', '.join(SEARCH_ROOT)}')
                continue

            # Apply letterbox cropping if the value is known to us
            if crop := clip_presets.get(f'letterbox-{src.shape[0]}'):
                if crop > 0:
                    src = src[crop:-crop,:]
            if crop := clip_presets.get(f'letterbox-{src.shape[1]}'):
                if crop > 0:
                    src = src[:,crop:-crop]

            src_h, src_w, _ = src.shape

            # XXX: apply stretch to DVD images, this is for debugging purposes
            display_h, display_w = src_h, src_w
            if 'revolution' in clip_movie and display_w == 720:
                display_w = 875
            if 'firecreek' in clip_movie and display_w == 1920:
                display_w = 2543
            if 'arrangement' in clip_movie and display_w == 1280:
                todo.append(f'Must unscale to 960')

            clipinfo.append(f'DAR: {display_w/display_h:.2f}:1')

            dst_w, dst_h = display_w, display_h
            match stretch := c.get('stretch') or clip_presets.get('stretch'):
                case None: pass
                case _: dst_w *= stretch
            if stretch is not None:
                fxinfo.append(f'Stretch: {stretch}')

            # Compute uniform scaling to target size, preserving aspect ratio
            if dst_w / dst_h > TARGET_SIZE[0] / TARGET_SIZE[1]:
                dst_w, dst_h = int(round(dst_w * TARGET_SIZE[1] / dst_h)), TARGET_SIZE[1]
            else:
                dst_w, dst_h = TARGET_SIZE[0], int(round(dst_h * TARGET_SIZE[0] / dst_w))

            # Special effects
            match transform := c.get('transform'):
                case None: pass
                case 'hflip': src = cv2.flip(src, 1)
                case _: errors.append(f'invalid transform {transform}')
            if transform is not None:
                fxinfo.append(f'Transform: {transform}')

            # Apply scaling to the image; after this step, our frame lies somewhere
            # in there as a rectangle of size TARGET_SIZE.
            if src.shape[:2] != (dst_h, dst_w):
                src = cv2.resize(src, (dst_w, dst_h), interpolation=cv2.INTER_LANCZOS4)

            # Apply gravity
            rect_x, rect_y = (dst_w - TARGET_SIZE[0]) // 2, (dst_h - TARGET_SIZE[1]) // 2
            match gravity := c.get('gravity') or clip_presets.get('gravity'):
                case None: pass
                case _:
                    for g in gravity.split(' '):
                        match g[0]:
                            case 'r': rect_x += (dst_w - TARGET_SIZE[0]) * int(g[1:]) // 200
                            case 'l': rect_x -= (dst_w - TARGET_SIZE[0]) * int(g[1:]) // 200
                            case 'd': rect_y += (dst_h - TARGET_SIZE[1]) * int(g[1:]) // 200
                            case 'u': rect_y -= (dst_h - TARGET_SIZE[1]) * int(g[1:]) // 200
                            case _: errors.append(f'invalid gravity {gravity}')
            if gravity is not None:
                fxinfo.append(f'Gravity: {gravity}')

            rect_x = min(max(0, rect_x), dst_w - TARGET_SIZE[0])
            rect_y = min(max(0, rect_y), dst_h - TARGET_SIZE[1])
            rect_w = TARGET_SIZE[0]
            rect_h = TARGET_SIZE[1]

            if opacity != 1.0:
                fxinfo.append(f'Opacity: {opacity:.2f}')

            # "Quality" is the density of source pixels in the target image, a rough
            # estimate of the number of pixels lost through scaling in both dimensions.
            quality = min(1, src_w / dst_w) * min(1, src_h / dst_h)
            clipinfo.append(f'Quality: {100 * quality:.1f}%')

            # Add debug info etc.
            ythumb = yinfo
            thumbscale = min(MINI_SIZE[0] / display_w, MINI_SIZE[1] / display_h)
            thumb = cv2.resize(src, (int(display_w * thumbscale), int(display_h * thumbscale)), interpolation=cv2.INTER_LANCZOS4)
            #if transform == 'hflip': thumb = cv2.flip(thumb, 1)
            x, y = xinfo, ythumb + 10
            cv2_blit(debug_image, thumb.clip(0, 255).astype(np.uint8), x, y)
            cv2.rectangle(debug_image, (x-1, y-1), (x+thumb.shape[1], y+thumb.shape[0]), YELLOW, 1)

            clip_x = x + int(rect_x / dst_w * display_w * thumbscale)
            clip_y = y + int(rect_y / dst_h * display_h * thumbscale)
            clip_x2 = x + int((rect_x + rect_w) / dst_w * display_w * thumbscale)
            clip_y2 = y + int((rect_y + rect_h) / dst_h * display_h * thumbscale)
            cv2.rectangle(debug_image, (clip_x - 1, clip_y - 1), (clip_x2, clip_y2), PURPLE, 2)

            cv2.putText(debug_image, f'File: {Path(filename).name}', (xinfo + MINI_SIZE[0] + 10, yinfo + 25), FONT, SMALL, GRAY, 3)
            yinfo += NEWLINE
            cv2.putText(debug_image, f'Size: {src_w}x{src_h}', (xinfo + MINI_SIZE[0] + 10, yinfo + 25), FONT, SMALL, GRAY, 3)
            yinfo += NEWLINE

            for e in clipinfo:
                cv2.putText(debug_image, e, (xinfo + MINI_SIZE[0] + 10, yinfo + 25), FONT, SMALL, WHITE, 3)
                yinfo += NEWLINE

            if e := c.get('todo'):
                todo.append(e)
            for e in todo:
                cv2.putText(debug_image, f'Todo: {e}', (xinfo + MINI_SIZE[0] + 10, yinfo + 25), FONT, SMALL, RED, 3)
                yinfo += NEWLINE

            for e in fxinfo:
                cv2.putText(debug_image, e, (xinfo + MINI_SIZE[0] + 10, yinfo + 25), FONT, SMALL, YELLOW, 3)
                yinfo += NEWLINE

            yinfo = max(yinfo, ythumb + thumb.shape[0] + 20)

            clips.append((src[rect_y:rect_y+rect_w,rect_x:rect_x+rect_w], opacity))

        if e := s.get('todo'):
            cv2.putText(debug_image, f'Todo: {e}', (xinfo, yinfo + 25), FONT, SMALL, RED, 3)
            yinfo += NEWLINE

        for e in errors:
            print(e)
            cv2.putText(debug_image, f'Error: {e}', (xinfo, yinfo + 25), FONT, SMALL, RED, 3)
            fatal = e
            yinfo += NEWLINE

        # Perform composition
        target_image = np.zeros((TARGET_SIZE[1], TARGET_SIZE[0], 3), float)

        # If more than one image, opacity is actually a blend for cross-fade, so
        # we will want to normalise it.
        total_opacity = sum(np.clip(opacity, 0.0, 1.0) for _, opacity in clips) if len(clips) > 2 else 1.0

        for image, opacity in clips:
            if total_opacity > 0.0:
                opacity /= total_opacity
            if opacity != 1.0:
                # FIXME: do this in linear colorspace one day?
                image = image * pow(opacity, 1.0)
            if opacity > 0.0:
                cv2_blit(target_image, image, 0, 0)

        # Blit final image to canvas
        x, y = MARGIN, MARGIN
        cv2_blit(debug_image, target_image.clip(0, 255).astype(np.uint8), x, y)
        cv2.rectangle(debug_image, (x-1, y-1), (x+TARGET_SIZE[0], y+TARGET_SIZE[1]), YELLOW, 1)

#        x, y = (CANVAS_WIDTH - WIDTH) // 2, (CANVAS_HEIGHT * 3 // 2 - HEIGHT) // 2
#        x2, y2 = x + WIDTH, y + HEIGHT
#
#        # Darken clipped area
#        debug_image[0:CANVAS_HEIGHT, 0:x] //= 3
#        debug_image[0:CANVAS_HEIGHT, x2:CANVAS_WIDTH] //= 3
#        debug_image[0:y, 0:CANVAS_WIDTH] //= 3
#        debug_image[y2:CANVAS_HEIGHT, 0:CANVAS_WIDTH] //= 3
#
#        # Draw border for destination frame
#        cv2.rectangle(debug_image, (x-2, y-2), (x2+1, y2+1), RED, 4)

        # FIXME: handle gamma, maybe HDR etc.
        cv2.imwrite(output_file, target_image.clip(0, 255).astype(np.uint8))

        cv2.imwrite(debug_file, debug_image)
        if n == count // 2:
            cv2.imwrite(f'{DEBUG_ROOT}/{segment:06d}.jpeg', debug_image)

    print(f'{segment:06d}: {fatal or 'OK'}')

    segment += count

print(f'{segment:06d}: END')
