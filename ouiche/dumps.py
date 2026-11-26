#!/usr/bin/env python

import click
import click_log
import cv2
import logging
import numpy as np
import re
from pathlib import Path

logger = logging.getLogger(__name__)
click_log.basic_config(logger)

def pathlib_glob(pattern: Path):
    from glob import glob
    for f in map(Path, glob(str(pattern))):
        yield f

@click.command()
@click.option('-t', '--threshold', default=0.0, help='Threshold value for absolute error for --delete')
@click.option('-r', '--rename',    is_flag=True, default=False, help='Renumber files along decimation')
@click.option('-f', '--force',     is_flag=True, default=False, help='Force renaming even if files are badly named')
@click.option('-d', '--delete',    is_flag=True, default=False, help='Delete duplicate frames')
@click.option('-c', '--compare',   is_flag=True, default=False, help='Compare consecutive frames')
@click.argument('directory', nargs=1, required=False,
                type=click.Path(exists=True, file_okay=False, dir_okay=True),
                help='Dump directory to analyse')
@click_log.simple_verbosity_option(logger)
def sequence(directory: str, threshold: float, rename: bool, force: bool, delete: bool, compare: bool):

    if directory is None:
        directory = '.'

    files = [*sorted(f for f in pathlib_glob(Path(directory) / '*') if f.is_file())]

    if len(files) <= 1:
        raise click.BadParameter(f'{directory} must contain at least 2 files')

    r_pattern = re.compile('^(.*[^0-9])([0-9]{3,})([^0-9].*)$')
    offset, m = next((n, m) for n, m in enumerate(r_pattern.match(f.name) for f in files) if m)
    if not m:
        raise click.BadParameter(f'at least one file in {directory} must be numbered')
    prefix, start, digits, suffix = m.group(1), int(m.group(2), 10) - offset, len(m.group(2)), m.group(3)
    pattern = f'{prefix}%0{digits}d{suffix}'

    #    raise click.BadParameter(f'{files[0]} does not contain an index number')

    logger.info(f"Analysing {len(files)} files in `{directory}'")
    if rename:
        logger.info(f'Renumbering to {pattern}, starting at {start:0{digits}d}')

    deleted, renamed = 0, 0

    prev_f, prev_p = None, None
    for n, f in enumerate(files):
        m = r_pattern.match(f.name)
        if not m and not force:
            raise click.BadParameter(f'{f} does not follow {pattern} pattern')

        if compare or delete:
            dup = False
            p = cv2.imread(f).astype(float)
            if prev_p is not None:
                delta = np.abs(p - prev_p)
                dmax = np.max(delta)
                dcount = np.count_nonzero(delta) if dmax > 0.0 else 0
                if dmax <= threshold:
                    logger.info(f'Dup! {f} (prev={prev_f}, max={dmax}, count={dcount}, avg={np.average(delta):.3f})')
                    if delete:
                        f.unlink()
                        deleted += 1
                    dup = True
                else:
                    logger.info(f'     {f} (prev={prev_f}, max={dmax}, count={dcount}, avg={np.average(delta):.3f})')

        if rename and not (delete and dup):
            expected = Path(directory) / f'{prefix}{start:0{digits}d}{suffix}'
            if expected != f:
                if expected.exists():
                    raise click.BadParameter(f'{f} cannot be renamed to {expected.name} because it already exists')
                logger.info(f'Renaming: {f} -> {expected}')
                f.rename(expected)
                renamed += 1

        if (compare or delete) and not dup:
            prev_f, prev_p = f, p
        if not (delete and dup):
            start += 1

    logger.info(f'Deleted {deleted} files, renamed {renamed} files')
