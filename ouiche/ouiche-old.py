#!/usr/bin/env python

import re
import sys
from pathlib import Path

def usage():
    print(f'Usage: {sys.argv[0]} <COMMAND> <ARGS...>')
    print(f'Available commands:')
    print(f'  renumber <ID>       Safely renumber files in a directory')

def fatal(s):
    print(f'{sys.argv[0]}: {s}')
    sys.exit(1)


def renumber(args):
    if len(args) == 0:
        fatal(f'missing argument (ID of first frame)')

    newstart = int(args[0], 10)
    if newstart > 999999:
        fatal(f'invalid argument {newstart}')

    files = [f for f in Path('./').iterdir() if f.is_file()]
    if len(files) == 0:
        fatal(f'no files to rename')

    numbers = set()
    prefix, suffix = None, None
    for f in files:
        sn = re.sub(r'[^0-9]*([0-9]+)[^0-9]*$', r'\1', f.name)
        if len(sn) != 6: fatal(f'{f.name} does not have a 6-digit number, found {sn} instead')
        n = int(sn, 10)
        if n in numbers: fatal(f'{f.name} has duplicate id {sn}')
        numbers.add(n)
        if prefix is None or suffix is None:
            prefix = re.sub(r'[0-9].*', '', f.name)
            suffix = re.sub(r'.*[0-9]', '', f.name)
        else:
            expected = f'{prefix}{sn}{suffix}'
            if f.name != expected:
                fatal(f'{f.name} has invalid name, expected {expected} for consistency')

    for n in range(min(numbers), max(numbers) + 1):
        if n not in numbers:
            fatal(f'directory goes from {min(numbers):06d} to {max(numbers):06d} but is missing id {n:06d}')

    if newstart == min(numbers):
        print(f'Everything already named correctly ({files[0].name} .. {files[-1].name}), doing nothing')
    else:
        print(f'Renaming {len(files)} files...')
        delta = newstart - min(numbers)
        order = sorted(numbers) if delta < 0 else reversed(sorted(numbers))
        for n in order:
            src, dst = f'{prefix}{n:06d}{suffix}', f'{prefix}{n+delta:06d}{suffix}'
            Path(src).rename(dst)
            
    #if len(numbers) != len(files):
#numbers:
#    print(numbers)
    #if any(f.suffix != files[0].suffix for f in files):
#        fatal(f'
#    suffix = files[0].suffix
 #   print(suffix)
#    #for d in args:


if len(sys.argv) < 2:
    print(f'{sys.argv[0]}: missing command')
    usage()
    sys.exit(1)

if all(re.match('^[-+0-9 ]+$', s) for s in sys.argv[1:]):
    expr = re.sub(' +', '', ''.join(sys.argv[1:]))
    if not re.match('^-?[0-9]+([-+][0-9]+)*$', expr):
        fatal(f'Bad expression {expr}')
    ret = 0
    for s in re.findall('([-+]?)([0-9]+)', expr):
        n = int(s[1], 10)
        ret += -n if s[0] == '-' else n
    print(f'{ret:06d}')
    sys.exit(0)

match sys.argv[1]:
    case 'renumber': renumber(sys.argv[2:])
    case 'seq':
        start = int(sys.argv[2], 10)
        count = int(sys.argv[3], 10)
        for n in range(count):
            print(f'{start+n:06d}')
    case None: usage(); sys.exit(1)
sys.exit(0)
