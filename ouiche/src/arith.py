#!/usr/bin/env python

import re

def try_parse(args):

    def_tokens = [
        ('name',    r'(?P<prefix>[^-+0-9]*)(?P<id>[0-9]{6})(?P<suffix>[^-+0-9]*)'),
        ('num',     r'\d+'),
        ('op',      r'[-+]'),
        ('ignore',  r'[ \t]+'),
        ('error',   r'.'),
    ]
    r_tokens = '|'.join(f'(?P<{t}>{r})' for t, r in def_tokens)

    ret, op, prefix, suffix = None, None, None, None
    for t in re.finditer(r_tokens, ''.join(args)):
        match t.lastgroup:
            case 'name' | 'num':
                if prefix is None or suffix is None:
                    prefix, suffix = t.group('prefix'), t.group('suffix')
                val = int(t.group('id') or t.group(), 10)
                if ret is None: ret = val
                elif op is None: return False
                else: ret, op = ret + val if op[0] == '+' else ret - val, None
            case 'op':
                if op is not None or ret is None: return False
                op = t.group()
            case 'error':
                return False
    if op is not None or ret is None: return False

    print(f'{prefix or ''}{ret % 1000000:06d}{suffix or ''}')
    return True
