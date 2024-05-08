import argparse
import os
from pydoc import locate
from typing import List

from ..utils import load_config

_DIR = os.path.dirname(__file__)

help_expand = dict()

help_expand = {k: ', '.join(v) for k, v in help_expand.items()}


def get_cmd_parser(cmd_keys: List[str]) -> argparse.ArgumentParser:
    args = load_config(os.path.join(_DIR, 'cmd_args.json'))
    for k, v in args.items():
        if 'type' in v:
            v['type'] = locate(v['type'])
        if 'help' in v:
            v['help'] = v['help'].format(**help_expand)

    parser = argparse.ArgumentParser()
    for key in cmd_keys:
        parser.add_argument(key, **args[key])

    return parser


def forward_cmd_args(args, exclude=[]):
    l = []
    for k, v in vars(args).items():
        if k not in exclude and v is not None:
            l.append(f"--{k.replace('_', '-')}")
            l.append(str(v))
    return l
