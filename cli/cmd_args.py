import argparse
from typing import List
from pydoc import locate

from ..icenode.utils import load_config


def get_cmd_parser(cmd_keys: List[str]) -> argparse.ArgumentParser:
    args = load_config('cmd_args.json')
    for k, v in args.items():
        if 'type' in v:
            v['type'] = locate(v['type'])

    parser = argparse.ArgumentParser()
    for key in cmd_keys:
        parser.add_argument(key, **args[key])

    return parser
