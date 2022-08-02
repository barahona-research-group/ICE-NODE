import argparse
from typing import List
from pydoc import locate

from icenode.utils import load_config
from icenode import ml
from icenode import ehr
from icenode import embeddings as E

help_expand = dict(dataset_labels=ehr.datasets,
                   dx_scheme_labels=[s for s in ehr.code_scheme if 'dx' in s],
                   pr_scheme_labels=[s for s in ehr.code_scheme if 'pr' in s],
                   dx_outcome_labels=[o for o in ehr.outcome_conf_files],
                   emb_labels=[f'{v} ({k})' for k, v in E.short_tag.items()],
                   model_labels=ml.model_cls)

help_expand = {k: ', '.join(v) for k, v in help_expand.items()}


def get_cmd_parser(cmd_keys: List[str]) -> argparse.ArgumentParser:
    args = load_config('cmd_args.json')
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
            l.append(f"--{k.replace('_','-')}")
            l.append(str(v))
    return l
