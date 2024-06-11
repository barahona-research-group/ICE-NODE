import argparse
import logging
from ..ml.evaluation import  EvaluationConfig, Evaluation
from ..utils import load_config, translate_path
from ..base import Config

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--override', type=str, required=False, default="")
    parser.add_argument('--experiments-dir', type=str, required=True)
    parser.add_argument('--db', type=str, required=True)

    args = parser.parse_args()

    config = load_config(translate_path(args.config))
    config = Config.from_dict(config)

    logging.warning(args)

    config = config.path_update('experiments_dir',
                                translate_path(args.experiments_dir))
    config = config.path_update('db', args.db)

    if args.override is not None and len(
            args.override) > 0 and args.override != '0':
        if ',' in args.override:
            splitter = ','
        elif ';' in args.override:
            splitter = ';'
        elif '&' in args.override:
            splitter = '&'

        for override in args.override.split(splitter):
            key, value = override.split('=')
            config = config.path_update(key, value)

    Evaluation(config).start()
