import argparse
import logging
from ..ml.experiment import Experiment
from ..utils import load_config, translate_path
from ..base import Config

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--dataset-path', type=str, required=False, default="")
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--override', type=str, required=False, default="")
    args = parser.parse_args()

    config = load_config(translate_path(args.config))
    config = Config.from_dict(config)

    logging.warning(args)

    config = config.path_update('reporting.output_dir',
                                translate_path(args.output_path))

    if args.dataset_path is not None and len(args.dataset_path) > 0:
        config = config.path_update('dataset.path',
                                    translate_path(args.dataset_path))


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

    experiment = Experiment(config)
    experiment.run(tvx_ehr_path=config.dataset.path, prng_seed=42)

