import argparse
import logging

from ..base import Config
from ..ml.experiment import Experiment
from ..utils import load_config, translate_path

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--dataset-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--override', type=str, required=False, default="")
    args = parser.parse_args()

    config = load_config(translate_path(args.config))
    config = Config.from_dict(config)

    logging.warning(args)

    config = config.path_update('reporting.output_dir',
                                translate_path(args.output_path))

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
            if isinstance(value, str):
                if value == 'True':
                    value = True
                elif value == 'False':
                    value = False

                elif value.lower() == 'none' or value.lower() == 'null':
                    value = None
                elif value.isdigit():
                    value = int(value)

                elif '.' in value:
                    try:
                        value = float(value)
                    except ValueError:
                        pass

            config = config.path_update(key, value)

    experiment = Experiment(config)
    experiment.run(tvx_ehr_path=args.dataset_path, prng_seed=42)
