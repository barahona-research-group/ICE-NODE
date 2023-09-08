import argparse
import logging
from ..ml import Experiment, InpatientExperiment
from ..utils import load_config, translate_path
from ..base import Config

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--override', type=str, required=False)
    args = parser.parse_args()

    config = load_config(translate_path(args.config))
    config = Config.from_dict(config)
    if args.override is not None:
        for override in args.override.split(','):
            key, value = override.split('=')
            config = config.path_update(key, value)

    if args.settings == 'inpatient':
        experiment = InpatientExperiment(config)
    else:
        experiment = Experiment(config)
    experiment.run()
