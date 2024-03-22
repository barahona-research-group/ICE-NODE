import argparse
import logging

from ..base import Config
from ..ehr.example_datasets.mimiciv_aki import AKIMIMICIVDataset, TVxAKIMIMICIVDataset
from ..utils import load_config, translate_path

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--tvx-path', type=str, required=True)
    parser.add_argument('--dataset-config-override', type=str, required=False, default="")
    parser.add_argument('--tvx-config-override', type=str, required=False, default="")
    args = parser.parse_args()

    config = load_config(translate_path(args.config))
    config = {'dataset': Config.from_dict(config['dataset']),
              'tvx_ehr': Config.from_dict(config['tvx_ehr'])}
    logging.warning(args)

    for k, override in [('dataset', args.dataset_config_override), ('tvx', args.tvx_config_override)]:
        if override is not None and len(override) > 0 and override != '0':
            if ',' in override:
                delimiter = ','
            elif ';' in override:
                delimiter = ';'
            elif '&' in override:
                delimiter = '&'
            else:
                raise ValueError("Delimiter not found in override string.")

            for override_item in override.split(delimiter):
                key, value = override_item.split('=')
                config[k] = config[k].path_update(key, value)

    dataset = AKIMIMICIVDataset(config=config['dataset']).execute_pipeline()
    tvx = TVxAKIMIMICIVDataset(config=config['tvx_ehr'], dataset=dataset).execute_pipeline()
    tvx.save(translate_path(args.tvx_path), overwrite=True)
