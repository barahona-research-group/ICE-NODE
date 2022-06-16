"""."""

import json
from string import Template
from pathlib import Path


def load_config(config_file):
    """Load a JSON file from `config_file`."""
    with open(config_file, encoding="utf-8") as json_file:
        return json.load(json_file)


def load_template(temp_file):
    with open(temp_file, encoding="utf-8") as text_file:
        return Template(text_file.read())


MAIN_TEMPLATE = load_template('template-job')
CMD_TEMPLATE_CONF = load_template('template-job-command-config')
CMD_TEMPLATE_OPTUNA = load_template('template-job-command-optuna')


def generate_cmd_optuna(job_class, config, hours):
    cls_config = config[job_class]
    mapping = {
        'temp_hours': hours,
        'temp_parallel': cls_config['parallel'],
        'temp_num_trials': cls_config['n_trials'],
        'temp_training_time_limit': cls_config['training_time_limit'],
    }

    return CMD_TEMPLATE_OPTUNA.safe_substitute(**mapping)


def generate_cmd_conf():
    return CMD_TEMPLATE_CONF.safe_substitute()


def generate_job(job_class, config, wconfig):
    cls_config = config[job_class]

    if 'array' in cls_config:
        job_array_head = f"#PBS -J 1-{cls_config['array']}"
    else:
        job_array_head = ""

    modules = cls_config.get('load_modules', [])
    module_lines = "\n".join(
        map(lambda module: f'module load {module}', modules))

    hours_mins = cls_config['hours']
    hours = hours_mins.split(':')[0]

    if wconfig:
        temp_command = generate_cmd_conf()
    else:
        temp_command = generate_cmd_optuna(job_class, config, hours)

    mapping = {
        'temp_spec': cls_config['spec'],
        'temp_hours_mins': hours_mins,
        'temp_array_place': job_array_head,
        'temp_modules_place': module_lines,
        'temp_command': temp_command,
        'temp_platform': cls_config['platform']
    }

    job_file_text = MAIN_TEMPLATE.safe_substitute(**mapping)

    Path('job_files').mkdir(parents=True, exist_ok=True)

    if wconfig:
        filename = f'job_files/wconf_{job_class}_job'
    else:
        filename = f'job_files/{job_class}_job'

    with open(filename, "w", encoding="utf-8") as job_file:
        job_file.write(job_file_text)


if __name__ == "__main__":
    config = load_config('config_job.json')
    for job_class in config:
        for wconf in [True, False]:
            generate_job(job_class, config, wconf)
