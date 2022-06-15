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


def generate_job(job_class, config):
    cls_config = config[job_class]

    if 'array' in cls_config:
        job_array_head = f"#PBS -J 1-{cls_config['array']}"
    else:
        job_array_head = ""

    if 'cpu' in cls_config and cls_config['cpu'] == True:
        temp_cpu = '--cpu'
    else:
        temp_cpu = ''

    modules = cls_config.get('load_modules', [])
    module_lines = "\n".join(
        map(lambda module: f'module load {module}', modules))

    hours_mins = cls_config['hours']
    hours = hours_mins.split(':')[0]

    mapping = {
        'temp_spec': cls_config['spec'],
        'temp_hours_mins': hours_mins,
        'temp_array_place': job_array_head,
        'temp_modules_place': module_lines,
        'temp_hours': hours,
        'temp_parallel': cls_config['parallel'],
        'temp_num_trials': cls_config['n_trials'],
        'temp_training_time_limit': cls_config['training_time_limit'],
        'temp_cpu': temp_cpu
    }

    template = load_template('template-job')

    job_file_text = template.safe_substitute(**mapping)

    Path('job_files').mkdir(parents=True, exist_ok=True)
    with open(f'job_files/{job_class}_job', "w", encoding="utf-8") as job_file:
        job_file.write(job_file_text)


if __name__ == "__main__":
    config = load_config('config_job.json')

    for job_class in config:
        generate_job(job_class, config)
