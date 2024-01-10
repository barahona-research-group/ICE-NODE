"""."""

from string import Template
import json
import os
from pathlib import Path


def load_config(config_file):
    """Load a JSON file from `config_file`."""
    with open(config_file, encoding="utf-8") as json_file:
        return json.load(json_file)


def load_file(temp_file):
    with open(temp_file, encoding="utf-8") as text_file:
        return text_file.read()


IO_DIRS = load_file('template-io-dirs')
CMD_TEMPLATE_CONF = load_file('template-command-config')
CMD_TEMPLATE_OPTUNA = Template(load_file('template-command-optuna'))
CMD_TEMPLATE_EVAL = load_file('template-command-eval')

RCS_TEMPLATE = Template(load_file('rcs-template'))
DOC_TEMPLATE = Template(load_file('doc-template'))
LOCAL_TEMPLATE = Template(load_file('local-template'))

ENV_HOLDERS = load_config('env_input.json')


def replace_env_holders(txt):
    mapping = {}
    help_lines = ["Input Environment Variables:"]
    for holder, data in ENV_HOLDERS.items():
        holder_txt = '_'.join(('env', holder))
        if holder_txt in txt:
            mapping[holder_txt] = f"${data['env']}"
            help_lines.append(
                f"${data['env']}: {data['desc']} Example: {data['examples'][0]}"
            )

    mapping['temp_doc'] = '\n'.join(map(lambda l: f'# {l}', help_lines))
    return Template(txt).safe_substitute(**mapping)


def gen_local_cmd_optuna():
    mapping = {
        'temp_hours': 120,
        'temp_parallel': 1,
        'temp_num_trials': 30,
        'temp_training_time_limit': 72,
        'temp_job_id': 0
    }

    return CMD_TEMPLATE_OPTUNA.safe_substitute(**mapping)


def gen_doc_cmd_optuna():
    mapping = {
        'temp_hours': 120,
        'temp_parallel': 1,
        'temp_num_trials': 30,
        'temp_training_time_limit': 72,
        'temp_job_id': '"doc-${slurm_job_id}"'
    }

    return CMD_TEMPLATE_OPTUNA.safe_substitute(**mapping)


def gen_local_jobs():
    Path('local-jobs').mkdir(parents=True, exist_ok=True)

    for cmd, cmd_l in zip([gen_local_cmd_optuna(), CMD_TEMPLATE_CONF],
                          ['optuna', 'wconf']):
        for plat in ['cpu', 'gpu']:
            job_text = LOCAL_TEMPLATE.safe_substitute(
                **{
                    'temp_io_dirs': IO_DIRS,
                    'temp_command': cmd,
                    'temp_platform': f'"{plat}"'
                })
            job_text = replace_env_holders(job_text)
            fname = f'{cmd_l}-job-{plat}'
            with open(f'local-jobs/{fname}.sh', "w",
                      encoding="utf-8") as job_file:
                job_file.write(job_text)


def gen_doc_jobs():
    Path('doc-jobs').mkdir(parents=True, exist_ok=True)

    optuna_job_text = DOC_TEMPLATE.safe_substitute(
        **{
            'temp_io_dirs': IO_DIRS,
            'temp_command': gen_doc_cmd_optuna()
        })

    wconf_job_text = DOC_TEMPLATE.safe_substitute(
        **{
            'temp_io_dirs': IO_DIRS,
            'temp_command': CMD_TEMPLATE_CONF
        })

    with open('doc-jobs/optuna-job.sh', "w", encoding="utf-8") as job_file:
        job_file.write(replace_env_holders(optuna_job_text))
    with open('doc-jobs/wconf-job.sh', "w", encoding="utf-8") as job_file:
        job_file.write(replace_env_holders(wconf_job_text))


def gen_rcs_cmd_optuna(job_class, config, hours):
    cls_config = config[job_class]
    mapping = {
        'temp_hours': hours,
        'temp_parallel': cls_config['parallel'],
        'temp_num_trials': cls_config['n_trials'],
        'temp_training_time_limit': cls_config['training_time_limit'],
        'temp_job_id': '"$PBS_JOBID $PBS_ARRAY_INDEX"'
    }

    return CMD_TEMPLATE_OPTUNA.safe_substitute(**mapping)


def gen_rcs_job(job_class, config, wconfig):
    cls_config = config[job_class]

    if 'array' in cls_config and cls_config['array'] is not None \
            and cls_config['array'] > 1 and wconfig == False:
        job_array_head = f"#PBS -J 1-{cls_config['array']}"
    else:
        job_array_head = ""

    modules = cls_config.get('load_modules', [])
    module_lines = "\n".join(
        map(lambda module: f'module load {module}', modules))

    hours_mins = cls_config['hours']
    hours = hours_mins.split(':')[0]

    if wconfig:
        temp_command = CMD_TEMPLATE_CONF
    else:
        temp_command = gen_rcs_cmd_optuna(job_class, config, hours)

    plat = cls_config['platform']
    mapping = {
        'temp_spec': cls_config['spec'],
        'temp_hours_mins': hours_mins,
        'temp_array_place': job_array_head,
        'temp_modules_place': module_lines,
        'temp_io_dirs': IO_DIRS,
        'temp_command': temp_command,
        'temp_platform': f'"{plat}"'
    }

    job_file_text = RCS_TEMPLATE.safe_substitute(**mapping)

    Path('rcs-jobs').mkdir(parents=True, exist_ok=True)

    if wconfig:
        filename = f'rcs-jobs/wconf_{job_class}_job'
    else:
        filename = f'rcs-jobs/{job_class}_job'

    with open(filename, "w", encoding="utf-8") as job_file:
        job_file.write(replace_env_holders(job_file_text))


def gen_rcs_eval_job(job_class, config, array=True):
    cls_config = config[job_class]

    if cls_config.get('array', False) and array:
        job_array_head = f"#PBS -J 1-{cls_config['array']}"
    else:
        job_array_head = ""

    modules = cls_config.get('load_modules', [])
    module_lines = "\n".join(
        map(lambda module: f'module load {module}', modules))

    hours_mins = cls_config['hours']

    plat = cls_config['platform']
    mapping = {
        'temp_spec': cls_config['spec'],
        'temp_hours_mins': hours_mins,
        'temp_array_place': job_array_head,
        'temp_modules_place': module_lines,
        'temp_io_dirs': IO_DIRS,
        'temp_command': CMD_TEMPLATE_EVAL,
        'temp_platform': f'"{plat}"'
    }

    job_file_text = RCS_TEMPLATE.safe_substitute(**mapping)

    Path('rcs-jobs').mkdir(parents=True, exist_ok=True)

    if array:
        filename = f'rcs-jobs/eval_{job_class}_arrayjob'
    else:
        filename = f'rcs-jobs/eval_{job_class}_job'

    with open(filename, "w", encoding="utf-8") as job_file:
        job_file.write(replace_env_holders(job_file_text))


def gen_rcs_jobs():
    config = load_config('rcs_config.json')
    for job_class in config:
        gen_rcs_eval_job(job_class, config, False)
        gen_rcs_eval_job(job_class, config, True)
        for wconf in [True, False]:
            gen_rcs_job(job_class, config, wconf)


if __name__ == "__main__":
    gen_rcs_jobs()
    gen_doc_jobs()
    gen_local_jobs()
