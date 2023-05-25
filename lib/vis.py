"""."""

from typing import Callable, Dict, Optional, Union, Tuple, Type, List
import glob
import math
from collections import defaultdict
from absl import logging

import numpy as np
import pandas as pd
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt

from . import utils as U
from . import ml
from . import ehr
from . import metric as M

ExtractorType = Callable[[pd.DataFrame, Union[int, str]],
                         Union[Tuple[int, float], float]]
ExtractorDict = Dict[str, ExtractorType]


def performance_traces(train_dir: str, metric_extractor: ExtractorDict):

    clf_dirs = glob.glob(f'{train_dir}/*')
    clf_dirs = {d.split('/')[-1]: d for d in clf_dirs}
    traces_df = []
    for clf, clf_dir in clf_dirs.items():
        try:
            df = pd.read_csv(f'{clf_dir}/_val_evals.csv.gz')
        except:
            continue
        data = {}
        for metric, extractor in metric_extractor.items():
            vals = [extractor(df, i) for i in range(len(df))]
            data[metric] = vals

        df = pd.melt(pd.DataFrame(data),
                     value_vars=df.columns,
                     id_vars='index',
                     var_name='metric',
                     value_name='value')
        df['model'] = clf
        traces_df.append(df)

    return pd.concat(traces_df)


def models_from_configs(train_dir: str,
                        model_cls: Dict[str, Type[ml.AbstractModel]],
                        subject_interface: ehr.Subject_JAX,
                        subject_splits: Tuple[List[int], ...]):
    models = {}
    clf_dirs = glob.glob(f'{train_dir}/*')
    clf_dirs = {d.split('/')[-1]: d for d in clf_dirs}

    for clf, clf_dir in clf_dirs.items():
        try:
            config = U.load_config(f'{clf_dir}/config.json')
            key = jrandom.PRNGKey(0)
            model_class = None
            for model_name in model_cls.keys():
                if model_name.lower() in clf.lower():
                    model_class = model_cls[model_name]

            models[clf] = model_class.from_config(config, subject_interface,
                                                  subject_splits[0], key)
        except Exception as e:
            logging.warning(f'{e}, clf: {clf}')

    return models


def probe_model_snapshots(train_dir: str,
                          metric_extractor: ExtractorDict,
                          selection_metric: Optional[str] = None,
                          models: Dict[str, ml.AbstractModel] = None):
    data = defaultdict(list)
    clf_dirs = glob.glob(f'{train_dir}/*')
    clf_dirs = {d.split('/')[-1]: d for d in clf_dirs}

    for clf, clf_dir in clf_dirs.items():
        try:
            df = pd.read_csv(f'{clf_dir}/_val_evals.csv.gz')
        except Exception as e:
            logging.warning(e)
            continue

        data['model'].append(clf)
        for metric, extractor in metric_extractor.items():
            index, value = extractor(df, 'best')
            data[f'{metric}_idx'].append(index)
            data[f'{metric}_val'].append(value)
            logging.warning(f'{clf}, {metric}')
            if metric == selection_metric:
                tarfname = f'{clf_dir}/params.tar.bz2'
                membername = f'step{index:04d}.eqx'
                try:
                    id1 = id(models[clf])
                    models[clf] = models[clf].load_params_from_tar_archive(
                        tarfname, membername)
                    id2 = id(models[clf])
                    logging.warning(
                        f'Loaded {clf} from {tarfname}:{membername}. id1: {id1}, id2: {id2}'
                    )
                except Exception as e:
                    logging.warning(e)
                    logging.warning(
                        f'tarfname: {tarfname}, membername: {membername}')

    return pd.DataFrame(data, index=data['model'])


def auc_upset(auc_metric: M.DeLongTest, auc_tests: pd.DataFrame,
              models: List[str], p_value: float, min_auc: float):
    models = tuple(sorted(models))

    auc_tests = auc_metric.filter_results(auc_tests, models, min_auc)

    # Rows when no significant difference of AUCs among all pairs of models.
    nodiff_tests = auc_metric.insignificant_difference_rows(
        auc_tests, models, p_value)

    # The intersection of all models.
    auc_sets = {m: set(nodiff_tests.index) for m in models}

    # The dichotomous tests
    auc_tests = auc_tests.drop(index=nodiff_tests.index)

    # Assign each code to the best model (max AUC), then assign it as well
    # to any model with no significant difference with the best.
    for code_index in auc_tests.index:
        auc = auc_metric.value_extractor(code_index=code_index, field='auc')
        p_val = auc_metric.value_extractor(code_index=code_index,
                                           field='p_val')

        # The first candidate model with the highest AUC.
        seed = max(models, key=lambda m: auc(auc_tests, model=m))
        equivalent_models = [
            m for m in models
            if m == seed or p_val(auc_tests, pair=(seed, m)) > p_value
        ]
        for m in equivalent_models:
            auc_sets[m].add(code_index)

    # Prepare for using Upset plot -> DataFrame Layout (passed to `from_indicators`)
    index = sorted(auc_tests.index)
    membership_indicator = {
        m: [i in auc_sets[m] for i in index]
        for m in models
    }
    membership_df = pd.DataFrame(membership_indicator, index=index)

    return membership_df, (nodiff_tests.index, auc_tests.index)


def selected_auc_barplot(clfs, auctest_df, horizontal=False, rotate_ccs=True):

    clfs = sorted(clfs)
    auc_df = []

    for clf in clfs:
        comp_auc = auctest_df[f'AUC({clf})']
        comp_var = auctest_df[f'VAR[AUC({clf})]']
        comp_std = comp_var.apply(np.sqrt)
        comp_desc = auctest_df['DESC'].apply(
            lambda t: t if len(t) < 15 else t.replace(' ', '\n'))
        df = pd.DataFrame({
            'AUC': comp_auc,
            'std': comp_std,
            'CCS': comp_desc,
            'Classifier': clf
        })
        auc_df.append(df)
    auc_df = pd.concat(auc_df)

    min_auc_tick = int(auc_df['AUC'].min() * 20) / 20
    max_auc_tick = int(auc_df['AUC'].max() * 20 + 1) / 20

    vals = auc_df.pivot(index='CCS', columns='Classifier', values='AUC')
    err = auc_df.pivot(index='CCS', columns='Classifier', values='std')

    icenode_idx = clfs.index('ICE-NODE')

    colors = ['green', 'gray', 'skyblue', 'brown', 'purple', 'navy', 'pink']
    patterns = ['o', '', '+', '', '', '', '/']
    patterns[icenode_idx] = 'x'

    colors[icenode_idx] = 'white'

    pltbarconf = dict(rot=0,
                      figsize=(10, 10),
                      width=0.7,
                      error_kw=dict(lw=5,
                                    capsize=8,
                                    capthick=5,
                                    ecolor='salmon'),
                      color=colors,
                      edgecolor='black')
    if horizontal:
        # plot vals with yerr
        ax = vals.plot.barh(xerr=err, **pltbarconf)
        plt.xlabel('AUC', fontsize=32)
        plt.xticks(fontsize=30)
        plt.xlim(min_auc_tick, max_auc_tick)

        xstart, xend = ax.get_xlim()
        ax.xaxis.set_ticks(np.arange(xstart, xend + 0.01, 0.05))

        plt.yticks(fontsize=24)

        plt.ylabel(None)
        ax.tick_params(bottom=True, left=False)

        ax.xaxis.grid(color='gray', linestyle='dashed')
        ax.xaxis.set_zorder(3)

    else:
        # plot vals with yerr
        ax = vals.plot.bar(yerr=err, **pltbarconf)
        plt.ylabel('AUC', fontsize=32)
        plt.yticks(fontsize=24)
        plt.ylim(min_auc_tick, max_auc_tick)

        ystart, yend = ax.get_ylim()
        ax.yaxis.set_ticks(np.arange(ystart, yend + 0.01, 0.05))

        plt.xticks(fontsize=30, rotation=90 * rotate_ccs)

        plt.xlabel(None)
        ax.tick_params(bottom=False, left=True)

        ax.yaxis.grid(color='gray', linestyle='dashed')
        ax.yaxis.set_zorder(3)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(6)  # change width
        ax.spines[axis].set_color('red')  # change color

    # Add hatches


#     patterns =('.', 'x', 'O','o','/','-', '+','O','o','\\','\\\\')
    bars = ax.patches

    hatches = [p for p in patterns for i in range(len(df))]
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    _ = ax.legend(loc='upper right', fontsize=22)
    return ax


def top_k_tables(group_acc_metric: M.CodeGroupTopAlarmAccuracy,
                 results: Dict[str, pd.DataFrame]):
    def styled_df(df):
        pd.set_option('precision', 3)
        data_sorted = -np.sort(-df.to_numpy(), axis=0)
        rank1_vals = data_sorted[0, :]
        rank2_vals = data_sorted[1, :]
        last1_vals = data_sorted[-1, :]
        last2_vals = data_sorted[-2, :]

        df_guide = df.copy()
        df_guide[:] = 0
        df_guide[df == rank1_vals] = 1
        df_guide[df == rank2_vals] = 0.5
        df_guide[df == last1_vals] = -1
        # df_guide[df == last2_vals] = -0.5
        s_df = df.style.background_gradient(axis=None,
                                            gmap=df_guide,
                                            cmap="PiYG")
        latex_str = s_df.to_latex(convert_css=True)
        for clf in df.index.tolist():
            latex_str = latex_str.replace(clf, f'\\texttt{{{clf}}}', 1)
        latex_str = latex_str.replace('_', '\\_')
        return s_df, latex_str

    groups_index = range(len(group_acc_metric.code_groups))
    k_list = group_acc_metric.top_k_list

    output = {}
    models = sorted(results.keys())
    for k in k_list:
        acc_val = lambda m, i: group_acc_metric.value_extractor(
            {
                'k': k,
                'group_index': i
            })(results[m])
        colname = lambda i: f'ACC-P{i}-k{k}'
        acc_dict = {colname(i): [] for i in groups_index}
        for i in groups_index:
            for model in models:
                acc_dict[colname(i)].append(acc_val(model, i))
        df_topk = pd.DataFrame(acc_dict, index=models)
        df_topk = df_topk.apply(lambda x: round(x, 3))
        s_df, ltx_s = styled_df(df_topk)

        output[k] = {'styled': s_df, 'latex': ltx_s, 'raw': df_topk}
    return output


def dx_label(label):
    return label if len(label) < 15 else label[:15] + ".."


def add_label(label, recorded_labels, group_priority=0):
    if group_priority not in recorded_labels:
        recorded_labels[group_priority] = set()
    if label in recorded_labels[group_priority]:
        return None
    else:
        recorded_labels[group_priority].add(label)
        return label


def plot_codes(codes_dict, outcome_color, legend_labels, idx2desc):

    for idx in codes_dict:
        desc = idx2desc[idx]
        time, traj_vals = zip(*codes_dict[idx])
        plt.scatter(time,
                    traj_vals,
                    s=300,
                    marker='^',
                    color=outcome_color[idx],
                    linewidths=2,
                    label=add_label(f'{label(desc)} (Diagnosis)',
                                    legend_labels, desc + 'd'))


def plot_risk_traj(trajs, outcome_color, legend_labels, idx2desc):
    for idx in trajs:
        desc = idx2desc[idx]
        time, traj_vals = zip(*trajs[idx])
        time = np.concatenate(time)
        traj_vals = np.concatenate(traj_vals)

        plt.plot(time,
                 traj_vals,
                 color=outcome_color[idx],
                 marker='o',
                 markersize=2,
                 linewidth=3,
                 label=add_label(f'{dx_label(desc)} (Predicted Risk)',
                                 legend_labels, desc + 'p'))


def plot_admission_lines(adms, legend_labels):
    adms, dischs = zip(*adms)
    for i, (adm_ti, disch_ti) in enumerate(zip(adms, dischs)):
        plt.fill_between([adm_ti, disch_ti], [1.0, 1.0],
                         alpha=0.3,
                         color='gray',
                         label=add_label('Hospital Stay', legend_labels, '0'))


def plot_trajectory(trajectories, interface, outcome_selection, outcome_color,
                    out_dir):

    style = {'axis_label_fs': 20, 'axis_ticks_fs': 18, 'legend_fs': 16}
    outcome = interface.dx_outcome
    index = outcome.index
    desc = outcome.desc
    idx2code = outcome.idx2code
    idx2desc = outcome.idx2desc

    outcome_selection = set(outcome_selection)

    for i, traj in list(trajectories.items()):
        adm_times = interface.adm_times(i)
        history = interface.dx_outcome_history(i)

        outcome_selection &= set(index[c] for c in history)
        if len(outcome_selection) == 0:
            continue

        time_segments = traj['t']
        plt_codes = defaultdict(list)
        plt_trajs = defaultdict(list)
        max_min = [-np.inf, np.inf]
        for ccs_idx in outcome_selection:
            risk = [r[:, ccs_idx] for r in traj['d']]
            code = idx2code[ccs_idx]
            code_history = history[code]
            code_history_adm, code_history_disch = zip(*code_history)

            # If diagnosis is made at the first discharge, then no point to
            # render the risk.
            if code_history_adm[0] == adm_times[0][0]:
                continue

            for time_segment, r, (adm_time_i,
                                  disch_time_i) in zip(time_segments, risk,
                                                       adm_times[1:]):
                max_min[0] = max(max_min[0], r.max())
                max_min[1] = min(max_min[1], r.min())
                plt_trajs[ccs_idx].append((time_segment, r))

                if disch_time_i in code_history_disch:
                    plt_codes[ccs_idx].append((disch_time_i, r[-1]))

        if len(plt_codes) == 0: continue

        plt.figure(i)

        legend_labels = {}
        plot_codes(codes_dict=plt_codes,
                   outcome_color=outcome_color,
                   legend_labels=legend_labels,
                   idx2desc=idx2desc)
        plot_risk_traj(trajs=plt_trajs,
                       outcome_color=outcome_color,
                       legend_labels=legend_labels,
                       idx2desc=idx2desc)

        # Make the major grid
        plt.grid(which='major', linestyle=':', color='gray', linewidth='1')
        # Turn on the minor ticks on
        # plt.minorticks_on()
        # Make the minor grid
        # plt.grid(which='minor', linestyle=':', color='black', linewidth='0.5')

        ystep = 0.1 if max_min[1] - max_min[0] > 0.2 else 0.05
        plt.ylim(
            math.floor(max_min[1] / ystep) * ystep,
            math.ceil(max_min[0] / ystep) * ystep)

        ystart, yend = plt.gca().get_ylim()
        plt.gca().yaxis.set_ticks(np.arange(ystart, yend + 0.01, ystep))

        plot_admission_lines(adm_times, legend_labels)
        plt.ylabel('Predicted Risk ($\widehat{v}(t)$)',
                   fontsize=style['axis_label_fs'],
                   labelpad=style['axis_label_fs'])
        plt.yticks(fontsize=style['axis_ticks_fs'])
        plt.xlabel('Days Since First Admission ($t$)',
                   fontsize=style['axis_label_fs'],
                   labelpad=style['axis_label_fs'])
        plt.xticks(fontsize=style['axis_ticks_fs'])

        labels = []
        for priority in sorted(legend_labels):
            labels.extend(legend_labels[priority])
        handles, _labels = plt.gca().get_legend_handles_labels()
        handles = dict(zip(_labels, handles))
        plt.legend(labels=labels,
                   handles=list(map(handles.get, labels)),
                   fontsize=style['legend_fs'],
                   loc='upper right',
                   bbox_to_anchor=(1, 1.5),
                   ncol=1)

        current_figure = plt.gcf()
        current_figure.savefig(f"{out_dir}/{i}.pdf", bbox_inches='tight')
