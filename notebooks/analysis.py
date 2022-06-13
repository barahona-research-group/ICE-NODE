"""."""

import glob
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(['.', '..'])
from icenode.metric.common_metrics import codes_auc_pairwise_tests
from icenode.metric.common_metrics import evaluation_table

import common as C


def performance_traces(data_tag, clfs, train_dir, model_dir):
    clfs_params_dir = train_dir[data_tag]
    traces_df = []
    for clf in clfs:
        clf_dir = model_dir[clf]
        csv_files = sorted(
            glob.glob(f'{clfs_params_dir}/{clf_dir}/*.csv', recursive=False))
        dfs = [pd.read_csv(csv_file, index_col=[0]) for csv_file in csv_files]
        df = pd.concat([df[["VAL"]].transpose() for df in dfs]).reset_index()
        df['comp'] = df[list(f'ACC-P{i}' for i in range(5))].prod(axis=1)
        df = df[['loss', 'accuracy', 'MICRO-AUC', 'comp']]

        df = pd.melt(df.reset_index(),
                     value_vars=df.columns,
                     id_vars='index',
                     var_name='metric',
                     value_name='value')
        df['loss'] = df['metric'] == 'loss'
        df['clf'] = clf
        traces_df.append(df)

    traces_df = pd.concat(traces_df)
    return traces_df


def get_trained_models(clfs, train_dir, model_dir, data_tag, criterion, comp):
    params = {}
    config = {}
    df = {}
    clfs_params_dir = train_dir[data_tag]

    best_iter, best_val = [], []
    for clf in clfs:
        clf_dir = model_dir[clf]
        csv_files = sorted(
            glob.glob(f'{clfs_params_dir}/{clf_dir}/*.csv', recursive=False))
        dfs = [pd.read_csv(csv_file, index_col=[0]) for csv_file in csv_files]

        if callable(criterion):
            max_i = comp(range(len(dfs)),
                         key=lambda i: criterion(dfs[i].loc[:, 'VAL']))
            best_val.append(criterion(dfs[max_i].loc[:, 'VAL']))
        else:
            max_i = comp(range(len(dfs)),
                         key=lambda i: dfs[i].loc[criterion, 'VAL'])
            best_val.append(dfs[max_i].loc[criterion, "VAL"])

        best_iter.append(max_i)
        df[clf] = dfs[max_i]
        csv_file = csv_files[max_i]
        prefix = csv_file.split('_')
        prefix[-1] = 'params.pickle'
        params_file = '_'.join(prefix)
        params[clf] = C.load_params(params_file)
        config[clf] = C.load_config(f'{clfs_params_dir}/{clf_dir}/config.json')

    summary = pd.DataFrame({
        'Clf': clfs,
        'Best_i': best_iter,
        str(criterion): best_val
    })
    return {
        'config': config,
        'params': params,
        'evaluation': df,
        'summary': summary
    }


def test_eval_table(dfs, metric):
    data = {}
    for clf, df in dfs.items():
        data[clf] = df.loc[metric, "TST"].tolist()
    return pd.DataFrame(data=data, index=metric).transpose()


def relative_performance_upset(auc_tests, selected_clfs, pvalue, min_auc):
    flatccs_idx2code = {
        idx: code
        for code, idx in C.ccs_dag.dx_flatccs_idx.items()
    }
    # flatccs_frequency_train = patient_interface.diag_flatccs_frequency(
    # train_ids)

    idx2desc = lambda i: C.ccs_dag.dx_flatccs_desc[flatccs_idx2code[i]]
    auc_tests['DESC'] = auc_tests['CODE_INDEX'].apply(idx2desc)

    # remove codes that no classifier has scored above `min_auc`
    accepted_aucs = auc_tests.loc[:, [f'AUC({clf})'
                                      for clf in selected_clfs]].max(
                                          axis=1) > min_auc
    accepted_auc_tests = auc_tests[accepted_aucs]
    print(
        f'{len(accepted_auc_tests)} codes predicted an AUC higher than {min_auc} by at least one model.'
    )

    test_cols = [col for col in auc_tests.columns if col[:2] == 'P0']

    # exclude tests with nans
    accepted_auc_tests = accepted_auc_tests[
        accepted_auc_tests.loc[:, test_cols].isnull().max(axis=1) == 0]

    print(
        f'{len(accepted_auc_tests)} codes predicted an AUC higher than {min_auc} by at least one model, with valid tests.'
    )

    tests = accepted_auc_tests

    # Codes when no significant difference of AUCs among all pairs of models.
    common_perf = tests[tests.loc[:, test_cols].min(axis=1) > pvalue]

    auc_sets = defaultdict(set)
    clfs = tuple(sorted(selected_clfs))
    auc_sets[clfs] = set(common_perf.CODE_INDEX)
    competing_tests = tests.drop(index=common_perf.index)

    clfs_pairs = make_clf_paris(clfs)

    # Assign each code to the best model (max AUC), then assign it as well
    # to any model with no significant difference with the best.
    for index, row in competing_tests.iterrows():
        max_auc_clf = max(clfs, key=lambda clf: row[f'AUC({clf})'])
        insignificant_diff = {(clf1, clf2): f'P0(AUC_{clf1}==AUC_{clf2})' for (clf1, clf2) in clfs_pairs \
                          if max_auc_clf in (clf1, clf2) and row[f'P0(AUC_{clf1}==AUC_{clf2})'] > pvalue}

        # Case 1: The best model is significantly outperforming all others.
        if len(insignificant_diff) == 0:
            auc_sets[max_auc_clf].add(int(row['CODE_INDEX']))
        # Case 2: Some insigificant difference with others though.
        else:
            for (clf1, clf2), test_col in insignificant_diff.items():
                # Populate the intersections.
                auc_sets[(clf1, clf2)].add(int(row['CODE_INDEX']))

    # Prepare for using Upset plot -> Set Layout (passed to `from_contents`)
    content_sets = {}
    for clf in clfs:
        content_sets[clf] = auc_sets[clf] | auc_sets[clfs]
        for clf1, clf2 in clfs_pairs:
            if clf in (clf1, clf2):
                content_sets[clf].update(auc_sets[(clf1, clf2)])

    # Prepare for using Upset plot -> DataFrame Layout (passed to `from_indicators`)
    code_index = tests.CODE_INDEX.tolist()
    competence_assignments = {}
    for clf in clfs:
        competence_assignments[clf] = [
            c in content_sets[clf] for c in code_index
        ]
    indicator_df = pd.DataFrame(competence_assignments, index=code_index)

    # Descriptive statistics for each code.
    avg_aucs, n_codes = [], []
    for c in code_index:
        competent_clfs = [clf for clf in clfs if indicator_df.loc[c, clf]]
        avg_auc = tests.loc[c,
                            list(f'AUC({clf})'
                                 for clf in competent_clfs)].mean()
        avg_aucs.append(avg_auc)
        # n_codes.append(flatccs_frequency_train[c])
    data = pd.DataFrame(
        {'Avg. AUC': avg_aucs
         # '#codes (train)': n_codes
         },
        index=code_index)
    return content_sets, indicator_df, data, common_perf, competing_tests


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


def make_clf_paris(clfs):
    clfs_pairs = []
    for i in range(len(clfs)):
        for j in range(i + 1, len(clfs)):
            clfs_pairs.append((clfs[i], clfs[j]))
    return tuple(sorted(clfs_pairs))
