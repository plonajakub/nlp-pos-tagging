from scipy.stats import ttest_rel
import numpy as np
from tabulate import tabulate
import pandas as pd


def load_experiment(path, rnn_type):
    names = []
    scores_list = []
    df = pd.read_csv(path)
    for col_name in df:
        if col_name.startswith(rnn_type):
            names.append(col_name)
            scores_list.append(df[col_name].to_numpy())
    return names, scores_list


def ttest(names, scores):
    tstats = np.zeros(shape=(len(names), len(names)))
    pvalues = np.zeros(shape=(len(names), len(names)))

    for i in range(len(names)):
        for j in range(len(names)):
            tstats[i, j], pvalues[i, j] = ttest_rel(scores[i], scores[j])

    pvalues[np.isnan(pvalues)] = 1
    tstats[np.isnan(tstats)] = 0

    names_arr = np.array([[name] for name in names])
    pvalues_table = tabulate(np.concatenate((names_arr, pvalues), axis=1), names)
    tstats_table = tabulate(np.concatenate((names_arr, tstats), axis=1), names)

    print('pvalues: \n')
    print(pvalues_table)
    print('\n')

    print('tstatistics: \n')
    print(tstats_table)
    print('\n')

    return tstats, pvalues


def advantage(names, tstats):
    names_arr = np.array([[name] for name in names])
    adv = np.zeros((len(names), len(names)))
    adv[tstats > 0] = 1

    # adv_table = tabulate(np.concatenate((names_arr, adv), axis=1), names)
    # print('advantage table: \n')
    # print(adv_table)
    # print('\n')

    return adv


def statistical_sign(names, pvalues):
    names_arr = np.array([[name] for name in names])
    stat_sign = np.zeros((len(names), len(names)))
    stat_sign[pvalues < 0.05] = 1

    # stat_sign_table = tabulate(np.concatenate((names_arr, stat_sign), axis=1), names)
    # print('statistical significance: \n')
    # print(stat_sign_table)
    # print('\n')

    return stat_sign


def statistically_better(names, stat_sign, adv):
    names_arr = np.array([[name] for name in names])
    statistically_significant_better = np.logical_and(stat_sign, adv) * 1

    statistically_significant_better_table = tabulate(np.concatenate((names_arr,
                                                                      statistically_significant_better), axis=1), names)
    print('statistically significant better: \n')
    print(statistically_significant_better_table)
    print('\n')

    return statistically_significant_better


def run_analysis(path, rnn_type):
    # experiment 1
    names, scores = load_experiment(path, rnn_type=rnn_type)
    tstats, pvalues = ttest(names, scores)
    adv = advantage(names, tstats)
    stat_sign = statistical_sign(names, pvalues)
    ssb = statistically_better(names, stat_sign, adv)

    pvalues_df = pd.DataFrame(pvalues)
    pvalues_df.to_csv(f'results/experiment_{rnn_type}_pvalues.csv', float_format='%.3f')
    ssb_df = pd.DataFrame(ssb)
    ssb_df.to_csv(f'results/experiment_{rnn_type}_ssb.csv')
    # stat_sign_df = pd.DataFrame(stat_sign)
    # stat_sign_df.to_csv('results/experiment_rnn_stat_signs.csv')


def print_save_statistic():
    df = pd.read_csv('results/test_results_balanced_accuracy.csv')
    df_statistics = df.describe()
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df_statistics)
    df_statistics.to_csv('results/test_results_balanced_accuracy_statistics.csv', float_format='%.4f')


def main():
    path = 'results/test_results_balanced_accuracy.csv'
    print_save_statistic()
    run_analysis(path, 'rnn')
    run_analysis(path, 'lstm')
    run_analysis(path, 'bi_lstm')


if __name__ == '__main__':
    main()
