from data_structure import Adn
from datetime import timedelta
import pandas as pd
from os.path import join, exists
from os import mkdir, walk
from jpyaml import yaml
from post_process.performance_analyzer import calculate_v_outofbounds
from front_end_utilities import load_df_result, write_df_result
from post_process.visualization import init_figure, plt
from post_process.post_process import df_to_tabular, merge_solutions
from post_process.performance_analyzer import calculate_avg_vv_pu, calculate_cvar, calculate_vv_pu,\
    utility_p_injection, losses_pu, calculate_pv_curtailment_ar_pu
import numpy as np
import matplotlib.pyplot as plt

l_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

linestyle_str = [
     ('solid', 'solid'),      # Same as (0, ()) or '-'
     ('dotted', 'dotted'),    # Same as (0, (1, 1)) or '.'
     ('dashed', 'dashed'),    # Same as '--'
     ('dashdot', 'dashdot')]  # Same as '-.'

linestyle_tuple = [
     ('dotted',                (0, (1, 1))),
     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]


# exp_name fn_report fn_df_sim fn_df_ins


def calculate_avg_cost_vlim(df_sim, cost_vlim):
    df_v_out = calculate_v_outofbounds(df_sim)
    avg_v_out = df_v_out.mean().mean()
    return cost_vlim * avg_v_out


def make_df_metrics(ifol, dict_cases, dict_strategies):
    l_strategies = list(dict_strategies.keys())
    l_cases = list(dict_cases.keys())
    idx_cols = pd.MultiIndex.from_product([l_cases, l_strategies])
    df_metrics = pd.DataFrame()


def make_df_metrics():
    # Inout hardcoded
    CASE_FOLDER, CASE_NAME = ('/home/jp/tesis/experiments/toy_lasso/toy1bus', 'toy1bus')
    IFOL = '/home/jp/tesis/experiments/ieee4bus_lasso/bilinear_benchmark_sce100_10days_rnd'
    OFOL = '/home/jp/tesis/experiments/toy_benchmark/report_bilinear_benchmark_sce100_10days'

    # Run report
    df_metrics = None
    l_metrics = None
    first_time = True
    l_exp_names = ['noctrl', 'ieee1547']
    cost_vlim = 500.
    for exp_name in l_exp_names:
        exp_folder = join(IFOL, exp_name)
        fn_df_sim = join(exp_folder, 'df_sim_' + exp_name + '.csv')
        fn_df_ins = join(exp_folder, 'df_ins_' + exp_name + '.csv')
        df_sim = load_df_result(fn_df_sim)
        try:
            df_ins = load_df_result(fn_df_ins)
        except FileNotFoundError:
            pass

        fn_report_general = join(exp_folder, 'report_general_' + exp_name + '.yaml')

        with open(fn_report_general, 'r') as hfile:
            dict_yaml = yaml.load(hfile, yaml.FullLoader)

        if first_time:
            l_metrics = list(dict_yaml['performance_metrics'].keys())
            df_metrics = pd.DataFrame(index=l_metrics + ['avg_cost_fobj'], columns=l_exp_names, dtype='float64')
            first_time = False

        # Analyse
         #dict_yaml['config']['cost_vlim']
        cost_putility = dict_yaml['config']['cost_putility']
        avg_cost_vlim = cost_vlim * dict_yaml['performance_metrics']['avg_hnvv'] #calculate_avg_cost_vlim(df_sim, cost_vlim)
        avg_cost_putility = cost_putility * dict_yaml['performance_metrics']['avg_utility_injection_pu']
        avg_cost_fobj = avg_cost_vlim + avg_cost_putility
        df_metrics.loc[l_metrics, exp_name] = dict_yaml['performance_metrics']
        df_metrics.loc['avg_cost_fobj', exp_name] = avg_cost_fobj

    exp_name = 'bilinear_droop'
    fn_report_general = '/home/jp/tesis/experiments/ieee4bus_lasso/bilinear_droop_sce100_3days/report_ieee4bus_rob_kme/report_general_ieee4bus_rob_kme.yaml'
    # fn_report_general = '/home/jp/tesis/experiments/toy_lasso/bilinear_droop_sce100_10days/report_toy1bus_sto_kme/report_general_toy1bus_sto_kme.yaml'
    fn_df_sim = '/home/jp/tesis/experiments/toy_lasso/bilinear_droop_sce100_10days/report_toy1bus_sto_kme/df_sim_toy1bus_sto_kme.csv'
    with open(fn_report_general, 'r') as hfile:
        dict_yaml = yaml.load(hfile, yaml.FullLoader)

    cost_putility = dict_yaml['config']['cost_putility']
    avg_cost_vlim = cost_vlim * dict_yaml['performance_metrics']['avg_hnvv'] # calculate_avg_cost_vlim(df_sim, cost_vlim)
    avg_cost_putility = cost_putility * dict_yaml['performance_metrics']['avg_utility_injection_pu']
    avg_cost_fobj = avg_cost_vlim + avg_cost_putility
    df_metrics.loc[l_metrics, exp_name] = dict_yaml['performance_metrics']
    df_metrics.loc['avg_cost_fobj', exp_name] = avg_cost_fobj


    exp_name = 'drooppolypol'
    fn_report_general = '/home/jp/tesis/experiments/ieee4bus_lasso/bilinear_lassodet_sce100_10days_ipopt_cvlim/rob_kme/report_general_ieee4bus_rob_kme_cx2.2360679774997894e-05.yaml'
    fn_df_sim = '/home/jp/tesis/experiments/toy_lasso/bilinear_lassodet_sce100_10days_ipopt/rob_kme/df_sim_toy1bus_rob_kme_cx4.1601676461038125e-06.csv'
    with open(fn_report_general, 'r') as hfile:
        dict_yaml = yaml.load(hfile, yaml.FullLoader)

    df_sim = load_df_result(fn_df_sim)

    cost_putility = dict_yaml['config']['cost_putility']
    avg_cost_vlim = cost_vlim * dict_yaml['performance_metrics']['avg_hnvv'] # calculate_avg_cost_vlim(df_sim, cost_vlim)
    avg_cost_putility = cost_putility * dict_yaml['performance_metrics']['avg_utility_injection_pu']
    avg_cost_fobj = avg_cost_vlim + avg_cost_putility
    df_metrics.loc[l_metrics, exp_name] = dict_yaml['performance_metrics']
    df_metrics.loc['avg_cost_fobj', exp_name] = avg_cost_fobj


    return df_metrics


def main0():
    df_metrics = make_df_metrics()
    df_metrics.sort_values('avg_cost_fobj', axis=1, inplace=True, ascending=False)
    print(df_metrics)
    print(df_to_tabular(df_metrics))

L_MARKERS = ['s', 'o', '.', ',', 'x', '+', 'v', '^', '<', '>', 'd']

L_STRATEGIES = ['noctrl', 'ieee1547', 'proposed', 'droop', 'kyri', 'linkfailure', 'proposed15min']
L_NAMES_STRATEGIES = ['SOPF', 'IEEE1547', 'Proposed', 'D=0', 'NCognizant', 'Link-failure', 'Proposed ($T=15$ min)']
MN_STRATEGIES = dict(zip(L_STRATEGIES, L_NAMES_STRATEGIES))

L_CASES = ['ieee4bus', 'ieee34bus', 'ieee123bus']
L_NAMES_CASES = ['IEEE 4-bus test feeder', 'IEEE 34-bus test feeder', 'IEEE 123-bus test feeder']
MN_CASES = dict(zip(L_CASES, L_NAMES_CASES))

L_METRICS = ['avg_vv', 'putility', 'pvcurt', 'losses']
L_NAMES_METRICS = ['AVV', 'HUPI', 'HCPVP', 'HNL']
MN_METRICS = dict(zip(L_METRICS, L_NAMES_METRICS))

DT_HOUR = timedelta(hours=1)


def df_metric_to_latex(ifol, dict_case_folder, vlim_r=0.05, v0=complex(1., 0.), ofol_ext=None):
    idx_cols = pd.MultiIndex.from_product([L_CASES, L_METRICS])
    df_metrics = pd.DataFrame(index=L_STRATEGIES, columns=idx_cols, dtype='float64')

    l_fol_cases = [join(ifol, i) for i in L_CASES]
    for case_name, folder_exp_case in zip(L_CASES, l_fol_cases):
        if not case_name in dict_case_folder:
            continue
        case_folder = dict_case_folder[case_name]
        pdata = Adn()
        if not exists(case_folder):
            continue
        pdata.read(case_folder, case_name)
        n_buses = pdata.buses.shape[0]
        sbase_mva = pdata.sbase_mva
        l_fol_strategies = [join(folder_exp_case, i) for i in L_STRATEGIES]
        for strategy, fol_strategy in zip(L_STRATEGIES, l_fol_strategies):
            if exists(fol_strategy):
                fn_sim = join(fol_strategy, 'df_sim_' + strategy + '.csv')
                if exists(fn_sim):
                    df_sim = load_df_result(fn_sim)

                    ar_vv_pu = calculate_vv_pu(df_sim, vlim_r).sum(axis=1)
                    ar_putility_pu = utility_p_injection(pdata, df_sim, v0=v0)
                    ar_losses_pu = losses_pu(pdata, df_sim, v0).flatten()
                    ar_pv_curtailment = calculate_pv_curtailment_ar_pu(pdata, df_sim)

                    if ofol_ext is not None:
                        df_metrics_extended = pd.DataFrame(index=df_sim.index, columns=L_METRICS,
                                                           dtype='float64')

                        df_metrics_extended.loc[:, 'avg_vv'] = ar_vv_pu
                        df_metrics_extended.loc[:, 'putility'] = ar_putility_pu
                        df_metrics_extended.loc[:, 'pvcurt'] = ar_pv_curtailment
                        df_metrics_extended.loc[:, 'losses'] = ar_losses_pu

                        fn_df_metrics_extended = join(
                            ofol_ext, 'df_mext_{}_{}.csv'.format(case_name, strategy))
                        write_df_result(df_metrics_extended, fn_df_metrics_extended)

                    dt_sim = df_sim.index[1] - df_sim.index[0]
                    n_dt_per_hour = int(DT_HOUR / dt_sim)
                    n_samples = ar_vv_pu.shape[0]
                    assert n_samples % n_dt_per_hour == 0
                    n_rows = n_samples // n_dt_per_hour

                    h_avg_vv = ar_vv_pu.mean()
                    h_avg_putility_pu = ar_putility_pu.mean()
                    h_avg_losses_pu = ar_losses_pu.mean()
                    h_avg_pv_curtailment_pu = ar_pv_curtailment.mean()

                    """
                    h_avg_vv = ar_vv_pu.reshape(
                        (n_rows, n_dt_per_hour)).sum(axis=1).mean() / n_dt_per_hour
                    h_avg_putility_pu = ar_putility_pu.reshape(
                        (n_rows, n_dt_per_hour)).sum(axis=1).mean() / n_dt_per_hour
                    h_avg_losses_pu = ar_losses_pu.reshape(
                        (n_rows, n_dt_per_hour)).sum(axis=1).mean() / n_dt_per_hour
                    h_avg_pv_curtailment_pu = ar_pv_curtailment.reshape(
                        (n_rows, n_dt_per_hour)).sum(axis=1).mean() / n_dt_per_hour
                    """

                    df_metrics.loc[strategy, (case_name, 'avg_vv')] = h_avg_vv * 1e7 / n_buses
                    df_metrics.loc[strategy, (case_name, 'putility')] = h_avg_putility_pu * sbase_mva * 1000.
                    df_metrics.loc[strategy, (case_name, 'losses')] = h_avg_losses_pu * sbase_mva * 1000.
                    df_metrics.loc[strategy, (case_name, 'pvcurt')] = h_avg_pv_curtailment_pu * sbase_mva * 1000.

    df_metrics.rename(columns=MN_CASES, index=MN_STRATEGIES, inplace=True)
    df_metrics.rename(columns=MN_METRICS, inplace=True)
    df_metrics.index.name = 'Strategies'

    str_latex = df_metrics.to_latex(float_format='%.1f')

    return str_latex


def make_pdistribution_plots(ifol, l_cases, l_strategies, cost_vlim=1.e5, cost_putility=1.):
    N_SUBSAMPLES = 200
    ar_fobj_subsampled = np.zeros((N_SUBSAMPLES,))
    for case_name in l_cases:
        fig, ax = init_figure(0.5 *1.1*1.1, 0.2 *1.1*1.1*1.05)
        count = 0
        for strategy in l_strategies:
            fn_metrics_ext = join(ifol, 'df_mext_' + case_name + '_' + strategy + '.csv')
            df_metrics_ext = load_df_result(fn_metrics_ext)
            ar_vv_pu = df_metrics_ext.loc[:, 'avg_vv'].values
            ar_putility_pu = df_metrics_ext.loc[:, 'putility'].values
            ar_fobj2stage = cost_vlim * ar_vv_pu + cost_putility * ar_putility_pu
            # Subsample
            n_samples = int(ar_fobj2stage.shape[0] * 0.1428)
            for i in range(N_SUBSAMPLES):
                ar_fobj_subsampled[i] = (
                    np.random.choice(ar_fobj2stage, size=n_samples, replace=False).mean())

            # Calculate objective function
            ar_p_density, ar_bin_lims = np.histogram(ar_fobj_subsampled, bins=ar_fobj_subsampled.shape[0],
                                                     density=True)

            ar_p_dist_y = (ar_p_density * (ar_bin_lims[1:] - ar_bin_lims[:-1])).cumsum()
            ar_p_dist_x = ar_bin_lims[1:]

            ar_p_dist_y = np.concatenate([np.asarray([0.]), ar_p_dist_y])
            ar_p_dist_x = np.concatenate([np.asarray([ar_p_dist_x[0]]), ar_p_dist_x])

            ar_p_dist_y = 100 * ar_p_dist_y

            # Plot pdistribution
            ax.plot(ar_p_dist_x, ar_p_dist_y, label=MN_STRATEGIES[strategy], color=l_colors[count])
            count += 1

        ax.tick_params(direction='in')
        ax.set_ylim([0, 100])
        #ax.set_xscale('log')
        #ax.set_xlim([0, 6])
        ax.set_xlabel(r'Out-of-sample operational costs (pu)')
        ax.set_ylabel(r'Probability distribution (\%)')
        fig.tight_layout()
        fig.legend(fontsize=7, ncol=3)

        fig.tight_layout()
        fig.show()
        fn_fig = join(ifol, 'plotpdist_{}.svg'.format(case_name))
        fig.savefig(fn_fig, format='svg')

        print('caca')


def main2():
    IFOL = '/home/jp/tesis/experiments/final'
    #IFOL = '/home/jp/tesis/tests/merged_ieee34bus'
    DICT_CASE_FOLDER = {
        'ieee34bus': '/home/jp/tesis/experiments/cases_final/ieee34bus',
        'ieee4bus': '/home/jp/tesis/experiments/cases_final/ieee4bus'
    }
    OFOL = join(IFOL, 'plots')
    if not exists(OFOL):
        mkdir(OFOL)
    #print(df_metric_to_latex(IFOL, DICT_CASE_FOLDER, ofol_ext=OFOL))
    make_pdistribution_plots(OFOL, l_cases=['ieee34bus'],
                             l_strategies=['noctrl', 'kyri', 'proposed']) # 'ieee1547' 'noctrl', 'kyri' 'proposed' 'noctrl', 'kyri', 'proposed', 'linkfailure'


def main3():
    l_strategies = ['noctrl', 'ieee1547', 'proposed', 'droop']
    l_names_strategies = ['STO OPF', 'IEEE1547', 'Proposed', 'D=0']
    mn_strategies = dict(zip(l_strategies, l_names_strategies))

    l_cases = ['ieee4bus', 'ieee34bus', 'ieee123bus']
    l_names_cases = ['IEEE 4-bus test feeder', 'IEEE 34-bus test feeder', 'IEEE 123-bus test feeder']
    mn_cases = dict(zip(l_cases, l_names_cases))

    l_metrics = ['avg_vv', 'putility', 'pvcurt', 'losses']
    l_names_metrics = ['HVV', 'HUPI', 'HCPVP', 'HNL']
    mn_metrics = dict(zip(l_metrics, l_names_metrics))

    n_strategies = len(l_strategies)
    n_cases = len(l_cases)
    n_metrics = len(l_metrics)

    n_cols = n_strategies * n_cases
    n_rows = n_metrics

    ar_metrics = np.random.random((n_rows, n_cols))

    idx_cols = pd.MultiIndex.from_product([l_cases, l_metrics])
    df_metrics = pd.DataFrame(ar_metrics, index=l_strategies, columns=idx_cols)

    df_metrics.rename(index=mn_strategies, columns=mn_cases, inplace=True)
    df_metrics.rename(columns=mn_metrics, inplace=True)

    df_metrics.index.name = 'Strategies'

    print(df_metrics.to_latex(float_format='%.3f'))


def merge_results_list(l_fn_sim):
    l_df_sim = []

    for fn_sim in l_fn_sim:
        l_df_sim.append(load_df_result(fn_sim))

    l_df_sim.sort(key=lambda x: x.index[0])

    idx_total = l_df_sim[0].index
    for df_sim in l_df_sim[1:]:
        idx_total = idx_total.union(df_sim.index)

    df_sim_ret = pd.DataFrame(index=idx_total, columns=l_df_sim[0].columns)

    for df_sim in l_df_sim[::-1]:
        df_sim_ret.loc[df_sim.index, :] = df_sim

    return df_sim_ret





if __name__ == '__main__':
    main2()
