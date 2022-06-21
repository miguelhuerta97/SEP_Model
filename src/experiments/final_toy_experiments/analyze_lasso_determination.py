from scipy.stats import ttest_ind
import re
import pandas as pd
import numpy as np


from os import mkdir, walk, makedirs
from os.path import join, exists, split

from global_definitions import *

from data_structure import Adn
from jpyaml import yaml
from post_process.post_process import df_to_tabular
from front_end_utilities import load_df_result

from post_process.visualization import plt, init_figure
from post_process.performance_analyzer import calculate_avg_vv_pu, calculate_cvar, calculate_vv_pu,\
    utility_p_injection, losses_pu




l_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

L_LOG_FOLDER_METRICS = ['avg_vv', 'avg_utility_injection_pu']
VLIM_R = 0.05
FLOAT_PATTERN = r'cx(.*).(?:csv|yaml)'
FLOAT_PATTERN_LOG = r'log_(.+)'
INT_PATTERN_LOGFILES = r'[a-z_]+([0-9]+)'
REPORT_PREFIX = 'report'
INSAMPLE_PREFIX = 'df_ins'
SIM_PREFIX = 'df_sim'
LOG_PREFIX = 'log'
LOG_SETPOINT_PREFIX = 'df_setpoint'
LOG_VARS_PREFIX = 'df_vars_insample'

KEY_METRICS = 'performance_metrics'

MAP_D_PREFIX2NAME = {
    SNAM_DLPP: r'$\mb{D}_{\mb{p_G,p_L}}$',
    SNAM_DLPQ: r'$\mb{D}_{\mb{p_G,q_L}}$',
    SNAM_DLQP: r'$\mb{D}_{\mb{q_G,p_L}}$',
    SNAM_DLQQ: r'$\mb{D}_{\mb{q_G,q_L}}$',
    SNAM_DPP: r'$\mb{D}_{\mb{p_G,\hat{p}_G}}$',
    SNAM_DPQ: r'$\mb{D}_{\mb{q_G,\hat{p}_G}}$',
}

MAP_DTYPE2COLOR = dict(zip(MAP_D_PREFIX2NAME.values(), l_colors[:len(MAP_D_PREFIX2NAME)]))


def read_report(fn_report):
    with open(fn_report, 'r') as hfile:
        dict_yaml = yaml.load(hfile, yaml.FullLoader)
    return dict_yaml


def key_sort(x: str):
    return float(re.search(FLOAT_PATTERN, x).group(1))


def key_sort_log_folders(x: str):
    return float(re.search(FLOAT_PATTERN_LOG, x).group(1))

def key_sort_log_innerfiles(x: str):
    return int(re.search(INT_PATTERN_LOGFILES, x).group(1))


def l_cols_D_from_df_insample(df_insample):
    l_cols_D = (
        [i for i in df_insample.columns if i.startswith(SNAM_DLPP)] +
        [i for i in df_insample.columns if i.startswith(SNAM_DLPQ)] +
        [i for i in df_insample.columns if i.startswith(SNAM_DLQP)] +
        [i for i in df_insample.columns if i.startswith(SNAM_DLQQ)] +
        [i for i in df_insample.columns if i.startswith(SNAM_DPP)] +
        [i for i in df_insample.columns if i.startswith(SNAM_DPQ)]
    )
    return l_cols_D


def calculate_fobj_metrics_ins(pdata, ifol, vlim_r=0.05, putility_only=False):
    E_TOL = 1e-6
    cost_vlim = pdata.cost_vlim
    cost_putility = pdata.cost_putility
    l_cols_dv = ['dv' + str(i) for i in pdata.l_buses]
    l_cols_duq = ['duq' + str(i) for i in pdata.l_buses]
    l_cols_dup = ['dup' + str(i) for i in pdata.l_buses]

    se_ret = pd.Series(index=L_LOG_FOLDER_METRICS, dtype='float64')
    _, _, l_filenames = next(walk(ifol))

    l_fn_vars = []
    l_fn_setpoint = []
    for fname in l_filenames:
        if fname.startswith(LOG_VARS_PREFIX):
            l_fn_vars.append(fname)
        elif fname.startswith(LOG_SETPOINT_PREFIX):
            l_fn_setpoint.append(fname)

    l_fn_vars.sort(key=key_sort_log_innerfiles)
    l_fn_setpoint.sort(key=key_sort_log_innerfiles)

    l_fn_vars = [join(ifol, i) for i in l_fn_vars]
    l_fn_setpoint = [join(ifol, i) for i in l_fn_setpoint]

    df_metrics = pd.DataFrame(index=range(len(l_fn_vars)),
                              columns=['fobj_expected', 'fobj_cvar_10', 'fobj_cvar_5', 'fobj_max'],
                              dtype='float64')


    for fn_vars, fn_setpoint, i in zip(l_fn_vars, l_fn_setpoint, range(len(l_fn_vars))):
        df_vars = pd.read_csv(fn_vars)
        df_setpoint = pd.read_csv(fn_setpoint)

        if i == 0:
            l_cols_e = [i for i in df_vars.columns if i.startswith('e_')]
        ar_losses = df_vars['losses'].values
        ar_pinjection = df_vars['pinjection'].values
        ar_v_set = df_setpoint['v_set'].values
        ar_dv = df_vars.loc[:, l_cols_dv].values
        ar_v = ar_v_set + ar_dv
        e_pos = np.maximum(0., ar_v - (1. + vlim_r))
        e_neg = - np.minimum(0., ar_v - (1. - vlim_r))
        ar_e = (e_pos + e_neg).sum(axis=1)

        ar_duq = df_vars.loc[:, l_cols_duq].values
        ar_uq_set = df_setpoint['uq_mean'].values
        ar_uq = ar_duq + ar_uq_set

        ar_dup = df_vars.loc[:, l_cols_dup].values
        ar_up_set = df_setpoint['up_mean'].values
        ar_up = ar_dup + ar_up_set

        if putility_only:
            ar_fobj2stage = cost_putility * (ar_losses - ar_pinjection)
        else:
            ar_fobj2stage = cost_putility * (ar_losses - ar_pinjection) + cost_vlim * ar_e

        ar_weights = df_vars['weights'].values

        df_metrics.loc[i, 'fobj_expected'] = ar_weights @ ar_fobj2stage
        df_metrics.loc[i, 'fobj_max'] = ar_fobj2stage.max()

        df_metrics.loc[i, 'fobj_cvar_5'] = calculate_cvar(ar_fobj2stage, ar_weights, gamma=0.95)
        df_metrics.loc[i, 'fobj_cvar_10'] = calculate_cvar(ar_fobj2stage, ar_weights, gamma=0.9)



    return df_metrics.mean()


LOG_FOLDER_PREFIX = 'log'


def make_df_lasso_in(ifol, case_name, case_folder, putility_only=False):

    pdata = Adn()
    pdata.read(case_folder, case_name)

    _, l_dirs, l_filenames = next(walk(ifol))

    l_fn_log_folders = [i for i in l_dirs if i.startswith(LOG_FOLDER_PREFIX)]
    l_fn_log_folders.sort(key=key_sort_log_folders)
    l_fn_log_folders = [join(ifol, i) for i in l_fn_log_folders]


    l_cols_fobj_metrics = ['fobj_expected', 'fobj_cvar_10', 'fobj_cvar_5', 'fobj_max']
    n = len(l_fn_log_folders)
    df_fobj_metrics_in = pd.DataFrame(
        index=range(n),
        columns=['cost_lasso_x'] + l_cols_fobj_metrics,
        dtype='float64')

    for fn_log_folder, i in zip(l_fn_log_folders, range(n)):
        df_fobj_metrics_in.loc[i, l_cols_fobj_metrics] =\
            calculate_fobj_metrics_ins(pdata, fn_log_folder, putility_only=putility_only)
        df_fobj_metrics_in.loc[i, 'cost_lasso_x'] = key_sort_log_folders(fn_log_folder)

    return df_fobj_metrics_in


def calculate_fobj_metrics_sim(pdata, df_sim, df_data=None, v0=complex(1., 0.),
                               vlim_r=0.05, putility_only=False):
    df_data = pdata.df_data.loc[df_sim.index, :] if df_data is None else df_data

    cost_vlim = pdata.cost_vlim
    cost_putility = pdata.cost_putility

    ar_vv_pu = calculate_vv_pu(df_sim, vlim_r)
    ar_putility_pu = utility_p_injection(pdata, df_sim, df_data, v0)
    if putility_only:
        ar_fobj2stage = cost_putility * ar_putility_pu
    else:
        ar_fobj2stage = cost_vlim * ar_vv_pu.sum(axis=1) + cost_putility * ar_putility_pu
    ar_weights = np.ones_like(ar_fobj2stage) / ar_fobj2stage.shape[0]

    l_cols_fobj_metrics = ['fobj_expected', 'fobj_cvar_10', 'fobj_cvar_5', 'fobj_max']
    se_fobj_metrics_sim = pd.Series(index=l_cols_fobj_metrics, dtype='float64')

    se_fobj_metrics_sim['fobj_expected'] = ar_fobj2stage.mean()
    se_fobj_metrics_sim['fobj_cvar_10'] = calculate_cvar(ar_fobj2stage, ar_weights, gamma=0.9)
    se_fobj_metrics_sim['fobj_cvar_5'] = calculate_cvar(ar_fobj2stage, ar_weights, gamma=0.95)
    se_fobj_metrics_sim['fobj_max'] = ar_fobj2stage.max()

    return se_fobj_metrics_sim


def make_df_lasso_sim(ifol, case_name, case_folder, putility_only=False):
    l_cols_fobj_metrics = ['fobj_expected', 'fobj_cvar_10', 'fobj_cvar_5', 'fobj_max']
    # Load pdata
    pdata = Adn()
    pdata.read(case_folder, case_name)

    _, _, l_files = next(walk(ifol))

    l_fn_sim = [join(ifol, i) for i in l_files if i.startswith('df_sim')]
    l_fn_sim.sort(key=key_sort)

    n = len(l_fn_sim)

    df_fobj_metrics_sim = pd.DataFrame(
        index=range(n), columns=['cost_lasso_x'] + l_cols_fobj_metrics, dtype='float64')

    for fn_sim, i in zip(l_fn_sim, range(n)):
        df_sim = load_df_result(fn_sim)
        df_fobj_metrics_sim.loc[i, l_cols_fobj_metrics] = calculate_fobj_metrics_sim(
            pdata, df_sim,putility_only=putility_only)
        df_fobj_metrics_sim.loc[i, 'cost_lasso_x'] = key_sort(fn_sim)

    return df_fobj_metrics_sim


def calculate_metrics_sim_past(ifol):
    _, _, l_filenames = next(walk(ifol))

    l_fn_sim = []
    l_fn_reports = []
    l_fn_log_folders = []
    for fname in l_filenames:
        if fname.startswith(REPORT_PREFIX):
            l_fn_reports.append(fname)
        elif fname.startswith(INSAMPLE_PREFIX):
            l_fn_insample.append(fname)
        elif fname.startswith(SIM_PREFIX):
            l_fn_sim.append(fname)
        elif fname.startswith(LOG_PREFIX):
            l_fn_log_folders.append(fname)

    l_fn_insample.sort(key=key_sort)
    l_fn_sim.sort(key=key_sort)
    l_fn_reports.sort(key=key_sort)
    l_fn_log_folders.sort(key=key_sort_log_folders)

    l_fn_reports = [join(ifol, i) for i in l_fn_reports]
    l_fn_insample = [join(ifol, i) for i in l_fn_insample]
    l_fn_sim = [join(ifol, i) for i in l_fn_sim]
    l_fn_log_folders = [join(ifol, i) for i in l_fn_log_folders]

    # Construct df_metrics
    dict_yaml = read_report(l_fn_reports[0])
    l_names_metrics = list(dict_yaml[KEY_METRICS].keys())
    l_cols_df_metrics = ['cost_lasso_D', 'avg_vv_sim', 'avg_vv_ins'] + l_names_metrics
    df_metrics = pd.DataFrame(columns=l_cols_df_metrics, dtype='float64')

    for fn_report, count in zip(l_fn_reports, range(len(l_fn_reports))):
        dict_yaml = read_report(fn_report)
        df_metrics.loc[count, 'cost_lasso_D'] = dict_yaml['config']['cost_lasso_x']
        df_metrics.loc[count, l_names_metrics] = dict_yaml[KEY_METRICS]

    df_metrics.sort_values('cost_lasso_D', inplace=True)


    # Calculate avg_vv_sim
    for fn_sim, count in zip(l_fn_sim, range(len(l_fn_sim))):
        df_sim = load_df_result(fn_sim)
        avg_vv_sim = calculate_avg_vv_pu(df_sim, vlim_r=VLIM_R)
        df_metrics.loc[count, 'avg_vv_sim'] = avg_vv_sim

    return df_metrics, df_insample_means


def __get_degree_from_D_col(colname):
    return int(re.search(r'[A-Za-z]+_[0-9]+_([0-9]+)', colname).group(1))


def plot_colored(df_insample_means, ofol= None, suffix=None):
    # Initialize figure
    fig_degree, axs_degree = init_figure(0.5, 0.3)
    fig_degree_accum, axs_degree_accum = init_figure(0.5, 0.3)
    fig_type, axs_type = init_figure(0.5, 0.3)
    fig_type_accum, axs_type_accum = init_figure(0.5, 0.3)

    fig_accum, axs_accum = init_figure(0.5, 0.3)

    config_plot = {'lw': 0.9}

    l_cols_D = l_cols_D_from_df_insample(df_insample_means)

    # Construct map (type_name) to (list of column names)
    map_by_typename = dict(zip(MAP_D_PREFIX2NAME.values(), [[] for i in MAP_D_PREFIX2NAME]))

    for colname in l_cols_D:
        prefix = re.search(r'([a-zA-z]+)_', colname).group(1)
        type_name = MAP_D_PREFIX2NAME[prefix]
        map_by_typename[type_name].append(colname)

    # Plot lambda decay by type name:
    for type_name, l_cols in map_by_typename.items():
        color = MAP_DTYPE2COLOR[type_name]

        axs_type_accum.plot(
            df_insample_means['cost_lasso_D'],
            df_insample_means.loc[:, l_cols].abs().mean(axis=1).values,
            label=type_name, **config_plot
        )

        for col in l_cols:
            axs_type.plot(
                df_insample_means['cost_lasso_D'].values, abs(df_insample_means[col].values),
                color=color, **config_plot)

    # Construct map (degree) to (list of column names)
    l_random_value = next(iter(map_by_typename.values()))
    set_degrees = set([__get_degree_from_D_col(i) for i in l_random_value])
    map_by_degree = {i: [] for i in set_degrees}

    for colname in l_cols_D:
        degree = __get_degree_from_D_col(colname)
        map_by_degree[degree].append(colname)

    for degree, l_cols in map_by_degree.items():
        axs_degree_accum.plot(
            df_insample_means['cost_lasso_D'].values,
            df_insample_means.loc[:, l_cols].abs().mean(axis=1).values,
            label='Degree {}'.format(degree), **config_plot
        )

        color = l_colors[degree]
        for col in l_cols:
            axs_degree.plot(
                df_insample_means['cost_lasso_D'].values, abs(df_insample_means[col].values),
                color=color, **config_plot)

    # Construct total mean plot
    axs_accum.plot(
        df_insample_means['cost_lasso_D'].values,
        df_insample_means.loc[:, l_cols_D].abs().mean(axis=1).values,
        **config_plot
    )

    # Configure figure final  delivery
    axs_type: plt.Axes
    fig_type: plt.Figure
    axs_degree: plt.Axes
    fig_degree: plt.Figure
    fig_type_accum: plt.Figure

    l_axs = [axs_type_accum, axs_degree_accum, axs_accum]

    axs_type.set_yscale('log')
    axs_type.set_xscale('log')
    axs_degree.set_yscale('log')
    axs_degree.set_xscale('log')
    axs_type_accum.set_xscale('log')
    axs_type_accum.set_yscale('log')
    axs_degree_accum.set_yscale('log')
    axs_degree_accum.set_xscale('log')
    axs_accum.set_yscale('log')
    axs_accum.set_xscale('log')

    for ax in l_axs:
        ax.set_xlabel(r'$\lambda_D$')
        ax.set_ylabel(r'Mean $|D_{ij}|$')

    axs_accum.set_ylabel(r'$|\mb{D}|_1$')

    fig_type_accum.legend(fontsize='x-small', loc='upper right', bbox_to_anchor=(0.95, 0.95))
    fig_degree_accum.legend(fontsize='x-small', loc='upper right', bbox_to_anchor=(0.95, 0.95))

    fig_type.tight_layout()
    fig_degree.tight_layout()
    fig_type_accum.tight_layout()
    fig_degree_accum.tight_layout()
    fig_accum.tight_layout()

    fig_type.show()
    fig_degree.show()
    fig_type_accum.show()
    fig_degree_accum.show()
    fig_accum.show()

    if (ofol is not None) and (suffix is not None):
        if not exists(ofol):
            mkdir(ofol)
        fig_type.savefig(join(ofol, 'fig_lasso_type_' + suffix + '.pdf'), format='pdf')
        fig_degree.savefig(join(ofol, 'fig_lasso_degree_' + suffix + '.pdf'), format='pdf')
        fig_type_accum.savefig(join(ofol, 'fig_lasso_type_accum_' + suffix + '.pdf'), format='pdf')
        fig_degree_accum.savefig(join(ofol, 'fig_lasso_degree_accum_' + suffix + '.pdf'),
                                 format='pdf')
        fig_accum.savefig(join(ofol, 'fig_lasso_accum_' + suffix + '.pdf'), format='pdf')


MAP_METRICS_COL2LATEX = zip(
    ['fobj_expected', 'fobj_cvar_10', 'fobj_cvar_5', 'fobj_max'],
    [r'Average operational costs (\$)',
     r'CVaR$_{10}$ of operational costs (\$)', r'CVaR$_{5}$ of operational costs (\$)',
     r'Max operational costs (\$)']
)


def make_lorca_plots_and_tables(ifol, ofol, case_name, case_folder, putility_only=False,
                                from_tables=False, idx_range=None):

    if not exists(ofol):
        makedirs(ofol)

    if from_tables:
        fn_sim = join(ifol, 'df_fobj_metrics_sim.csv')
        fn_ins = join(ifol, 'df_fobj_metrics_in.csv')
        df_fobj_metrics_in = pd.read_csv(fn_ins, index_col='index')
        df_fobj_metrics_sim = pd.read_csv(fn_sim, index_col='index')
    else:
        df_fobj_metrics_in = make_df_lasso_in(ifol, case_name, case_folder, putility_only=putility_only)
        df_fobj_metrics_sim = make_df_lasso_sim(ifol, case_name, case_folder, putility_only=putility_only)

        # Dump tables
        fn_fobj_metrics_in = join(ofol, 'df_fobj_metrics_in.csv')
        fn_fobj_metrics_sim = join(ofol, 'df_fobj_metrics_sim.csv')
        df_fobj_metrics_in.to_csv(fn_fobj_metrics_in, index_label='index')
        df_fobj_metrics_sim.to_csv(fn_fobj_metrics_sim, index_label='index')

    if idx_range is not None:
        df_fobj_metrics_in = df_fobj_metrics_in.loc[idx_range, :]
        df_fobj_metrics_sim = df_fobj_metrics_sim.loc[idx_range, :]

    # Dump in plots
    for col_name, label_name in MAP_METRICS_COL2LATEX:
        fig, axs = init_figure(0.5*1.1*1.1, 0.3*1.0*1.1)
        axs.plot(df_fobj_metrics_in.loc[:, 'cost_lasso_x'].values,
                 df_fobj_metrics_in.loc[:, col_name].values, label='In-sample', ls='--', lw=0.6)
        axs.plot(df_fobj_metrics_sim.loc[:, 'cost_lasso_x'].values,
                 df_fobj_metrics_sim.loc[:, col_name].values, label='Out-of-sample',
                 lw=0.6)

        axs.set_xlabel(r'$\lambda$')
        axs.set_ylabel(label_name)
        axs.set_xscale('log')
        axs.set_xlim([10**-13.2, 10**-2.8])
        axs.set_ylim([0.15, 1.1])
        axs.set_xticks([10**-13, 10**-8 ,10**-3])
        if col_name == 'fobj_expected':
            # Plot minimum
            idx_min = df_fobj_metrics_sim[col_name].idxmin()

            axs.plot([10**-13, df_fobj_metrics_sim.loc[idx_min, 'cost_lasso_x']],
                     [df_fobj_metrics_sim.loc[0, col_name], df_fobj_metrics_sim.loc[idx_min, col_name]], marker='s', markersize=2.,
                     markerfacecolor='k', color='k')

            #props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

            textstr = '$\lambda = ${:.1e}\n $h$ ={:.3f}'.format(df_fobj_metrics_sim.loc[idx_min, 'cost_lasso_x'], df_fobj_metrics_sim.loc[idx_min, col_name])
            # place a text box in upper left in axes coords
            axs.text(0.05, 0.95, textstr, transform=axs.transAxes,
                    verticalalignment='top')

            textstr = '$\lambda = ${}\n $h$ ={:.3f}'.format(df_fobj_metrics_sim.loc[0, 'cost_lasso_x'], df_fobj_metrics_sim.loc[0, col_name])
            # place a text box in upper left in axes coords
            axs.text(0.05, 0.5, textstr, transform=axs.transAxes,
                     verticalalignment='top')
        # axs.set_yscale('log')
        axs.legend()
        fn_plot = join(ofol, 'plot_' + col_name + '.svg')
        fig.tight_layout()
        fig.savefig(fn_plot, format='svg')


def main():
    base_folder = '/home/jp/tesis/experiments/ieee4bus_lasso/08_01'
    exp_name = 'rob_kme'
    IFOL = join(
        base_folder,
        exp_name
    )

    df_metrics, df_insample_means = load_relevant_data(IFOL)
    print(df_metrics.to_string())
    df_metrics['avg_fobj_cont'] = 1.e5 * df_metrics['avg_vv_sim'] + 1. * df_metrics['avg_utility_injection_pu']

    ar_fobj = df_metrics['avg_fobj_cont'].values

    ofol_plots = join(base_folder, 'plots_lasso_' + exp_name)
    plot_colored(df_insample_means, ofol=ofol_plots, suffix=exp_name)

    ar_lambda = df_metrics.loc[:, 'cost_lasso_D'].values

    print(df_metrics.to_string())
    map_metrics_rename = {
        'cost_lasso_D': r'$\lambda_D$',
        'avg_hnvv': r'Avg. HNNV',
        'avg_losses_pu': r'Avg. Losses pu',
        'avg_pv_curtailment_pu': r'Avg. PV Curtailment pu',
        'avg_utility_injection_pu': r'Avg. Utility injection pu',
        'vmax': r'V max pu',
        'vmin': r'V min pu'
    }
    df_metrics.rename(columns=map_metrics_rename, inplace=True)
    print(df_metrics.to_string())
    print(df_to_tabular(df_metrics))


    fig, axs = init_figure(0.5, 0.3)
    axs.plot(ar_lambda, ar_fobj)
    axs.set_xscale('log')
    axs.set_xlabel('$\lambda_{D}$')
    axs.set_ylabel(r'Avg. Operational Cost')

    fig.tight_layout()
    fig.show()
    fig.savefig(join(ofol_plots, 'fig_fobj_' + exp_name + '.pdf'), format='pdf')


def main2():
    #fobj2stage_metrics_from_logfolder('/home/jp/tesis/experiments/ieee4bus_lasso/07_31/rob_kme/'
    #                                  'log_1e-16')

    IFOL = '/home/jp/tesis/experiments/ieee4bus_lasso/08_01/rob_kme/'
    CASE_NAME = 'ieee4bus'
    CASE_FOLDER = '/home/jp/tesis/experiments/toy_lasso/ieee4bus'
    df_fobj_metrics_in = make_df_lasso_in(IFOL, CASE_NAME, CASE_FOLDER)
    df_fobj_metrics_sim = make_df_lasso_sim(IFOL, CASE_NAME, CASE_FOLDER)

    print(df_fobj_metrics_in)
    print(df_fobj_metrics_sim)



    fig, axs = init_figure(0.5, 0.3)
    ar_lambda = df_fobj_metrics_in['cost_lasso_x'].values

    l_cols_fobj_metrics = ['fobj_expected', 'fobj_cvar_10', 'fobj_cvar_5', 'fobj_max']

    map_names = dict(zip(['fobj_expected', 'fobj_cvar_10', 'fobj_cvar_5', 'fobj_max'],
                         ['$E(\cdot)$', '$\text{CVaR}_{10}(\cdot)$', '$\text{CVaR}_{5}(\cdot)$', 'max(\cdot)']))

    for i in l_cols_fobj_metrics:
        name = map_names[i]
        ar_values = df_fobj_metrics_in[i].values
        axs.plot(ar_lambda, ar_values, label=name)

    #axs.set_yscale('log')
    axs.set_xscale('log')
    fig.tight_layout()
    fig.show()


def plot_lorca_custom(ofol, ifol, idx_range=None):
    fn_sim = join(ifol, 'df_fobj_metrics_sim.csv')
    fn_ins = join(ifol, 'df_fobj_metrics_in.csv')

    df_fobj_metrics_sim = pd.read_csv(fn_sim, index_col='index')
    df_fobj_metrics_in = pd.read_csv(fn_ins, index_col='index')

    idx_range = df_fobj_metrics_sim.index if idx_range is None else idx_range

    # Dump in plots
    for col_name, label_name in MAP_METRICS_COL2LATEX:
        fig, axs = init_figure(0.5*1.3*1.1, 0.3*1.1*1.1)
        #axs.plot(df_fobj_metrics_in.loc[idx_range, 'cost_lasso_x'].values,
        #         df_fobj_metrics_in.loc[idx_range, col_name].values, label='In-sample', ls='--')
        axs.plot(df_fobj_metrics_sim.loc[idx_range, 'cost_lasso_x'].values,
                 df_fobj_metrics_sim.loc[idx_range, col_name].values, label='Out-of-sample')

        axs.set_xlabel(r'$\lambda$')
        axs.set_ylabel(label_name)
        axs.set_xscale('log')

        if col_name == 'fobj_expected':
            axs.set_xticks(np.logspace(start=np.log10(1e-15), stop=np.log10(1e0), num=4))
            #axs.set_yticks(np.linspace(start=0.2, stop=1., num=5))
            axs.tick_params(direction='in')
            axs: plt.Axes
            axins = axs.inset_axes([0.55, 0.15, 0.4, 0.35])
            axins.plot(df_fobj_metrics_sim.loc[idx_range[:6], 'cost_lasso_x'].values,
                 df_fobj_metrics_sim.loc[idx_range[:6], col_name].values)
            axins.set_xscale('log')
            #axins.set_xticks([1e-15, 1e-13, 1e-11])
            #axins.set_xlim([0.7e-15, 1.3e-11])
            axins.tick_params(direction='in')
            #for axis in ['top', 'bottom', 'left', 'right']:
            #    axins.spines[axis].set_linewidth(0.5)
            #axs.indicate_inset_zoom(axins, edgecolor="black", lw=0.4, linewidth=0.4)
            #axs.set_xlim([1e-15, 1e0])
            #axs.set_ylim([0.2, 1.0])

        #axs.set_yscale('log')
        # axs.legend()
        fn_plot = join(ofol, 'plot_' + col_name + '.svg')
        fig.tight_layout()
        fig.savefig(fn_plot, format='svg')
        fig.show()


def main3():
    #IFOL = '/home/jp/tesis/experiments/ieee4bus_lasso/08_04/rob_kme/'
    #OFOL = '/home/jp/tesis/experiments/ieee4bus_lasso/08_04/rob_kme_lorca_plots'
    #CASE_NAME = 'ieee4bus'
    #CASE_FOLDER = '/home/jp/tesis/experiments/toy_lasso/ieee4bus'
    #IFOL = '/home/jp/tesis/experiments/34bus_lasso/08_16/rob_kme'
    #OFOL = '/home/jp/tesis/experiments/34bus_lasso/08_16/rob_kme_lorca_plots/only_putility'
    #CASE_FOLDER = '/home/jp/tesis/experiments/scaled_cases/ieee34bus_ev0.1_pv1.0_scaled_ordered_noB'

    CASE_NAME = 'ieee34bus'
    CASE_FOLDER = '/home/jp/tesis/experiments/cases_final/ieee34bus'

    IFOL = '/home/jp/tesis/experiments/final/lasso_ieee34bus/table_and_plots'
    OFOL = '/home/jp/tesis/experiments/final/lasso_ieee34bus/table_and_plots'

    make_lorca_plots_and_tables(IFOL, OFOL, CASE_NAME, CASE_FOLDER, from_tables=True,
                                idx_range=range(15))




def main_ttest():
    CASE_NAME = 'ieee34bus'
    CASE_FOLDER = '/home/jp/tesis/experiments/cases_final/ieee34bus'
    IFOL = '/home/jp/tesis/experiments/final/lasso_ieee34bus/'
    #OFOL = '/home/jp/tesis/experiments/final/lasso_ieee34bus/table_and_plots'

    pdata = Adn()
    pdata.read(CASE_FOLDER, CASE_NAME)

    fn_sim_9 = 'df_sim_ieee34bus_rob_kme_cx1e-09.csv'
    fn_sim_0 = 'df_sim_ieee34bus_rob_kme_cx0.0.csv'

    fn_sim_0 = join(IFOL, fn_sim_0)
    fn_sim_9 = join(IFOL, fn_sim_9)

    df_sim_0 = load_df_result(fn_sim_0)
    df_sim_9 = load_df_result(fn_sim_9)

    ar_fobj_0 = calculate_ar_fobj(pdata, df_sim_0)
    ar_fobj_9 = calculate_ar_fobj(pdata, df_sim_9)

    print(ttest_ind(ar_fobj_0, ar_fobj_9))




def calculate_ar_fobj(pdata, df_sim, vlim_r=0.05, v0=complex(1., 0.), cost_vlim=1.e5, cost_putility=1.):
    ar_vv_pu = calculate_vv_pu(df_sim, vlim_r).sum(axis=1)
    ar_putility_pu = utility_p_injection(pdata, df_sim, v0=v0)
    ar_losses_pu = losses_pu(pdata, df_sim, v0).flatten()
    ar_fobj2stage = cost_vlim * ar_vv_pu + cost_putility * ar_putility_pu
    return ar_fobj2stage




def analyze_zero_norm(pdata, ifol):
    _, _, l_filenames = next(walk(ifol))
    l_fn_ins = [i for i in l_filenames if i.startswith(INSAMPLE_PREFIX)]
    l_fn_ins.sort(key=key_sort)
    l_fn_ins = [join(ifol, i) for i in l_fn_ins]

    l_cols_D = None
    n_deg = pdata.polypol_deg
    ar_dg_buses = pdata.dgs['bus'].to_numpy()
    ar_nodg_buses = np.asarray(list(set(pdata.l_buses) - set(ar_dg_buses)), dtype='int64')
    ar_degs = np.asarray(range(0, n_deg + 1), dtype='int64')

    L_D_NAMES = [SNAM_DLPP, SNAM_DLPQ, SNAM_DLQP, SNAM_DLQQ, SNAM_DPP, SNAM_DPQ]
    FLOAT_EPS = 1e-12

    ar_cols_D_nonzero = ['{}_{}_{}'.format(k, i, j) for k in L_D_NAMES for i in ar_dg_buses
                         for j in ar_degs]
    ar_cols_D_zero = ['{}_{}_{}'.format(k, i, j) for k in L_D_NAMES for i in ar_nodg_buses
                      for j in ar_degs]

    df_nnz = pd.DataFrame(index=range(len(l_fn_ins)), columns=['Lambda Lasso', 'Nonzero elements', 'norm1'])
    count = 0
    for fn_ins in l_fn_ins:

        df_ins = load_df_result(fn_ins)
        if l_cols_D is None:
            l_cols_D = l_cols_D_from_df_insample(df_ins)

        df_nnz.loc[count, 'Nonzero elements'] = int(
            (df_ins.loc[:, ar_cols_D_nonzero].abs() > FLOAT_EPS).sum(axis=1).mean())
        df_nnz.loc[count, 'norm1'] = df_ins.loc[:, ar_cols_D_nonzero].abs().sum(axis=1).mean()

        df_nnz.loc[count, 'Lambda Lasso'] = key_sort(fn_ins)
        count += 1

    print('caca')

    fig, axs = init_figure(0.5, 0.3)
    axs.plot(df_nnz.loc[:, 'Lambda Lasso'], df_nnz.loc[:, 'Nonzero elements'])

    axs.set_xscale('log')
    axs.set_xlabel('$\lambda_{D}$')
    axs.set_ylabel('$||\mathbf{D}||_0$')
    fig.tight_layout()
    fig.show()

    fig, axs = init_figure(0.5, 0.3)
    axs.plot(df_nnz.loc[:, 'Lambda Lasso'], df_nnz.loc[:, 'norm1'])

    axs.set_xscale('log')
    axs.set_yscale('log')
    axs.set_xlabel('$\lambda_{D}$')
    axs.set_ylabel('$||\mathbf{D}||_1$')
    fig.tight_layout()
    fig.show()


def main4():
    CASE_FOLDER = '/home/jp/tesis/experiments/scaled_cases/ieee34bus_ev0.1_pv1.0_scaled_ordered_noB'
    CASE_NAME = 'ieee34bus'
    IFOL = '/home/jp/tesis/experiments/34bus_lasso/08_12_after_meeting/rob_kme'

    pdata = Adn()
    pdata.read(CASE_FOLDER, CASE_NAME)
    analyze_zero_norm(pdata, IFOL)


def main5():
    IFOL = '/home/jp/tesis/experiments/final/lasso_ieee34bus'
    OFOL = '/home/jp/tesis/experiments/final/lasso_ieee34bus/plots_and_tables'
    if not exists(OFOL):
        mkdir(OFOL)
    plot_lorca_custom(OFOL, IFOL)


def print_variables(fn_vars, dict_vars):
    with open(fn_vars, 'w') as hfile:
        for k, v in dict_vars.items():
            hfile.write('\\newcommand{{\{}}}{{{}}}'.format(k, v))
            hfile.write('\n')


def ar_lambda_from_df_sim(ifol):
    _, _, l_filenames = next(walk(ifol))

    l_lambda = [key_sort(i) for i in l_filenames if i.startswith('df_sim')]
    l_lambda.sort()
    ar_lambda = np.asarray(l_lambda)

    print('n = {}'.format(ar_lambda.shape[0]))
    for i in ar_lambda:
        print(i)



def main6():
    IFOL = '/home/jp/tesis/latex/paper_adn/vars/adalasso_vars.tex'
    DICT_VARS = {
        'adaNDaysOut': 2,
        'adaNDaysIn': 15,
        'adaNSce': 500,
        'adaNCtrlMins': 15,
        'locDGsTbus': '8, 9, 10, 11, 13, 15, 17, 18, 19, 21, 22, 24, 26, 28, 30, 31, 32 and 33'
    }
    DICT_VARS['adaNCtrlTunIns'] = DICT_VARS['adaNDaysOut'] * 24 * 4
    print_variables(IFOL, DICT_VARS)


def main7():
    IFOL = '/home/jp/tesis/experiments/34bus_lasso/08_16/rob_kme'
    ar_lambda_from_df_sim(IFOL)


if __name__ == '__main__':
    main3()
