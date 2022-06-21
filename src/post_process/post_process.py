from data_structure import Adn
from mpc import Mpc
from post_process.performance_analyzer import *
import re
from os.path import join, exists
from os import walk
from front_end_utilities import write_df_result, load_df_result
import pandas as pd
from post_process.visualization import plot_putility_boxplot
from errors import InputError
# TODO:
#   - Standarize parameters for performance analyzer report
#       - nvv() : nvv_dt_interval, nvv_vr_ratio (obtain them from pdata insted of asking for
#       paramters)


def data_metrics(pdata, time_config=None):
    dict_report_yaml = {}
    df_data = pdata.df_data

    if time_config is None:
        dt = pd.to_timedelta(to_offset(pd.infer_freq(df_data.index)))
        if dt is None:
            InputError('pandas can''t infer freq from df_data')
    else:
        dt = time_config['dt']
        if dt != df_data.index[1] - df_data.index[0]:
            raise InputError('Incoherent time_config!')

        trange = pd.date_range(start=time_config['tini'], end=time_config['tend'] - dt, freq=dt)
        df_data = df_data.index[trange, :]

    l_cols_loadp = [i for i in df_data.columns if i.startswith(SNAM_LOADP)]
    l_cols_loadq = [i for i in df_data.columns if i.startswith(SNAM_LOADQ)]
    l_cols_dgpmax = [i for i in df_data.columns if i.startswith(SNAM_DGPMAX)]
    l_cols_loadevp = [i for i in df_data.columns if i.startswith(SNAM_LOADEVP)]
    l_cols_loaddgp = [i for i in df_data.columns if i.startswith(SNAM_LOADDGP)]

    ar_loadp = df_data.loc[:, l_cols_loadp].values
    ar_loadevp = df_data.loc[:, l_cols_loadevp].values
    ar_loaddgp = df_data.loc[:, l_cols_loaddgp].values
    ar_dgpmax = df_data.loc[:, l_cols_dgpmax].values
    ar_loadp_base = ar_loadp - ar_loadevp + ar_loaddgp

    ar_total_demand = ar_loadp + ar_loaddgp

    # Integrate to obtain the total energy of each component in mwh
    n_days = (df_data.index[-1] - df_data.index[0]) / timedelta(days=1)
    hours_per_dt = dt / timedelta(hours=1)
    factor_to_mwh = hours_per_dt * pdata.sbase_mva / n_days

    accum_loadp_mwh = ar_loadp.sum() * factor_to_mwh
    accum_loadp_base_mwh = ar_loadp_base.sum() * factor_to_mwh
    accum_loadevp_mwh = ar_loadevp.sum() * factor_to_mwh
    accum_loaddgp_mwh = ar_loaddgp.sum() * factor_to_mwh
    accum_dgpmax_mwh = ar_dgpmax.sum() * factor_to_mwh
    accum_total_demand_mwh = ar_total_demand.sum() * factor_to_mwh

    # Make yaml report
    dict_report_yaml['accum_loadp_mwh'] = float(accum_total_demand_mwh)
    dict_report_yaml['accum_loaddgp_mwh'] = float(accum_loaddgp_mwh)
    dict_report_yaml['accum_loadevp_mwh'] = float(accum_loadevp_mwh)
    dict_report_yaml['accum_dgpmax_mwh'] = float(accum_dgpmax_mwh)

    dict_report_yaml['tini_data'] = df_data.index[0].to_pydatetime()
    dict_report_yaml['tend_data'] = df_data.index[-1].to_pydatetime()
    dict_report_yaml['dt_data'] = dt.to_pytimedelta()

    return dict_report_yaml


def max_loads(df_data, time_config):
    dt = time_config['dt']
    if dt != df_data.index[1] - df_data.index[0]:
        raise InputError('Incoherent time_config!')

    trange = pd.date_range(start=time_config['tini'], end=time_config['tend'] - dt, freq=dt)
    df_data = df_data.index[trange, :]

    l_cols_loadp = [i for i in df_data.columns if i.startswith(SNAM_LOADP)]

    df_loadp = df_data.loc[:, l_cols_loadp].max()
    print(df_loadp)


def config_yaml(pdata: Adn):
    dict_general = pdata.grid_tables['general'].loc[:, 'value'].to_dict()
    dict_general.update(pdata.time_config)

    return dict_general


def write_report(pdata: Adn, df_sim: pd.DataFrame, mpc: Mpc, co='#',
                 nvv_dt_interval=timedelta(minutes=5), nvv_vr_ratio=0.05):

    # Data summary                                                                                 #
    dict_data_summary = data_metrics(pdata)

    # Pdata config                                                                                 #
    dict_config = config_yaml(pdata)

    # Mpc config                                                                                   #
    dict_mpc = {
        'mod_file_in': mpc.mod_file_in,
        'mod_file_sim': mpc.mod_file_sim,
        'mod_in_string': mpc.mod_in_string,
        'mod_sim_string': mpc.mod_sim_string
    }
    dict_config.update(dict_mpc)

    # Performance metrics                                                                          #
    dict_performance_metrics = se_make_metrics(pdata, df_sim, nvv_dt_interval, nvv_vr_ratio).to_dict()

    dict_config['nvv_dt'] = nvv_dt_interval
    dict_config['nvv_vr'] = nvv_vr_ratio

    # Construct master table                                                                       #
    dict_master_yaml = {
        'config': dict_config,
        'data_summary': dict_data_summary,
        'performance_metrics': dict_performance_metrics
    }

    return dict_master_yaml


MAP_REGEX2CONSTRUCTOR = {
    r'[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{1,2}:[0-9]{2}:[0-9]{2}': pd.Timestamp,
    r'[0-9]{1,2}:[0-9]{2}:[0-9]{2}': pd.Timedelta,
    r'-*[0-9]+\.[0-9]*': float,
    r'-*[0-9]+': int,
    r'(True|False)': lambda x: True if x == 'True' else False
}


def read_report(file_name):
    dict_ret = {}
    with open(file_name, 'r') as hfile:
        l_lines = hfile.readlines()

    value = None
    for i in l_lines:
        if (not i.startswith('#')) and (not i == '\n'):
            re_groups = re.search(r'(\w+)\s+(.*)', i)
            key = re_groups.group(1)
            value_str = re_groups.group(2)

            # Casting
            bool_casted = False
            for re_expr, constr in MAP_REGEX2CONSTRUCTOR.items():
                if re.match(re_expr, value_str) is not None:
                    value = constr(value_str)
                    bool_casted = True
                    break
            if not bool_casted:
                value = value_str

            # Insert into dict_ret
            dict_ret[key] = value

    return dict_ret


def write_results(folder_name, suffix, pdata, mpc, df_sim, df_ins=None):
    write_df_result(df_sim, join(folder_name, 'df_sim_' + suffix + '.csv'))
    if df_ins is not None:
        write_df_result(df_ins, join(folder_name, 'df_ins_' + suffix + '.csv'))
    with open(join(folder_name, 'report_' + suffix + '.txt'), 'w') as hfile:
        report_str = write_report(pdata, df_sim, mpc)
        hfile.write(report_str)


def load_results(folder_name, suffix):
    fn_df_sim = join(folder_name, 'df_sim' + suffix + '.csv')
    fn_df_ins = join(folder_name, 'df_ins' + suffix + '.csv')
    df_sim = load_df_result(fn_df_sim)
    if exists(fn_df_ins):
        df_ins = load_df_result(fn_df_ins)
        tup_ret = (df_sim, df_ins)
    else:
        tup_ret = (df_sim, None)

    return tup_ret


def benchmark_solutions(pdata: Adn, mpc: Mpc, dict_sols: dict):
    ...


def se_make_metrics(pdata: Adn, df_sim: pd.DataFrame, dt_nvv_interval=timedelta(minutes=5),
                    nvv_vr=0.05):
    l_buses = pdata.l_buses
    l_cols_v = [SNAM_V + str(i) for i in l_buses]
    n_hours = (df_sim.index[1] - df_sim.index[0])*df_sim.shape[0] / timedelta(hours=1)

    avg_losses_pu = losses_pu(pdata, df_sim).mean()
    avg_utility_injection_pu = utility_p_injection(pdata, df_sim).mean()
    avg_hnvv = nvv(df_sim, Dt_interval=dt_nvv_interval, Vr=nvv_vr) / n_hours
    avg_pv_curtailment = avg_pv_curtailment_pu(pdata, df_sim)
    v_max = df_sim.loc[:, l_cols_v].max().max()
    v_min = df_sim.loc[:, l_cols_v].min().min()
    avg_vv = calculate_avg_vv_pu(df_sim)

    return pd.Series({
        'avg_losses_pu': avg_losses_pu,
        'avg_utility_injection_pu': avg_utility_injection_pu,
        'avg_hnvv': avg_hnvv,
        'avg_pv_curtailment_pu': avg_pv_curtailment,
        'vmax': v_max,
        'vmin': v_min,
        'avg_vv': avg_vv
    }, dtype='float64')


def df_to_tabular(df_data: pd.DataFrame, index_label='index'):
    n_cols = df_data.shape[1]
    str_data = df_data.to_csv(sep='&', index_label=index_label, line_terminator='\\\\\n', float_format='%.4e') #float_format='%.4f'
    tabular_begin = '\\begin{tabular}{l ' + 'c ' * n_cols + '}\n'
    tabular_end = '\\end{tabular}'
    return tabular_begin + str_data[:-4] + '\n' + tabular_end


def putility_boxplot(pdata: Adn, df_sim: pd.DataFrame, folder_name=None, suffix=''):
    ar_utility_p_injection = utility_p_injection(pdata, df_sim)
    ar_hours = df_sim.index.hour
    fig, ax = plot_putility_boxplot(ar_utility_p_injection, ar_hours)
    fig.show()
    if folder_name is not None:
        fig.savefig(join(folder_name, 'p_utility_box_plot' + suffix + '.pdf'), format='pdf')


PREFIX_DF_INS = 'df_ins'
PREFIX_DF_SIM = 'df_sim'


def key_sort_idx_trange(x):
    return x.index[0]


def merge_solutions(l_ifol, ofol=None, suffix=None):
    l_fn_ins = []
    l_fn_sim = []
    for ifol in l_ifol:
        _, _, l_files = next(walk(ifol))
        l_local_ins = [join(ifol, i) for i in l_files if i.startswith(PREFIX_DF_INS)]
        l_local_sim = [join(ifol, i) for i in l_files if i.startswith(PREFIX_DF_SIM)]

        l_fn_ins = l_fn_ins + l_local_ins
        l_fn_sim = l_fn_sim + l_local_sim


    l_df_sim = [load_df_result(i) for i in l_fn_sim]
    l_df_ins = [load_df_result(i) for i in l_fn_ins]

    l_df_sim.sort(key=key_sort_idx_trange)
    l_df_ins.sort(key=key_sort_idx_trange)

    idx_total = l_df_sim[0].index
    for df_sim in l_df_sim[1:]:
        idx_total = idx_total.union(df_sim.index)

    df_sim_ret = pd.DataFrame(index=idx_total, columns=l_df_sim[0].columns)
    df_ins_ret = pd.DataFrame(index=idx_total, columns=l_df_ins[0].columns)

    for df_sim in l_df_sim[::-1]:
        df_sim_ret.loc[df_sim.index, :] = df_sim

    for df_ins in l_df_ins[::-1]:
        df_ins_ret.loc[df_ins.index, :] = df_ins

    if not (suffix is None or ofol is None):
        fn_sim = join(ofol, PREFIX_DF_SIM + '_' + suffix + '.csv')
        fn_ins = join(ofol, PREFIX_DF_INS + '_' + suffix + '.csv')
        write_df_result(df_sim_ret, fn_sim)
        write_df_result(df_ins_ret, fn_ins)

    return df_sim_ret, df_ins_ret


def test_ieee1547_curtailment(case_folder, case_name, fn_sim):
    EPS = 1e-6
    pdata = Adn()
    pdata.read(case_folder, case_name)
    df_sim = load_df_result(fn_sim)

    l_dgs = pdata.l_dgs
    l_cols_dgp = [SNAM_DGP + str(i) for i in l_dgs]
    l_cols_dgq = [SNAM_DGQ + str(i) for i in l_dgs]
    l_cols_dgpmax = [SNAM_DGPMAX + str(i) for i in l_dgs]

    ar_dgp = df_sim.loc[:, l_cols_dgp].values
    ar_dgq = df_sim.loc[:, l_cols_dgq].values
    ar_dgpmax = pdata.df_data.loc[df_sim.index, l_cols_dgpmax].values

    ar_dgp_snom = pdata.dgs.loc[:, ['snom']].values.transpose()

    ar_dgs = np.sqrt(ar_dgp ** 2 + ar_dgq ** 2)

    idx_curt = ar_dgpmax > ar_dgp + EPS

    ar_snom_diff = ar_dgp_snom - ar_dgs

    return ar_snom_diff[idx_curt].sum()






