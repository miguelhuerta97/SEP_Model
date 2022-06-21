"""
Data characterization
    Paper report
        1. Report
        2. Report data metrics
        3. Typical day profile visualization: loadp and dgpmax
"""
import sys
from datetime import timedelta, datetime
import yaml
from pandas.tseries.frequencies import to_offset
from post_process.visualization import plot_df_data_day
import pandas as pd

from data_structure import Adn
from os.path import dirname, join, exists
from errors import InputError
from global_definitions import *
from os import mkdir

"""
L_CASENAMES = ['ieee123bus', 'ieee34bus']
THIS_DIR = dirname(__file__)
OFOL = join(THIS_DIR, 'data_report')
if not exists(OFOL):
    mkdir(OFOL)

# Hardcode shit
L_FOLDERS = [join(THIS_DIR, i) for i in L_CASENAMES]

DIC_CASES = dict(zip(L_CASENAMES, L_FOLDERS))
"""


def timedelta_representer(dumper, data):
    return dumper.represent_scalar(u'!dt', u'{} {}'.format(data.days, data.seconds))


def timedelta_constructor(loader, node):
    value = loader.construct_scalar(node)
    days, seconds = map(float, value.split[' '])
    return timedelta(days=days, seconds=seconds)


def list_to_text(l_list):
    str_ret = ', '.join(list(map(str, l_list[:-1])))
    str_ret += ' and {}'.format(l_list[-1])
    return str_ret


def def_command_line(name: str, body: str):
    return '\\newcommand{{\{name}}}{{{body}}}'.format(name=name, body=body)


def dg_location_report(pdata, case_name):
    if pdata.dgs.empty:
        raise InputError('pdata.dgs is empty')
    dg_location_name = '{}_dglist'.format(case_name)

    l_buses_dgs = pdata.dgs.loc[:, 'bus'].to_list()
    str_dg_list = list_to_text(l_buses_dgs)
    return def_command_line(dg_location_name, str_dg_list)


def data_metrics(pdata):
    dict_report_yaml = {}
    df_data = pdata.df_data
    dt = pd.to_timedelta(to_offset(pd.infer_freq(df_data.index)))
    if dt is None:
        InputError('pandas can''t infer freq from df_data')

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

    dict_report_yaml['tini'] = df_data.index[0].to_pydatetime()
    dict_report_yaml['tend'] = df_data.index[-1].to_pydatetime()
    dict_report_yaml['dt'] = dt.to_pytimedelta()

    return dict_report_yaml


def main():
    # Default input
    case_folder = ('/home/jp/Data/Dropbox/tesis/experiments/scaled_cases/'
                   'ieee34bus_ev0.1_pv1.0_scaled')
    case_name = 'ieee34bus'

    # Config YAML parser
    yaml.add_representer(timedelta, timedelta_representer)
    yaml.add_constructor(u'!dt', timedelta_constructor)

    ofol = join(case_folder, 'report')
    if not exists(ofol):
        mkdir(ofol)

    # Load pdata
    pdata = Adn()
    pdata.read(case_folder, case_name)

    # DG location
    str_latex = dg_location_report(pdata, case_name)
    fn_report_dg_location = join(ofol, 'dg_location.txt')
    with open(fn_report_dg_location, 'w') as hfile:
        hfile.write(str_latex)

    # Data metrics
    dic_report_metrics = data_metrics(pdata)
    fn_report_metrics = join(ofol, 'metrics.yaml')
    with open(fn_report_metrics, 'w') as hfile:
        yaml.dump(dic_report_metrics, hfile, default_style=None)


if __name__ == '__main__':
    main()


"""
IFOL = '/home/jp/Data/Dropbox/tesis/tests/case_generation_scaled'
CASE_NAME_OUT = 'ieee34bus_ev0.1_pv1.0_scaled'
L_FOLDERS = [join(IFOL, CASE_NAME_OUT)]  # [join(IFOL, CASE_NAME_OUT + '_scaled')]
L_CASENAMES = ['ieee34bus']
DIC_CASES = dict(zip(L_CASENAMES, L_FOLDERS))
OFOL = join(IFOL, 'data_report_' + CASE_NAME_OUT)
if not exists(OFOL):
    mkdir(OFOL)


def list_to_text(l_list):
    str_ret = ', '.join(list(map(str, l_list[:-1])))
    str_ret += ' and {}'.format(l_list[-1])
    return str_ret


def def_command_line(name: str, body: str):
    return '\\newcommand{{\{name}}}{{{body}}}'.format(name=name, body=body)


l_report = []
map_table_accum_index = {
    'accum_loadp_mwh': 'Total demand (MWh)',
    'accum_loadevp_mwh': 'Electric vehicles demand (MWh)',
    'accum_loaddgp_mwh': 'Uncontrolled power generation (MWh)',
    'accum_dgpmax_mwh': 'Available DG power (MWh)'
}

df_table_accum = pd.DataFrame(index=map_table_accum_index.values(), columns=L_CASENAMES,
                              dtype='float64')
dict_report_yaml = {}

for case_name, case_folder in DIC_CASES.items():
    pdata = Adn()
    pdata.read(case_folder, case_name)

    if pdata.dgs.empty:
        raise InputError('pdata.dgs is empty')

    df_data = pdata.df_data
    dt = pd.to_timedelta(to_offset(pd.infer_freq(df_data.index)))
    if dt is None:
        InputError('pandas can''t infer freq from df_data')

    # Report dg location
    dg_location_name = '{}_dglist'.format(case_name)

    l_buses_dgs = pdata.dgs.loc[:, 'bus'].to_list()
    str_dg_list = list_to_text(l_buses_dgs)
    l_report.append(def_command_line(dg_location_name, str_dg_list))

    ################################################################################################
    # 2. Report data metrics
    ################################################################################################
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

    # Make table_accum
    df_table_accum.loc[map_table_accum_index['accum_loadp_mwh'], case_name] = accum_total_demand_mwh
    df_table_accum.loc[map_table_accum_index['accum_loaddgp_mwh'], case_name] = accum_loaddgp_mwh
    df_table_accum.loc[map_table_accum_index['accum_loadevp_mwh'], case_name] = accum_loadevp_mwh
    df_table_accum.loc[map_table_accum_index['accum_dgpmax_mwh'], case_name] = accum_dgpmax_mwh

    # Make Yaml report


    for k, v in pdata.time_config.items():
        dict_report_yaml[k] = v
    ################################################################################################
    # 3. Typical day profile visualization: loadp and dgpmax                                       #
    ################################################################################################
    # Typical day sample selection (manual)
    idx_day = 3
    idx_loadp = 4
    idx_dgpmax = 4
    tini = df_data.index[0] + timedelta(days=idx_day)
    tend = tini + timedelta(days=1) - dt
    trange_plot = pd.date_range(start=tini, end=tend, freq=dt)
    ar_loadp_sample = df_data.loc[trange_plot, l_cols_loadp[idx_loadp]]
    ar_dgpmax_sample = df_data.loc[trange_plot, l_cols_dgpmax[idx_dgpmax]]
    label_load = 'Uncontrolled net power consumption'
    label_dgpmax = 'Available solar power'
    # Typical day plot
    fig, _ = plot_df_data_day(trange_plot, ar_loadp_sample, ar_dgpmax_sample, label_load, label_dgpmax)
    fig.savefig(join(OFOL, 'typical_day.pdf'), format='pdf')


with open(join(OFOL, 'doc_variables.dat'), 'w') as hfile:
    for i in l_report:
        hfile.write(i + '\n')

df_table_accum.to_csv(join(OFOL, 'data_summary.dat'), sep=',', index_label='Metric')


"""