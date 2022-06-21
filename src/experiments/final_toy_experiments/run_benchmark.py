import numpy as np
import pandas as pd
from os.path import exists, join
from os import mkdir
from data_structure import Adn
from mpc import Mpc
from datetime import datetime, timedelta
from errors import InputError
from post_process.post_process import write_report
from front_end_utilities import write_df_result
from jpyaml import yaml


def make_exp_suffix(case_name, exp_name):
    return case_name + '_' + exp_name


def write_report_single_experiment(folder_report: str, name: str, mpc, df_sim: pd.DataFrame,
                                   df_ins: pd.DataFrame = None):
    pdata = mpc.pdata
    if pdata is None:
        raise InputError('pdata is None!')
    if not exists(folder_report):
        mkdir(folder_report)
    # Write df_sim
    fn_df_sim = join(folder_report, 'df_sim_' + name + '.csv')
    write_df_result(df_sim, fn_df_sim)

    # Write df_ins
    if df_ins is not None:
        fn_df_ins = join(folder_report, 'df_ins_' + name + '.csv')
        write_df_result(df_ins, fn_df_ins)

    # Write outofsample performance report
    fn_report_general = join(folder_report, 'report_general_' + name + '.yaml')
    dict_general_report = write_report(pdata, df_sim, mpc)
    with open(fn_report_general, 'w') as hfile:
        hfile.write(yaml.dump(dict_general_report))


def run_benchmark_others(l_cases_tuples, ofol, n_sce, n_win_ahead, n_win_before,
                            n_days, n_days_delay, time_config=None):
    if not exists(ofol):
        mkdir(ofol)

    for case_folder, case_name in l_cases_tuples:
        # ---------------------------------------------------------------------------------------- #
        # Read case
        # ---------------------------------------------------------------------------------------- #

        pdata = Adn()
        pdata.read(case_folder, case_name)
        if time_config is not None:
            pdata.time_config = time_config

        # General config
        n_dgs = pdata.dgs.shape[0]
        map_dgpmax = dict(zip(pdata.dgs.index.to_list(), range(n_dgs)))
        pdata.dgs.reset_index(inplace=True, drop=True)
        pdata.branches.loc[:, 'b'] = 0.



        pdata.set('ctrl_robust_n_sce', n_sce)
        pdata.set('scegen_type', 'ddus_kmeans')
        pdata.set('cost_vlim', 1e5)
        pdata.set('cost_putility', 1.)

        # Validation
        pdata.validate()

        """
        # ---------------------------------------------------------------------------------------- #
        # Run perfect OPF                                                                          #
        # ---------------------------------------------------------------------------------------- #
        # Set config
        exp_name = 'perfect_opf'
        exp_folder = join(ofol, exp_name)
        if not exists(exp_folder):
            mkdir(exp_folder)

        # Run experiment
        mpc = Mpc(pdata)
        df_sim = mpc.run_acopf()

        # Write report
        write_report_single_experiment(exp_folder, exp_name, mpc, df_sim)
        """

        # ---------------------------------------------------------------------------------------- #
        # Run noctrl                                                                               #
        # ---------------------------------------------------------------------------------------- #

        # Set config
        exp_name = 'noctrl'
        exp_folder = join(ofol, exp_name)
        if not exists(exp_folder):
            mkdir(exp_folder)

        pdata.set('scegen_type', 'all')
        pdata.set('log_folder', join(exp_folder, 'log'))
        pdata.set('risk_measure', 'expected_value')
        pdata.set('scegen_n_win_ahead', n_win_ahead)
        pdata.set('scegen_n_win_before', n_win_before)
        pdata.set('scegen_n_days', n_days)
        pdata.set('scegen_n_days_delay', n_days_delay)

        # Run experiment
        mpc = Mpc(pdata)
        df_sim, df_ins = mpc.run_noctrl_new()

        # Write report
        write_report_single_experiment(exp_folder, exp_name, mpc, df_sim, df_ins)
        """
        # ---------------------------------------------------------------------------------------- #
        # Run IEEE-1547                                                                            #
        # ---------------------------------------------------------------------------------------- #
        # Set config
        exp_name = 'ieee1547'
        exp_folder = join(ofol, exp_name)
        if not exists(exp_folder):
            mkdir(exp_folder)

        # Run experiment
        mpc = Mpc(pdata)
        df_sim = mpc.run_ieee_1547()

        # Write report
        write_report_single_experiment(exp_folder, exp_name, mpc, df_sim)
        """

def main():
    # Input hardcoded
    OFOL = '/home/jp/tesis/experiments/34bus_lasso/benchmark_08_10_scaled_v4'
    # L_CASES = [('/home/jp/tesis/experiments/toy_lasso/toy1bus', 'toy1bus')]
    # L_CASES = [('/home/jp/tesis/experiments/toy_lasso/ieee4bus', 'ieee4bus')]
    L_CASES = [('/home/jp/tesis/experiments/scaled_cases/ieee34bus_ev0.1_pv1.0_scaled_ordered_noB', 'ieee34bus')]

    time_config = {
        'tini': datetime(2018, 7, 27, 0, 0, 0),
        'tiniout': datetime(2018, 8, 11, 0, 0, 0),
        'tend': datetime(2018, 8, 11, 23, 59, 54),
        'dt': timedelta(seconds=6),
        'n_rh': 150
    }

    # Run
    run_benchmark_others(L_CASES, OFOL, n_sce=150, n_win_ahead=0, n_win_before=0,
                         n_days=0, n_days_delay=0, time_config=time_config)


if __name__ == '__main__':
    main()