import gc

import pandas as pd
from jpyaml import yaml
from data_structure import Adn
from errors import InputError
from mpc import Mpc
from os.path import join, exists
from os import mkdir
from front_end_utilities import write_df_result
from post_process.post_process import write_report
from datetime import datetime, timedelta


def write_report_single_experiment(mpc, df_sim: pd.DataFrame, df_ins: pd.DataFrame,
                                   folder_report: str, name: str):
    pdata = mpc.pdata
    if pdata is None:
        raise InputError('pdata is None!')
    if not exists(folder_report):
        mkdir(folder_report)
    # Write df_sim
    fn_df_sim = join(folder_report, 'df_sim_' + name + '.csv')
    write_df_result(df_sim, fn_df_sim)

    # Write df_ins
    fn_df_ins = join(folder_report, 'df_ins_' + name + '.csv')
    write_df_result(df_ins, fn_df_ins)

    # Write outofsample performance report
    fn_report_general = join(folder_report, 'report_general_' + name + '.yaml')
    dict_general_report = write_report(pdata, df_sim, mpc)
    with open(fn_report_general, 'w') as hfile:
        hfile.write(yaml.dump(dict_general_report))


def run_bilinear_experiments(l_cases_tuples, ofol, n_sce, n_win_ahead, n_win_before, n_days,
                             n_days_delay, time_config=None):
    if not exists(ofol):
        mkdir(ofol)

    # Run for all cases
    for arg in l_cases_tuples:
        # Read case
        pdata = Adn()
        pdata.read(*arg)

        if time_config is not None:
            pdata.time_config = time_config

        # ---------------------------------------------------------------------------------------- #
        # Run stochastic perfect                                                                   #
        # ---------------------------------------------------------------------------------------- #
        # Set config
        pdata.set('ctrl_type', 'droop')
        pdata.set('scegen_type', 'ddus_kmeans')
        pdata.set('scegen_n_win_ahead', 0)
        pdata.set('scegen_n_win_before', 0)
        pdata.set('scegen_n_days', 0)
        pdata.set('scegen_n_days_delay', 0)
        pdata.set('ctrl_robust_n_sce', n_sce)
        pdata.set('ctrl_robust', True)
        pdata.set('risk_measure', 'expected_value')
        pdata.set('cost_lasso_v', 1.e-12)
        pdata.set('cost_vlim', 1.e-5)
        pdata.set('cost_putility', 1.)

        """
        # Run experiment
        mpc = Mpc(pdata)
        df_sim, df_ins = mpc.run_toy_test()

        # Write report
        exp_name = 'sto_per'
        exp_name = arg[1] + '_' + exp_name
        folder_report = join(ofol, 'report_' + exp_name)

        write_report_single_experiment(mpc, df_sim, df_ins, folder_report, exp_name)
        gc.collect()
        # ---------------------------------------------------------------------------------------- #
        # Run robust perfect
        # ---------------------------------------------------------------------------------------- #
        # Set config
        pdata.set('risk_measure', 'worst_case')

        # Run experiment
        mpc = Mpc(pdata)
        df_sim, df_ins = mpc.run_toy_test()

        # Write report
        exp_name = 'rob_per'
        exp_name = arg[1] + '_' + exp_name
        folder_report = join(ofol, 'report_' + exp_name)

        write_report_single_experiment(mpc, df_sim, df_ins, folder_report, exp_name)

        gc.collect()
        
        
        # ---------------------------------------------------------------------------------------- #
        # Run stochastic DDUSKmeans                                                                #
        # ---------------------------------------------------------------------------------------- #
        # Set config
        pdata.set('risk_measure', 'expected_value')
        pdata.set('scegen_n_win_ahead', n_win_ahead)
        pdata.set('scegen_n_win_before', n_win_before)
        pdata.set('scegen_n_days', n_days)
        pdata.set('scegen_n_days_delay', n_days_delay)

        # Run experiment
        mpc = Mpc(pdata)
        df_sim, df_ins = mpc.run_toy_test()

        # Write report
        exp_name = 'sto_kme'
        exp_name = arg[1] + '_' + exp_name
        folder_report = join(ofol, 'report_' + exp_name)

        write_report_single_experiment(mpc, df_sim, df_ins, folder_report, exp_name)

        gc.collect()

        # ---------------------------------------------------------------------------------------- #
        # Run robust DDUSKmeans
        # ---------------------------------------------------------------------------------------- #"""

        # Set config
        pdata.set('risk_measure', 'worst_case')
        pdata.set('scegen_n_win_ahead', n_win_ahead)
        pdata.set('scegen_n_win_before', n_win_before)
        pdata.set('scegen_n_days', n_days)
        pdata.set('scegen_n_days_delay', n_days_delay)

        # Run experiment
        mpc = Mpc(pdata)
        df_sim, df_ins = mpc.run_toy_test()

        # Write report
        exp_name = 'rob_kme'
        exp_name = arg[1] + '_' + exp_name
        folder_report = join(ofol, 'report_' + exp_name)

        write_report_single_experiment(mpc, df_sim, df_ins, folder_report, exp_name)

        gc.collect()


def main():
    OFOL = '/home/jp/tesis/experiments/ieee4bus_lasso/droop_08_08'
    if not exists(OFOL):
        mkdir(OFOL)

    time_config = {
        'tini': datetime(2018, 7, 27, 0, 0, 0),
        'tiniout': datetime(2018, 8, 11, 0, 0, 0),
        'tend': datetime(2018, 8, 13, 23, 59, 54),
        'dt': timedelta(seconds=6),
        'n_rh': 150
    }

    #l_cases = [('/home/jp/tesis/experiments/toy_lasso/toy1bus', 'toy1bus')]
    l_cases = [('/home/jp/tesis/experiments/toy_lasso/ieee4bus', 'ieee4bus')]
    #l_cases = [('/home/jp/tesis/experiments/scaled_cases/ieee34bus_ev0.2_pv1.0_scaled_v5_rescaled',
    #           'ieee34bus')]

    run_bilinear_experiments(l_cases, OFOL, n_sce=100, n_win_ahead=2, n_win_before=2, n_days=10,
                             n_days_delay=1, time_config=time_config)


def main_to_benchmark():
    OFOL = '/home/jp/tesis/experiments/34bus_lasso/08_18_benchmark'
    AR_LASSO_COEFF = [3.684031498640386e-12]
    CASE_FOLDER = '/home/jp/tesis/experiments/scaled_cases/ieee34bus_ev0.1_pv1.0_scaled_ordered_noB'
    CASE_NAME = 'ieee34bus'
    BILINEAR_APPROX = 'mccormick'
    FN_ADAWEIGHTS = ('/home/jp/tesis/experiments/34bus_lasso/08_18_benchmark/rob_kme/'
                     'df_ins_ieee34bus_rob_kme_cx0.0.csv')
    lcases = [(CASE_FOLDER, CASE_NAME)]
    TIME_CONFIG = {
        'tini': datetime(2018, 7, 27, 0, 0, 0),
        'tiniout': datetime(2018, 8, 15, 0, 0, 0),
        'tend': datetime(2018, 8, 17, 23, 59, 54),
        'dt': timedelta(seconds=6),
        'n_rh': 150
    }
    # -------------------------------------------------------------------------------------------- #
    # Set io
    # -------------------------------------------------------------------------------------------- #
    if not exists(OFOL):
        mkdir(OFOL)

    # -------------------------------------------------------------------------------------------- #
    # Run experiment and write reports
    # -------------------------------------------------------------------------------------------- #
    run_bilinear_experiments(l_cases, OFOL, n_sce=500, n_win_ahead=3, n_win_before=3, n_days=15,
                             n_days_delay=1, time_config=TIME_CONFIG)


if __name__ == '__main__':
    main()
