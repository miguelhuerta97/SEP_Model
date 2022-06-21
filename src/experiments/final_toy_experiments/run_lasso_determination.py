import gc
import pandas as pd
import numpy as np

import sys
from os import mkdir, walk
from os.path import join, exists, dirname

from front_end_utilities import write_df_result
from post_process.post_process import write_report
from errors import InputError
from jpyaml import yaml
from data_structure import Adn
from mpc import Mpc
from datetime import datetime, timedelta


def make_exp_suffix(case_name, exp_name, cost_lasso_D):
    return case_name + '_' + exp_name + '_' + 'cx{}'.format(cost_lasso_D)


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


def run_general_experiment_and_report(pdata, ar_lasso_coeff, case_name, exp_name, exp_folder):
    # Run experiment
    for cost_lasso_D in ar_lasso_coeff:
        exp_suffix = make_exp_suffix(case_name, exp_name, cost_lasso_D)
        pdata.set('cost_lasso_x', float(cost_lasso_D))

        # Setting new log_folder to dump necessary insample results
        new_log_folder = join(exp_folder, 'log_' + str(float(cost_lasso_D)))
        if not exists(new_log_folder):
            mkdir(new_log_folder)
        pdata.set('log_folder', new_log_folder)

        mpc = Mpc(pdata)
        df_sim, df_ins = mpc.run()

        # Write report
        write_report_single_experiment(mpc, df_sim, df_ins, exp_folder, exp_suffix)


def run_lasso_determination(case_folder, case_name, ofol, ar_lasso_coeff, n_sce, n_win_ahead, n_win_before,
                            n_days, n_days_delay, bilinear_approx, time_config=None,
                            fn_adalasso_weights=None):
    if not exists(ofol):
        mkdir(ofol)

    # -------------------------------------------------------------------------------------------- #
    # Load pdata and general config
    # -------------------------------------------------------------------------------------------- #
    pdata = Adn()
    pdata.read(case_folder, case_name)

    # Set time config
    if time_config is not None:
        pdata.time_config = time_config

    # Set general config

    pdata.set('bilinear_approx', bilinear_approx)
    pdata.branches.loc[:, 'b'] = 0.

    if bilinear_approx == 'mccormick':
        l_buses = pdata.l_buses
        pdata.buses.loc[l_buses, 'mc_dv_ub'] = 0.05
        pdata.buses.loc[l_buses, 'mc_dv_lb'] = -0.05
        pdata.buses.loc[l_buses, 'mc_gp_lb'] = -0.2
        pdata.buses.loc[l_buses, 'mc_gq_lb'] = -0.2

    if fn_adalasso_weights:
        pdata.set('fn_adalasso_weights', fn_adalasso_weights)

    pdata.set('ctrl_type', 'droop_polypol')
    pdata.set('scegen_type', 'ddus_kmeans')
    pdata.set('ctrl_robust_n_sce', n_sce)
    pdata.set('cost_lasso_v', 1e-11)
    pdata.set('cost_putility', 1.)
    pdata.set('cost_vlim', 1.e5)
    assert pdata.cost_putility == 1.
    pdata.set('ctrl_robust', True)  # Name of the config param doesn't represent the feature
    #pdata.write('/home/jp/tesis/experiments/34bus_lasso/ieee34bus', 'ieee34bus')
    #exit()
    """FDF
    # ---------------------------------------------------------------------------------------- #
    # Run stochastic perfect                                                                   #
    # ---------------------------------------------------------------------------------------- #
    # Set config

    exp_name = 'sto_per'
    exp_folder = join(ofol, exp_name)
    if not exists(exp_folder):
        mkdir(exp_folder)

    pdata.set('log_folder', join(ofol, 'log_' + exp_name))
    pdata.set('scegen_n_win_ahead', 0)
    pdata.set('scegen_n_win_before', 0)
    pdata.set('scegen_n_days', 0)
    pdata.set('scegen_n_days_delay', 0)
    pdata.set('risk_measure', 'expected_value')

    # Run experiment
    run_general_experiment_and_report(pdata, ar_lasso_coeff, case_name, exp_name, exp_folder)
    gc.collect()


    # ---------------------------------------------------------------------------------------- #
    # Run robust perfect
    # ---------------------------------------------------------------------------------------- #
    # Set config
    exp_name = 'rob_per'
    exp_folder = join(ofol, exp_name)
    if not exists(exp_folder):
        mkdir(exp_folder)

    pdata.set('log_folder', join(ofol, 'log_' + exp_name))
    pdata.set('scegen_type', 'all')
    pdata.set('ctrl_robust_n_sce', 150)
    pdata.set('scegen_n_win_ahead', 0)
    pdata.set('scegen_n_win_before', 0)
    pdata.set('scegen_n_days', 0)
    pdata.set('scegen_n_days_delay', 0)
    pdata.set('risk_measure', 'worst_case')

    # Run experiment
    run_general_experiment_and_report(pdata, ar_lasso_coeff, case_name, exp_name, exp_folder)
    gc.collect()
    
    
    # ---------------------------------------------------------------------------------------- #
    # Run stochastic DDUSKmeans                                                                #
    # ---------------------------------------------------------------------------------------- #
    # Set config
    exp_name = 'sto_kme'
    exp_folder = join(ofol, exp_name)
    if not exists(exp_folder):
        mkdir(exp_folder)

    pdata.set('risk_measure', 'expected_value')
    pdata.set('scegen_n_win_ahead', n_win_ahead)
    pdata.set('scegen_n_win_before', n_win_before)
    pdata.set('scegen_n_days', n_days)
    pdata.set('scegen_n_days_delay', n_days_delay)

    # Run experiment
    run_general_experiment_and_report(pdata, ar_lasso_coeff, case_name, exp_name, exp_folder)
    gc.collect()
    
    """
    # ---------------------------------------------------------------------------------------- #
    # Run robust DDUSKmeans
    # ---------------------------------------------------------------------------------------- #

    exp_name = 'rob_kme'
    exp_folder = join(ofol, exp_name)
    if not exists(exp_folder):
        mkdir(exp_folder)
    # Set config
    pdata.set('opt_gurobi', 'outlev=1 barhomogeneous=1 presolve=0 numericfocus=3 barcorrectors=2000 presolve=0 aggregate=0 barconvtol=1e-14')
    pdata.set('scegen_type', 'ddus_kmeans')
    pdata.set('log_folder', join(ofol, 'log_' + exp_name))
    pdata.set('ctrl_robust_n_sce', n_sce)
    pdata.set('risk_measure', 'worst_case')
    pdata.set('scegen_n_win_ahead', n_win_ahead)
    pdata.set('scegen_n_win_before', n_win_before)
    pdata.set('scegen_n_days', n_days)
    pdata.set('scegen_n_days_delay', n_days_delay)

    # Run experiment
    run_general_experiment_and_report(pdata, ar_lasso_coeff, case_name, exp_name, exp_folder)
    gc.collect()


def main():
    # -------------------------------------------------------------------------------------------- #
    # Input
    # -------------------------------------------------------------------------------------------- #
    """
    Exp_08_12:

    OFOL = '/home/jp/tesis/experiments/34bus_lasso/08_12'
    AR_LASSO_COEFF = np.logspace(start=np.log10(5e-15), stop=np.log10(1.), num=10)
    AR_LASSO_COEFF = AR_LASSO_COEFF[2:]
    CASE_FOLDER = '/home/jp/tesis/experiments/scaled_cases/ieee34bus_ev0.1_pv1.0_scaled_ordered_noB'
    CASE_NAME = 'ieee34bus'
    BILINEAR_APPROX = 'mccormick'

    # CASE_FOLDER = '/home/jp/tesis/experiments/toy_lasso/ieee4bus'
    # CASE_NAME = 'ieee4bus'

    TIME_CONFIG = {
        'tini': datetime(2018, 7, 27, 0, 0, 0),
        'tiniout': datetime(2018, 8, 15, 0, 0, 0),
        'tend': datetime(2018, 8, 15, 23, 59, 54),
        'dt': timedelta(seconds=6),
        'n_rh': 150
    }

    run_lasso_determination(
        case_folder=CASE_FOLDER, case_name=CASE_NAME,
        ofol=OFOL,
        ar_lasso_coeff=AR_LASSO_COEFF,
        n_sce=500, n_win_ahead=3, n_win_before=3, n_days=16, n_days_delay=0,
        bilinear_approx=BILINEAR_APPROX,
        time_config=TIME_CONFIG
    )
    """

    OFOL = '/home/jp/tesis/experiments/34bus_lasso/08_17'
    #AR_LASSO_COEFF = np.logspace(start=np.log10(1e-14), stop=np.log10(1e-10), num=4)
    AR_LASSO_COEFF = np.logspace(start=3.68403150e-12, stop=7.74263683e-11, num=3)
    AR_LASSO_COEFF = [AR_LASSO_COEFF[1]]

    print(AR_LASSO_COEFF)
    CASE_FOLDER = '/home/jp/tesis/experiments/scaled_cases/ieee34bus_ev0.1_pv1.0_scaled_ordered_noB'
    CASE_NAME = 'ieee34bus'
    BILINEAR_APPROX = 'mccormick'
    FN_ADAWEIGHTS = ('/home/jp/Data/Dropbox/tesis/experiments/34bus_lasso/08_16/rob_kme/'
                     'df_ins_ieee34bus_rob_kme_cx0.0.csv')

    # CASE_FOLDER = '/home/jp/tesis/experiments/toy_lasso/ieee4bus'
    # CASE_NAME = 'ieee4bus'

    TIME_CONFIG = {
        'tini': datetime(2018, 7, 27, 0, 0, 0),
        'tiniout': datetime(2018, 8, 15, 0, 0, 0),
        'tend': datetime(2018, 8, 15, 23, 59, 54),
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
    run_lasso_determination(
        case_folder=CASE_FOLDER, case_name=CASE_NAME,
        ofol=OFOL,
        ar_lasso_coeff=AR_LASSO_COEFF,
        n_sce=500, n_win_ahead=3, n_win_before=3, n_days=16, n_days_delay=1,
        bilinear_approx=BILINEAR_APPROX,
        time_config=TIME_CONFIG,
        fn_adalasso_weights=FN_ADAWEIGHTS
    )


def main_to_definitive_lasso():
    OFOL = '/home/jp/tesis/experiments/34bus_lasso/08_19/compare'
    #AR_LASSO_COEFF = np.logspace(start=np.log10(1e-13), stop=np.log10(1e-11), num=3)
    AR_LASSO_COEFF = [1e-11]
    CASE_FOLDER = '/home/jp/tesis/experiments/scaled_cases/ieee34bus_ev0.1_pv1.0_scaled_ordered_noB'
    CASE_NAME = 'ieee34bus'
    BILINEAR_APPROX = 'mccormick'
    FN_ADAWEIGHTS = ('/home/jp/tesis/experiments/34bus_lasso/08_18_benchmark/rob_kme/'
                     'df_ins_ieee34bus_rob_kme_cx0.0.csv')

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
    run_lasso_determination(
        case_folder=CASE_FOLDER, case_name=CASE_NAME,
        ofol=OFOL,
        ar_lasso_coeff=AR_LASSO_COEFF,
        n_sce=500, n_win_ahead=3, n_win_before=3, n_days=15, n_days_delay=1,
        bilinear_approx=BILINEAR_APPROX,
        time_config=TIME_CONFIG,
        fn_adalasso_weights=FN_ADAWEIGHTS
    )


def main_to_benchmark():
    OFOL = '/home/jp/tesis/experiments/34bus_lasso/08_18_benchmark'
    AR_LASSO_COEFF = [3.684031498640386e-12]
    CASE_FOLDER = '/home/jp/tesis/experiments/scaled_cases/ieee34bus_ev0.1_pv1.0_scaled_ordered_noB'
    CASE_NAME = 'ieee34bus'
    BILINEAR_APPROX = 'mccormick'
    FN_ADAWEIGHTS = ('/home/jp/tesis/experiments/34bus_lasso/08_18_benchmark/rob_kme/'
                     'df_ins_ieee34bus_rob_kme_cx0.0.csv')

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
    run_lasso_determination(
        case_folder=CASE_FOLDER, case_name=CASE_NAME,
        ofol=OFOL,
        ar_lasso_coeff=AR_LASSO_COEFF,
        n_sce=500, n_win_ahead=3, n_win_before=3, n_days=15, n_days_delay=1,
        bilinear_approx=BILINEAR_APPROX,
        time_config=TIME_CONFIG,
        fn_adalasso_weights=FN_ADAWEIGHTS
    )


def main_run_benchmark_preliminaries():
    # Parse arguments
    OFOL = None
    CASE_FOLDER = None
    CASE_NAME = None
    n_args = len(sys.argv)
    if n_args == 4:
        OFOL = sys.argv[1]
        CASE_FOLDER = sys.argv[2]
        CASE_NAME = sys.argv[3]

    # -------------------------------------------------------------------------------------------- #
    # Input
    # -------------------------------------------------------------------------------------------- #
    if OFOL is None:
        OFOL = '/home/jp/tesis/experiments/34bus_lasso/08_20_benchmark_preliminaries'
    if CASE_FOLDER is None:
        CASE_FOLDER = '/home/jp/tesis/experiments/cases_final/ieee4bus'
    if CASE_NAME is None:
        CASE_NAME = 'ieee4bus'

    AR_LASSO_COEFF = [1e-14]
    BILINEAR_APPROX = 'bilinear'

    FN_ADAWEIGHTS = None

    TIME_CONFIG = {
        'tini': datetime(2018, 7, 27, 0, 0, 0),
        'tiniout': datetime(2018, 8, 15, 0, 0, 0),
        'tend': datetime(2018, 8, 21, 23, 59, 54),
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
    run_lasso_determination(
        case_folder=CASE_FOLDER, case_name=CASE_NAME,
        ofol=OFOL,
        ar_lasso_coeff=AR_LASSO_COEFF,
        n_sce=100, n_win_ahead=3, n_win_before=3, n_days=15, n_days_delay=1,
        bilinear_approx=BILINEAR_APPROX,
        time_config=TIME_CONFIG,
        fn_adalasso_weights=FN_ADAWEIGHTS
    )


def main_run_benchmark_preliminaries_34bus():
    # Parse arguments
    OFOL = None
    CASE_FOLDER = None
    CASE_NAME = None
    n_args = len(sys.argv)
    if n_args == 4:
        OFOL = sys.argv[1]
        CASE_FOLDER = sys.argv[2]
        CASE_NAME = sys.argv[3]

    # -------------------------------------------------------------------------------------------- #
    # Input
    # -------------------------------------------------------------------------------------------- #
    if OFOL is None:
        OFOL = '/home/jp/tesis/experiments/34bus_lasso/08_20_benchmark_preliminaries'
    if CASE_FOLDER is None:
        CASE_FOLDER = '/home/jp/tesis/experiments/cases_final/ieee4bus'
    if CASE_NAME is None:
        CASE_NAME = 'ieee4bus'

    AR_LASSO_COEFF = [1e-14]
    BILINEAR_APPROX = 'bilinear'

    FN_ADAWEIGHTS = None

    TIME_CONFIG = {
        'tini': datetime(2018, 7, 27, 0, 0, 0),
        'tiniout': datetime(2018, 8, 15, 0, 0, 0),
        'tend': datetime(2018, 8, 21, 23, 59, 54),
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
    run_lasso_determination(
        case_folder=CASE_FOLDER, case_name=CASE_NAME,
        ofol=OFOL,
        ar_lasso_coeff=AR_LASSO_COEFF,
        n_sce=100, n_win_ahead=3, n_win_before=3, n_days=15, n_days_delay=1,
        bilinear_approx=BILINEAR_APPROX,
        time_config=TIME_CONFIG,
        fn_adalasso_weights=FN_ADAWEIGHTS
    )


