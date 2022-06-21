import gc
import pandas as pd
import numpy as np
from os import mkdir, walk
from os.path import join, exists, dirname
from front_end_utilities import write_df_result
from post_process.post_process import write_report
from errors import InputError
from jpyaml import yaml
from data_structure import Adn
from mpc import Mpc
from datetime import datetime, timedelta


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


def run_benchmark_others(dict_cases, ofol, time_config=None, noctrl=False, kyri=False,
                         ieee1547=False, proposed=False, droop=False, bikyri=False, polypoldeg=3):

    if not (
        isinstance(noctrl, bool) and
        isinstance(kyri, bool) and
        isinstance(ieee1547, bool) and
        isinstance(proposed, bool) and
        isinstance(droop, bool) and
        isinstance(bikyri, bool)
    ):
        raise TypeError

    if not exists(ofol):
        mkdir(ofol)

    for case_name, case_props in dict_cases.items():
        case_folder = case_props['case_folder']
        fn_adalasso_weights = case_props['fn_adalasso_weights']
        cost_lasso_x = case_props['cost_lasso_x']
        n_sce = case_props['n_sce']
        n_win_ahead = case_props['n_win_ahead']
        n_win_before = case_props['n_win_before']
        n_days = case_props['n_days']
        n_days_delay = case_props['n_days_delay']

        # ---------------------------------------------------------------------------------------- #
        # Read case
        # ---------------------------------------------------------------------------------------- #
        pdata = Adn()
        pdata.read(case_folder, case_name)
        if time_config is not None:
            pdata.time_config = time_config

        # Validation
        assert pdata.branches.loc[:, 'b'].sum() == 0.
        # TODO validate DG index and bus order. ssame with loads?
        # ---------------------------------------------------------------------------------------- #
        # General configuration
        # ---------------------------------------------------------------------------------------- #
        pdata.set('log_folder', None)

        pdata.set('cost_vlim', 1e5)
        pdata.set('cost_putility', 1.)
        pdata.set('cost_lasso_v', 1e-11)

        pdata.set('ctrl_robust', True)
        pdata.set('risk_measure', 'worst_case')
        pdata.set('scegen_type', 'ddus_kmeans')
        pdata.set('ctrl_robust_n_sce', n_sce)
        pdata.set('scegen_n_win_ahead', n_win_ahead)
        pdata.set('scegen_n_win_before', n_win_before)
        pdata.set('scegen_n_days', n_days)
        pdata.set('scegen_n_days_delay', n_days_delay)

        pdata.set('cost_lasso_x', cost_lasso_x)
        pdata.set('fn_adalasso_weights', fn_adalasso_weights)

        # Case dependent config
        if case_name == 'ieee4bus':
            pdata.set('bilinear_approx', 'bilinear')
        else:
            pdata.set('bilinear_approx', 'mccormick')
            l_buses = pdata.l_buses
            pdata.buses.loc[l_buses, 'mc_dv_ub'] = 0.09
            pdata.buses.loc[l_buses, 'mc_dv_lb'] = -0.09
            pdata.buses.loc[l_buses, 'mc_gp_lb'] = -0.3
            pdata.buses.loc[l_buses, 'mc_gq_lb'] = -0.3

        # Validation
        pdata.validate()

        # ---------------------------------------------------------------------------------------- #
        # Run Kyri                                                                                 #
        # ---------------------------------------------------------------------------------------- #
        if kyri:
            # Prelimiries
            exp_name = 'kyri'
            exp_folder = join(ofol, case_name)
            if not exists(exp_folder):
                mkdir(exp_folder)
            # Experiment config
            pdata.set('ctrl_type', 'droop')
            pdata.set('bilinear_approx', 'neuman')
            pdata.set('cost_vlim', 1.)

            # Run experiment
            mpc = Mpc(pdata)
            df_sim, df_ins = mpc.run()

            # Write report
            write_report_single_experiment(exp_folder, exp_name, mpc, df_sim, df_ins)

        # ---------------------------------------------------------------------------------------- #
        # Run Proposed                                                                             #
        # ---------------------------------------------------------------------------------------- #
        if proposed:
            # Preliminaries
            exp_name = 'proposed'
            exp_folder = join(ofol, case_name)
            if not exists(exp_folder):
                mkdir(exp_folder)
            # Experiment config
            pdata.set('cost_vlim', 1.e5)
            pdata.set('ctrl_type', 'droop_polypol')
            pdata.set('polypol_deg', polypoldeg)

            # Run experiment
            mpc = Mpc(pdata)
            df_sim, df_ins = mpc.run()

            # Write report
            write_report_single_experiment(exp_folder, exp_name, mpc, df_sim, df_ins)
        # ---------------------------------------------------------------------------------------- #
        # Run Droop                                                                                #
        # ---------------------------------------------------------------------------------------- #
        if droop:
            # Experiment config
            exp_name = 'droop'
            exp_folder = join(ofol, case_name)
            if not exists(exp_folder):
                mkdir(exp_folder)
            pdata.set('ctrl_type', 'droop')
            pdata.set('cost_vlim', 1.e5)

            # Run experiment
            mpc = Mpc(pdata)
            df_sim, df_ins = mpc.run()

            # Write report
            write_report_single_experiment(exp_folder, exp_name, mpc, df_sim, df_ins)
        # ---------------------------------------------------------------------------------------- #
        # Run noctrl                                                                               #
        # ---------------------------------------------------------------------------------------- #
        if noctrl:
            # Set config
            exp_name = 'noctrl'
            pdata.set('cost_vlim', 1.e5)
            exp_folder = join(ofol, case_name)
            if not exists(exp_folder):
                mkdir(exp_folder)

            # Run experiment
            mpc = Mpc(pdata)
            df_sim, df_ins = mpc.run_noctrl_new()

            # Write report
            write_report_single_experiment(exp_folder, exp_name, mpc, df_sim, df_ins)

        # ---------------------------------------------------------------------------------------- #
        # Run IEEE-1547                                                                            #
        # ---------------------------------------------------------------------------------------- #
        if ieee1547:
            # Set config
            exp_name = 'ieee1547'
            exp_folder = join(ofol, case_name)
            if not exists(exp_folder):
                mkdir(exp_folder)

            # Run experiment
            mpc = Mpc(pdata)
            df_sim = mpc.run_ieee_1547()

            # Write report
            write_report_single_experiment(exp_folder, exp_name, mpc, df_sim)

        # ---------------------------------------------------------------------------------------- #
        # Run bilinear kyri                                                                        #
        # ---------------------------------------------------------------------------------------- #
        if bikyri:
            # Set config
            exp_name = 'bikyri'
            pdata.set('ctrl_type', 'droop')
            pdata.set('bilinear_approx', 'bilinear_neuman')
            pdata.set('cost_vlim', 1.)

            exp_folder = join(ofol, case_name)
            if not exists(exp_folder):
                mkdir(exp_folder)

            # Run experiment
            mpc = Mpc(pdata)
            df_sim, df_ins = mpc.run()

            # Write report
            write_report_single_experiment(exp_folder, exp_name, mpc, df_sim)
