from experiments.final_toy_experiments.run_lasso_determination import write_report_single_experiment
from data_structure import Adn
from datetime import datetime, timedelta
from mpc import Mpc
from os.path import exists, join
from os import mkdir


def run_link_failure_total(
    ofol,
    logfol,
    case_folder,
    case_name,
    fn_adalasso_weights,
    cost_lasso_x,
    time_config,
    n_sce,
    n_days,
    n_win_ahead,
    n_win_before,
    n_days_delay
):

    exp_name = 'link_failure_all'

    # ---------------------------------------------------------------------------------------------#
    # Preliminaries
    # ---------------------------------------------------------------------------------------------#
    if not exists(ofol):
        mkdir(ofol)

    # Load pdata
    pdata = Adn()
    pdata.read(case_folder, case_name)
    pdata.time_config = time_config

    pdata.branches.loc[:, 'b'] = 0.
    pdata.validate()

    # ---------------------------------------------------------------------------------------------#
    # Config
    # ---------------------------------------------------------------------------------------------#
    pdata.set('log_folder', logfol)
    pdata.set('ctrl_type', 'droop_polypol')
    pdata.set('fn_adalasso_weights', fn_adalasso_weights)
    pdata.set('cost_lasso_x', cost_lasso_x)
    pdata.set('bilinear_approx', 'mccormick')
    pdata.set('cost_lasso_v', 1e-11)

    pdata.set('cost_putility', 1.)
    pdata.set('cost_vlim', 1.e5)
    pdata.set('ctrl_robust', True)

    l_buses = pdata.l_buses
    pdata.buses.loc[l_buses, 'mc_dv_ub'] = 0.05
    pdata.buses.loc[l_buses, 'mc_dv_lb'] = -0.05
    pdata.buses.loc[l_buses, 'mc_gp_lb'] = -0.2
    pdata.buses.loc[l_buses, 'mc_gq_lb'] = -0.2

    pdata.set('risk_measure', 'worst_case')
    pdata.set('scegen_type', 'ddus_kmeans')
    pdata.set('ctrl_robust_n_sce', n_sce)
    pdata.set('scegen_n_win_ahead', n_win_ahead)
    pdata.set('scegen_n_win_before', n_win_before)
    pdata.set('scegen_n_days', n_days)
    pdata.set('scegen_n_days_delay', n_days_delay)

    # ---------------------------------------------------------------------------------------------#
    # Run experiment
    # ---------------------------------------------------------------------------------------------#
    mpc = Mpc(pdata)
    df_sim, df_ins = mpc.run()
    write_report_single_experiment(mpc, df_sim, df_ins, ofol, name=exp_name)



