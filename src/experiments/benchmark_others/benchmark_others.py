import gc
import pandas as pd
from data_structure import Adn
from global_definitions import *
from mpc import Mpc
from post_process.post_process import benchmark_solutions, write_report
from post_process.visualization import plot_debug_panel
from front_end_utilities import down_sample_pdata, write_df_result, load_df_result
from datetime import timedelta
from os.path import join, exists
from os import mkdir
import subprocess

""" ############################################################################################ """
""" ##################################### Setting up pdata #####################################"""
""" ############################################################################################ """
OFOL = '/home/jp/tesis/experiments/benchmark_2weeks_bw33_v4'
CASE_FOLDER = '/home/jp/tesis/experiments/bw33_chunk_0/case33bw_6'
CASE_NAME = 'case33bw_6'

if not exists(OFOL):
    mkdir(OFOL)

# Reading pdata from file
pdata = Adn()
pdata.read(CASE_FOLDER, CASE_NAME)

# Modifying pdata
# Scaling dgpmax
l_cols_dgpmax = [i for i in pdata.df_data.columns if i.startswith(SNAM_DGPMAX)]
pdata.df_data.loc[:, l_cols_dgpmax] = pdata.df_data.loc[:, l_cols_dgpmax] * 2.0
pdata.dgs.loc[:, 'snom'] = pdata.df_data.loc[:, l_cols_dgpmax].max().max() * 3
pdata.branches.loc[:, 'imax'] = 9999.
l_buses = pdata.l_buses
# Scaling loads
l_loads = pdata.l_loads
l_cols_loads = [SNAM_LOADP + str(i) for i in l_loads]
l_cols_loaddgp = [SNAM_LOADDGP + str(i) for i in l_loads]
l_cols_loadevp = [SNAM_LOADEVP + str(i) for i in l_loads]
l_cols_load_related = l_cols_loads + l_cols_loaddgp + l_cols_loadevp
pdata.df_data.loc[:, l_cols_load_related] = pdata.df_data.loc[:, l_cols_load_related] * 1.03

pdata.buses.loc[0, 'vmin'] = 1.
pdata.buses.loc[0, 'vmax'] = 1.
pdata.buses.loc[l_buses, 'vmin'] = 0.95
pdata.buses.loc[l_buses, 'vmax'] = 1.05
pdata.set('cost_losses', 0.0001)
pdata.set('cost_vlim', 9999999.)
pdata.set('opf_model', 'lindistflow')
vr = 0.05

# Time config
idx_tini_all = 0
idx_tini = 1440*1
idx_tend = 1440*15 - 1
dt_original = pd.to_timedelta(pdata.df_data.index.freq)
tini_all = pdata.df_data.index[idx_tini_all]
tini = pdata.df_data.index[idx_tini]
tend = pdata.df_data.index[idx_tend]

# Down sample data
dt_hour = timedelta(hours=1)
assert (tend - tini_all + dt_original) % dt_hour == timedelta(0)
n_hours = (tend - tini_all + dt_original) // dt_hour
dt = timedelta(seconds=30)
n_rh = 15 * (dt_original // dt)
down_sample_pdata(pdata, n_hours, n_rh, Dt_down=dt, idx_ini=idx_tini_all)
pdata.time_config['tini'] = tini

o_case_folder = join(OFOL, CASE_NAME)
if not exists(o_case_folder):
    mkdir(o_case_folder)
pdata.write(o_case_folder, CASE_NAME)

""" ############################################################################################ """
""" ################################### Running experiments ####################################"""
""" ############################################################################################ """
try:

    # NCognizant deterministic
    pdata.time_config['n_rh'] = n_rh
    mpc = Mpc(pdata)
    df_sol_nc_det, _ = mpc.run_ncognizant()

    write_df_result(df_sol_nc_det, join(OFOL, 'df_sim_ncognizant'))
    with open(join(OFOL, 'report_ncognizant.txt'), 'w') as hfile:
        hfile.write(write_report(pdata, df_sol_nc_det, mpc, nvv_vr_ratio=vr))

    """
    # IEEE
    df_sol_ieee = mpc.run_ieee_1547()
    write_df_result(df_sol_ieee, join(OFOL, 'df_sim_ieee'))
    with open(join(OFOL, 'report_ieee.txt'), 'w') as hfile:
        hfile.write(write_report(pdata, df_sol_ieee, mpc, nvv_vr_ratio=vr))

    gc.collect()
    
    # Omnicient OPF
    pdata.time_config['n_rh'] = 144
    mpc = Mpc(pdata)
    df_sol_opf = mpc.run_perfect_opf()
    write_df_result(df_sol_opf, join(OFOL, 'df_sim_opf'))
    with open(join(OFOL, 'report_opf.txt'), 'w') as hfile:
        hfile.write(write_report(pdata, df_sol_opf, mpc, nvv_vr_ratio=vr))

    mpc = None
    gc.collect()
    
    # OPF-based MPC noctrl
    pdata.time_config['n_rh'] = n_rh
    mpc = Mpc(pdata)
    df_sol_noctrl, _ = mpc.run_noctrl()
    write_df_result(df_sol_noctrl, join(OFOL, 'df_sim_noctrl'))
    with open(join(OFOL, 'report_noctrl.txt'), 'w') as hfile:
        hfile.write(write_report(pdata, df_sol_noctrl, mpc, nvv_vr_ratio=vr))

    gc.collect()
    """

finally:
    ...
    # subprocess.call(['shutdown', '+0'])

"""
pdata = Adn()
pdata.read(o_case_folder, CASE_NAME)
df_sim_opf = load_df_result(join(OFOL, 'df_sim_opf'))
df_sim_ieee = load_df_result(join(OFOL, 'df_sim_ieee'))
df_sim_ncognizant = load_df_result(join(OFOL, 'df_sim_ncognizant'))
fig, _ = plot_debug_panel(pdata, df_sim_opf, pdata.df_data.loc[df_sim_opf.index, :])
fig.show()
fig, _ = plot_debug_panel(pdata, df_sim_ieee, pdata.df_data.loc[df_sim_opf.index, :])
fig.show()
fig, _ = plot_debug_panel(pdata, df_sim_ncognizant, pdata.df_data.loc[df_sim_opf.index, :])
fig.show()
"""





