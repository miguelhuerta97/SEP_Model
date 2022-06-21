from os.path import join
from data_structure import Adn
from front_end_utilities import load_df_result
from post_process.visualization import plot_debug_panel
from global_definitions import *

IFOL = '/home/jp/tesis/experiments/benchmark_1week_33bw'
CASE_FOLDER = '/home/jp/tesis/experiments/benchmark_1week_33bw/case33bw_6'
CASE_NAME = 'case33bw_6'

o_case_folder = CASE_FOLDER

pdata = Adn()
pdata.read(o_case_folder, CASE_NAME)
df_sim_opf = load_df_result(join(IFOL, 'df_sim_opf'))
df_sim_ieee = load_df_result(join(IFOL, 'df_sim_ieee'))
df_sim_ncognizant = load_df_result(join(IFOL, 'df_sim_ncognizant'))
# df_sim_ieee = load_df_result(join(IFOL, 'df_sim_ieee'))

l_buses = pdata.l_buses
l_cols_v_opf = [SNAM_V_OPF + str(i) for i in l_buses]
l_cols_v = [SNAM_V + str(i) for i in l_buses]
l_cols_v_related = l_cols_v + l_cols_v_opf
FLOAT_EPS = 1e-2

df_v_opf = df_sim_opf[l_cols_v_opf]
df_v = df_sim_opf[l_cols_v]

df_idx_up = (df_v >= df_v_opf.values + FLOAT_EPS)
df_idx_down = (df_v <= df_v_opf.values - FLOAT_EPS)

print('caca')

fig, _ = plot_debug_panel(pdata, df_sim_opf, pdata.df_data.loc[df_sim_opf.index, :])
fig.show()
fig, _ = plot_debug_panel(pdata, df_sim_ncognizant, pdata.df_data.loc[df_sim_opf.index, :])
fig.show()
fig, _ = plot_debug_panel(pdata, df_sim_ieee, pdata.df_data.loc[df_sim_opf.index, :])
fig.show()


