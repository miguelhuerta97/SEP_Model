from post_process.post_process import load_df_result
from data_structure import Adn
import pandas as pd
from post_process.visualization import plot_debug_panel


CASE_FOLDER = '/home/jp/tesis/experiments/cases_final/ieee34bus'
CASE_NAME = 'ieee34bus'
FN_SIM = '/home/jp/tesis/experiments/final/ieee34bus/proposed/df_sim_proposed.csv'

pdata = Adn()
pdata.read(CASE_FOLDER, CASE_NAME)

df_sim = load_df_result(FN_SIM)

fig, axs = plot_debug_panel(pdata, df_sim)

fig.show()


