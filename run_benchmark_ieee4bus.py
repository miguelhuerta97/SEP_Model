import sys, os
sys.path.append(os.path.normpath('/home/miguel/jp_sep/src/'))
from experiments.final_toy_experiments.run_benchmark_others import *
from config import *

print('ieee4bus')
OFOL = '/home/miguel/jp_sep/results_experiments/ieee4bus/'
if not exists(OFOL):
	os.makedirs(OFOL)

TIME_CONFIG = {
           'tini': datetime(2018, 7, 27, 0, 0, 0),
        'tiniout': datetime(2018, 8, 15, 0, 0, 0),
           'tend': datetime(2018, 8, 21, 23, 59, 54),
        'dt': timedelta(seconds=6),
        'n_rh': 150
    }

DICT_CASES = {
        'ieee4bus': {
        'case_folder':'/home/miguel/jp_sep/cases/ieee4bus/',
        'fn_adalasso_weights': None,
        'cost_lasso_x': 1.e-9,
        'n_sce': 100, 
        'n_win_ahead': 0, 
        'n_win_before': 0, 
        'n_days': 15, 
        'n_days_delay': 0
    }

    
}

run_benchmark_others(DICT_CASES, OFOL, TIME_CONFIG, proposed=True)
