from experiments.final_toy_experiments.run_benchmark_others import *
from os import makedirs

OFOL = '../experiments/123bus_benchmark_others_high'
if not exists(OFOL):
	makedirs(OFOL)

"""
TIME_CONFIG = {
    'tini': datetime(2018, 7, 27, 0, 0, 0),
    'tiniout': datetime(2018, 8, 15, 0, 0, 0),
    'tend': datetime(2018, 8, 17, 23, 59, 54),
    'dt': timedelta(seconds=6),
    'n_rh': 150
}
"""
TIME_CONFIG = {
        'tini': datetime(2018, 7, 27, 0, 0, 0),
        'tiniout': datetime(2018, 8, 15, 0, 0, 0),
        'tend': datetime(2018, 8, 15, 23, 59, 54),
        'dt': timedelta(seconds=6),
        'n_rh': 150
    }

DICT_CASES = {
	'ieee123bus': {
		'case_folder': '..\cases\ieee123bus_high',
		'fn_adalasso_weights': None,
		'cost_lasso_x': 0.0,
		'n_sce': 500, 'n_win_ahead': 3, 'n_win_before': 3, 'n_days': 15, 'n_days_delay': 1
	}


    
}
"""
'ieee34bus': {
            'case_folder': '/home/jp/tesis/experiments/cases_final/ieee34bus',
            'fn_adalasso_weights': '/home/jp/tesis/experiments/34bus_lasso/08_18_benchmark/rob_kme/df_ins_ieee34bus_rob_kme_cx0.0.csv',
            'cost_lasso_x': 3.684031498640386e-12,
            'n_sce': 500, 'n_win_ahead': 3, 'n_win_before': 3, 'n_days': 15, 'n_days_delay': 1
            }
"""


"""
    'ieee4bus': {
        'case_folder': '/home/jp/tesis/experiments/cases_final/ieee4bus',
        'fn_adalasso_weights': None,
        'cost_lasso_x': 3.684031498640386e-12,
        'n_sce': 100, 'n_win_ahead': 3, 'n_win_before': 3, 'n_days': 15, 'n_days_delay': 1
    }
"""

run_benchmark_others(DICT_CASES, OFOL, TIME_CONFIG, ieee1547=True)



