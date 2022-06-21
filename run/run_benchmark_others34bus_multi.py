from experiments.final_toy_experiments.run_benchmark_others import *
from multiprocessing import Process
from time import sleep
from os import makedirs


if __name__ == '__main__':

	OFOL = '../experiments/34bus_proposed_deg1'
	DICT_CASES = {
		'ieee34bus': {
			'case_folder': '..\cases\ieee34bus',
			'fn_adalasso_weights': None,
			'cost_lasso_x': 0.0,
			'n_sce': 500, 'n_win_ahead': 0, 'n_win_before': 0, 'n_days': 15, 'n_days_delay': 1
		}
	}
	kwargs = {'proposed': True, 'polypoldeg': 1}
	l_processes = []
	count = 0
	for day in range(15,22):
		TIME_CONFIG = {
			'tini': datetime(2018, 7, 27, 0, 0, 0),
			'tiniout': datetime(2018, 8, day, 0, 0, 0),
			'tend': datetime(2018, 8, day, 23, 59, 54),
			'dt': timedelta(seconds=6),
			'n_rh': 14400
		}
		ofol_sub = join(OFOL, '{}'.format(day))

		if not exists(ofol_sub):
			makedirs(ofol_sub)
			
		args = [DICT_CASES, ofol_sub, TIME_CONFIG]
		l_processes.append(
			Process(target=run_benchmark_others, args=args, kwargs=kwargs)
		)
		l_processes[count].daemon = True
		l_processes[count].start()
		count += 1
		
		
	while True:
		c_noalive = 0
		for p in l_processes:
			if not p.is_alive():
				c_noalive += 1
			print('{} alive={}'.format(p.name, p.is_alive()))
		if c_noalive == len(l_processes):
			break
		print('-'*80)
		sleep(60)
		
	print('done!')
	"""
	TIME_CONFIG = {
		'tini': datetime(2018, 7, 27, 0, 0, 0),
		'tiniout': datetime(2018, 8, 15, 0, 0, 0),
		'tend': datetime(2018, 8, 17, 23, 59, 54),
		'dt': timedelta(seconds=6),
		'n_rh': 150
	}
	"""
	

	


		
	
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

	



