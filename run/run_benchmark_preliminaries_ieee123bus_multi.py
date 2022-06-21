from experiments.final_toy_experiments.run_lasso_determination import *
from multiprocessing import Process
from time import sleep
from os import makedirs

# ------------------------------------------------------------------------------------------------ #
# Input
# ------------------------------------------------------------------------------------------------ #
if __name__ == '__main__':
	CASE_FOLDER = '..\cases\ieee123bus'
	CASE_NAME = 'ieee123bus'
	OFOL = '..\experiments\exp_benchmark_prelim123bus_nrh1day'

	AR_LASSO_COEFF = [1e-14]
	BILINEAR_APPROX = 'mccormick'

	FN_ADAWEIGHTS = None

	l_processes = []
	count = 0
	day = 15
	for hour in range(0, 4):
		TIME_CONFIG = {
			'tini': datetime(2018, 7, 27, 0, 0, 0),
			'tiniout': datetime(2018, 8, 15, hour, 0, 0),
			'tend': datetime(2018, 8, 15, hour, 59, 54),
			'dt': timedelta(seconds=6),
			'n_rh': 150
		}
		ofol_sub = join(OFOL, '{}'.format(hour))
		# ------------------------------------------------------------------------------------------------ #
		# Set io
		# ------------------------------------------------------------------------------------------------ #
		if not exists(ofol_sub):
			makedirs(ofol_sub)

		# ------------------------------------------------------------------------------------------------ #
		# Run experiment and write reports
		# ------------------------------------------------------------------------------------------------ #
		kwargs = {
			'case_folder':CASE_FOLDER, 'case_name':CASE_NAME,
			'ofol':ofol_sub,
			'ar_lasso_coeff':AR_LASSO_COEFF,
			'n_sce':600, 'n_win_ahead':3, 'n_win_before':3, 'n_days':15, 'n_days_delay':1,
			'bilinear_approx':BILINEAR_APPROX,
			'time_config':TIME_CONFIG,
			'fn_adalasso_weights':FN_ADAWEIGHTS
		}
		l_processes.append(Process(target=run_lasso_determination, kwargs=kwargs, name='Proc. day={}'.format(day)))
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