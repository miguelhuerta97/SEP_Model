from experiments.final_toy_experiments.run_lasso_determination import *

# ------------------------------------------------------------------------------------------------ #
# Input
# ------------------------------------------------------------------------------------------------ #
CASE_FOLDER = '..\cases\ieee34bus'
CASE_NAME = 'ieee34bus'
OFOL = '..\experiments\exp_benchmark_prelim34bus_1week_nrh1day'

AR_LASSO_COEFF = [0.]
BILINEAR_APPROX = 'mccormick'

FN_ADAWEIGHTS = None

TIME_CONFIG = {
	'tini': datetime(2018, 7, 27, 0, 0, 0),
	'tiniout': datetime(2018, 8, 15, 0, 0, 0),
	'tend': datetime(2018, 8, 21, 23, 59, 54),
	'dt': timedelta(seconds=6),
	'n_rh': 14400
}

# ------------------------------------------------------------------------------------------------ #
# Set io
# ------------------------------------------------------------------------------------------------ #
if not exists(OFOL):
	mkdir(OFOL)

# ------------------------------------------------------------------------------------------------ #
# Run experiment and write reports
# ------------------------------------------------------------------------------------------------ #

run_lasso_determination(
	case_folder=CASE_FOLDER, case_name=CASE_NAME,
	ofol=OFOL,
	ar_lasso_coeff=AR_LASSO_COEFF,
	n_sce=500, n_win_ahead=0, n_win_before=0, n_days=15, n_days_delay=0,
	bilinear_approx=BILINEAR_APPROX,
	time_config=TIME_CONFIG,
	fn_adalasso_weights=FN_ADAWEIGHTS
)