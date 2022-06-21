from experiments.final_toy_experiments.run_lasso_determination import *

# ------------------------------------------------------------------------------------------------ #
# Input
# ------------------------------------------------------------------------------------------------ #
CASE_FOLDER = '..\cases\ieee123bus_high'
CASE_NAME = 'ieee123bus'
OFOL = r'..\experiments\exp_benchmark_prelim123bus_nrh1day_high'

AR_LASSO_COEFF = [1.e-12]
BILINEAR_APPROX = 'mccormick'

FN_ADAWEIGHTS = r'..\experiments\exp_benchmark_prelim123bus_nrh1day_high\rob_kme\df_ins_ieee123bus_rob_kme_cx0.0.csv'

TIME_CONFIG = {
	'tini': datetime(2018, 7, 27, 0, 0, 0),
	'tiniout': datetime(2018, 8, 15, 0, 0, 0),
	'tend': datetime(2018, 8, 15, 23, 59, 54),
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