import sys, os
sys.path.append(os.path.normpath('/home/miguel/jp_sep/src/'))
from experiments.final_toy_experiments.run_lasso_determination import *
from config import *


# ------------------------------------------------------------------------------------------------ #
# Input
# ------------------------------------------------------------------------------------------------ #
CASE_NAME   = 'ieee4bus'
CASE_FOLDER = '/home/miguel/jp_sep/cases/ieee4bus'
OFOL        = '/home/miguel/jp_sep/results_experiments/ieee4bus/exp_benchmark_prelim_4bus_1week_nrh1day'

AR_LASSO_COEFF = [0.]
BILINEAR_APPROX = 'bilinear'

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
    n_sce=100, n_win_ahead=0, n_win_before=0, n_days=15, n_days_delay=0,
    bilinear_approx=BILINEAR_APPROX,
    time_config=TIME_CONFIG,
    fn_adalasso_weights=FN_ADAWEIGHTS
)