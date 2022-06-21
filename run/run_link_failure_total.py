from os.path import join
from datetime import datetime, timedelta
from experiments.final_toy_experiments.run_link_failure import run_link_failure_total

# ------------------------------------------------------------------------------------------------ #
# Input
# ------------------------------------------------------------------------------------------------ #
CASE_FOLDER = '..\cases\ieee34bus'
CASE_NAME = 'ieee34bus'
OFOL = '..\experiments\link_failure_total_1week'
FN_ADALASSO_WEIGHTS = '..\experiments\link_failure_total_1week\lasso0.0\df_ins_link_failure_all.csv' #'..\experiments\lasso_determination_3days\df_ins_ieee34bus_rob_kme_cx0.0.csv'
LOGFOL = join(OFOL, 'log')
COST_LASSO_X = 1.e-9  #0.0 #3.684031498640386e-12

TIME_CONFIG = {
        'tini': datetime(2018, 7, 27, 0, 0, 0),
        'tiniout': datetime(2018, 8, 15, 0, 0, 0),
        'tend': datetime(2018, 8, 21, 23, 59, 54),
        'dt': timedelta(seconds=6),
        'n_rh': 14400*7
    }

N_SCE = 500
N_DAYS = 2
N_WIN_AHEAD = 0
N_WIN_BEFORE = 0
N_DAYS_DELAY = 0

# ------------------------------------------------------------------------------------------------ #
# Run experiment
# ------------------------------------------------------------------------------------------------ #
run_link_failure_total(
    ofol=OFOL,
    logfol=LOGFOL,
    case_folder=CASE_FOLDER,
    case_name=CASE_NAME,
    fn_adalasso_weights=FN_ADALASSO_WEIGHTS,
    cost_lasso_x=COST_LASSO_X,
    time_config=TIME_CONFIG,
    n_sce=N_SCE,
    n_days=N_DAYS,
    n_win_ahead=N_WIN_AHEAD,
    n_win_before=N_WIN_BEFORE,
    n_days_delay=N_DAYS_DELAY,
)