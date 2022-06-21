from experiments.final_toy_experiments.run_lasso_determination import *
from numpy import logspace
# -------------------------------------------------------------------------------------------- #
# Input
# -------------------------------------------------------------------------------------------- #
CASE_FOLDER = '..\cases\ieee34bus'
CASE_NAME = 'ieee34bus'
OFOL = '..\experiments\lasso_determination_nrh1day_3days'

BILINEAR_APPROX = 'mccormick'
FN_ADAWEIGHTS = r'..\experiments\lasso_determination_nrh1day_3days\rob_kme\df_ins_ieee34bus_rob_kme_cx0.0.csv'
AR_LASSO_COEFF = [10**-10.25, 10**-9.75] #[10**-7]#[10**-7.5] logspace(-9., -8., num=3) #[1.e-11, 1.e-13, 1e-15, 1e-14]

TIME_CONFIG = {
    'tini': datetime(2018, 7, 27, 0, 0, 0),
    'tiniout': datetime(2018, 8, 15, 0, 0, 0),
    'tend': datetime(2018, 8, 17, 23, 59, 54),
    'dt': timedelta(seconds=6),
    'n_rh': 14400
}
# -------------------------------------------------------------------------------------------- #
# Set io
# -------------------------------------------------------------------------------------------- #
if not exists(OFOL):
    mkdir(OFOL)

# -------------------------------------------------------------------------------------------- #
# Run experiment and write reports
# -------------------------------------------------------------------------------------------- #
run_lasso_determination(
    case_folder=CASE_FOLDER, case_name=CASE_NAME,
    ofol=OFOL,
    ar_lasso_coeff=AR_LASSO_COEFF,
    n_sce=500, n_win_ahead=0, n_win_before=0, n_days=15, n_days_delay=0,
    bilinear_approx=BILINEAR_APPROX,
    time_config=TIME_CONFIG,
    fn_adalasso_weights=FN_ADAWEIGHTS
)
