from experiments.final_toy_experiments.analyze_benchmark import df_metric_to_latex, make_pdistribution_plots
from os.path import join, exists
from os import mkdir

IFOL = r'../experiments/final_123bushigh'

DICT_CASE_FOLDER = {
	'ieee123bus': '../cases/ieee123bus_high'
}
OFOL ='plots'
if not exists(OFOL):
    mkdir(OFOL)
print(df_metric_to_latex(IFOL, DICT_CASE_FOLDER, ofol_ext=OFOL))
make_pdistribution_plots(OFOL, l_cases=['ieee123bus'],
                         l_strategies=['proposed', 'kyri', 'linkfailure']) # 'ieee1547'


