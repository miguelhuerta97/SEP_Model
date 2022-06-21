from post_process.post_process import test_ieee1547_curtailment

CASE_FOLDER = '../../cases/ieee123bus_moderate'
CASE_NAME = 'ieee123bus'

FN_SIM = r'..\..\experiments\experiments\123bus_moderate_benchmark_others_4hourssteps\final_format\ieee123bus\ieee1547\df_sim_ieee1547.csv'

print(test_ieee1547_curtailment(CASE_FOLDER, CASE_NAME, FN_SIM))