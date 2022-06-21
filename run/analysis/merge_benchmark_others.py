from front_end_utilities import write_df_result
from os.path import join, exists
from os import makedirs
from experiments.final_toy_experiments.analyze_benchmark import merge_results_list


PREFIX_DF_SIM = 'df_sim'
PREFIX_DF_INS = None #'df_ins'


IFOL = r'..\..\experiments\123bus_factor2_benchmark_others_4hourssteps'
OFOL = join(IFOL, 'merged_results')
CASE_NAME = 'ieee123bus'


STRATEGY = 'ieee1547'

if not exists(OFOL):
	makedirs(OFOL)

df_sim_name = PREFIX_DF_SIM + '_' + STRATEGY + '.csv'
if PREFIX_DF_INS is not None:
	df_ins_name = PREFIX_DF_INS + '_' + STRATEGY + '.csv'

l_fn_ins = []
l_fn_sim = []

for i in range(6):
	exp_fol = join(join(IFOL, str(i)), CASE_NAME)
	
	l_fn_sim.append(join(exp_fol, df_sim_name))
	if PREFIX_DF_INS is not None:
		l_fn_ins.append(join(exp_fol, df_ins_name))


df_sim = merge_results_list(l_fn_sim)
if PREFIX_DF_INS is not None:
	df_ins = merge_results_list(l_fn_ins)

# Write merged results
fn_sim = join(OFOL, df_sim_name)
write_df_result(df_sim, fn_sim)
if PREFIX_DF_INS is not None:
	fn_ins = join(OFOL, df_ins_name)
	write_df_result(df_ins, fn_ins)
