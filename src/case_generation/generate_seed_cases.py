# TODO: The input should be an instance of data_structure.Adn
import numpy as np
from datetime import timedelta, datetime
from case_generation.ts_base_demand import proc_nan_data, load_ts_data, reactive_power
from os.path import exists, join
from os import mkdir
from os import listdir
import pandas as pd
from global_definitions import SNAM_LOADP_KW, SNAM_DGPMAX_KW, SNAM_LOADP, SNAM_LOADQ, SNAM_DGPMAX, SNAM_LOADDGP

# Input
IFOL = '/home/jp/tesis/data_seed/pecan_1_min_proc_cont_chunks_noisy'
OFOL = '/home/jp/tesis/data_seed/pecan_case_specific_chunks_noisy'
if not exists(OFOL):
    mkdir(OFOL)

rnd_seed = 1
np.random.seed(rnd_seed)

map_case2nbuses = {'ieee4bus': 3}  # 'ieee34bus': 33, 'ieee123bus': 124 {'bw33': 33, '4bus_dist': 3}

# Body
l_seed_files = listdir(IFOL)

for name, n in map_case2nbuses.items():
    ofol_case = join(OFOL, name)
    if not exists(ofol_case):
        mkdir(ofol_case)

    l_cols_loadp = [SNAM_LOADP + str(i) for i in range(1, n + 1)]
    l_cols_loadq = [SNAM_LOADQ + str(i) for i in range(1, n + 1)]
    l_cols_dgpmax = [SNAM_DGPMAX + str(i) for i in range(1, n + 1)]
    l_cols_loaddgp = [SNAM_LOADDGP + str(i) for i in range(1, n + 1)]
    l_data_columns = l_cols_loadp + l_cols_loadq + l_cols_dgpmax + l_cols_loaddgp

    for ifile in l_seed_files:
        n_seed = int(ifile.split('_')[-1].split('.')[0])

        df_seed = pd.read_csv(join(IFOL, ifile), delimiter='\t', index_col='time')
        df_seed.index = pd.to_datetime(df_seed.index)
        assert df_seed.shape[1] % 2 == 0
        m = df_seed.shape[1] // 2

        df_data = pd.DataFrame(index=df_seed.index, columns=l_data_columns, dtype='float64')

        l_cols_seed_loadp = [i for i in df_seed.columns if i.startswith(SNAM_LOADP_KW)]
        l_cols_seed_dgpmax = [i for i in df_seed.columns if i.startswith(SNAM_DGPMAX_KW)]

        weights = np.random.rand(m, n)

        df_data.loc[:, l_cols_loadp] = (df_seed.loc[:, l_cols_seed_loadp].values @ weights)
        df_data.loc[:, l_cols_loadq] = (reactive_power(df_data.loc[:, l_cols_loadp], 0.95)).values
        df_data.loc[:, l_cols_loaddgp] = (df_seed.loc[:, l_cols_seed_dgpmax] @ weights).values
        df_data.loc[:, l_cols_dgpmax] = (df_seed.loc[:, l_cols_seed_dgpmax] @ weights).values

        df_data.to_csv(join(ofol_case, '{}_{}_data.csv'.format(name, n_seed)), sep='\t',
                       index_label='time')
