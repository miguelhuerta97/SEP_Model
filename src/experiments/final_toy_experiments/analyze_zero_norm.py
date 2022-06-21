from os.path import join
from mpl_toolkits.axes_grid1 import make_axes_locatable
from global_definitions import *
import gc
import pandas as pd
from datetime import datetime, timedelta
from post_process.visualization import init_figure
import numpy as np
import matplotlib.pyplot as plt
from front_end_utilities import load_df_result


OFOL = '/home/jp/tesis/experiments/final/plots'

dict_D_types = {
    SNAM_DLPP: [],
    SNAM_DLPQ: [],
    SNAM_DLQP: [],
    SNAM_DLQQ: [],
    SNAM_DPP: [],
    SNAM_DPQ: []
}


FN_DGS = '/home/jp/tesis/experiments/cases_final/ieee123/ieee123bus_dgs.csv'
FN_INS = '/home/jp/tesis/experiments/final/ieee123bus/proposed/df_ins_proposed.csv'

df_dgs = pd.read_csv(FN_DGS, sep='\t', index_col='index')
l_dg_buses = df_dgs.loc[:, 'bus'].to_list()
l_degs = list(range(4))

l_cols_D = [k + '_' + str(i) + '_' + str(j) for k in dict_D_types.keys() for i in l_dg_buses for j in l_degs]


df_D = load_df_result(FN_INS)

tini = datetime(2018,8,15,0,0,0)
tend = datetime(2018,8,21,0,0,0)
trange = pd.date_range(start=tini, end=tend, freq=timedelta(days=1))

df_D = df_D.loc[trange, l_cols_D]


gc.collect()


print('caca')
dict_D_types = {
    SNAM_DLPP: [],
    SNAM_DLPQ: [],
    SNAM_DLQP: [],
    SNAM_DLQQ: [],
    SNAM_DPP: [],
    SNAM_DPQ: []
}

for i in l_cols_D:
    for k in dict_D_types:
        if i.startswith(k):
            dict_D_types[k].append(i)

dict_norm_zero = dict.fromkeys(dict_D_types)

idx_mul = pd.MultiIndex.from_product([dict_D_types.keys(), range(4)])

se_ret = pd.Series(index=idx_mul)

EPS_FLOAT = 1.e-4

for k, l_cols in dict_D_types.items():
    df_iter = df_D.loc[:, l_cols]


    for d in range(4):
        l_cols_degree = [i for i in df_iter.columns if i.endswith(str(d))]

        df_iter_iter = df_iter.loc[:, l_cols_degree]
        n = df_iter_iter.shape[0] * df_iter_iter.shape[1]

        df_idx_non_zeros = df_iter_iter.abs() > EPS_FLOAT

        se_ret.loc[(k, d)] = df_idx_non_zeros.sum().sum() / n


print(se_ret.unstack(1))
df_matrix_ret = se_ret.unstack(1) * 100.


fig, ax = init_figure(0.45*1.05*1.05, 0.5*1.05*1.05)

name_mapper = {'DLpp': r'$\mathbf{p}_{\text{G}} \gets \mathbf{p}_{\text{L}}$',
               'DLpq': r'$\mathbf{p}_{\text{G}} \gets \mathbf{q}_{\text{L}}$',
               'DLqq': r'$\mathbf{q}_{\text{G}} \gets\mathbf{q}_{\text{L}}$',
               'DLqp': r'$\mathbf{q}_{\text{G}} \gets \mathbf{p}_{\text{L}}$',
               'DPp': r'$\mathbf{p}_{\text{G}} \gets \mathbf{\hat{p}}_{\text{G}}$',
               'DPq': r'$\mathbf{q}_{\text{G}} \gets \mathbf{\hat{p}}_{\text{G}}$',
               }
df_matrix_ret.rename(index=name_mapper, inplace=True)

im = ax.matshow(df_matrix_ret.values, alpha=0.7) # cmap=plt.cm.Blues
ax.set_xticklabels([''] + [str(i) for i in l_degs])
ax.set_yticklabels([''] + df_matrix_ret.index.to_list())

ax.set_xlabel('Monomial degree')
ax.set_ylabel('Related variables')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)


for i in range(6):
    for j in range(4):
        ax.text(j, i, '{}'.format(int(df_matrix_ret.values[i, j])), va='center', ha='center')
#for i in range(3):
#    for j in range(4):
#        c = df_matrix_ret.values[i, j]
#        ax.text(i, j, str(c), va='center', ha='center')

fig.colorbar(im, cax=cax)
fig.tight_layout()

fig.show()



fig.savefig(join(OFOL, 'zero_norm.svg'), format='svg')
