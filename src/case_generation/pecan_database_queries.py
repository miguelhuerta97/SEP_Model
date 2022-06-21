import pandas as pd
import psycopg2
import gc
import os
from os.path import exists, join
import datetime as dt
import numpy.random as nr


OFOL = '/home/jp/tesis/data_seed/pecan_1min_by_user_solar_NaNvals'
if not exists(OFOL):
    os.mkdir(OFOL)

connection = psycopg2.connect(
    host='localhost',  # host on which the database is running
    database='pecan',  # name of the database to connect to
    user='jp',  # username to connect with
    password='jp'  # password associated with your username
)

cursor = connection.cursor()
cursor.execute('select distinct id FROM data1;')
l_users = list(cursor.fetchall())
l_users = [i[0] for i in l_users]
cols = ['localminute', 'grid', 'solar']
n_cols = len(cols)
assert n_cols > 1
l_incomplete = []
#l_discarded = [661, 6139, 3039]
l_discarded = []
df_tbounds = pd.DataFrame(index=list(set(l_users) - set(l_discarded)), columns=['tini', 'tend'])
for i in l_users:
    print('solving user {}'.format(i))
    if i in l_discarded:
        continue
    cursor.execute(
        ('select ' + (n_cols - 1) * '{}, ' + '{} ' + 'from data1 where id = cast({} as integer)').format(*cols, i))
    userdata = cursor.fetchall()
    df_userdata = pd.DataFrame([list(j) for j in userdata],
                               columns=['time', 'loadp_kw', 'dgpmax_kw'])
    df_userdata.sort_values('time', inplace=True)
    df_userdata.set_index('time', drop=True, inplace=True)
    df_userdata.drop(df_userdata.index[df_userdata.index.duplicated()], inplace=True)

    tini = dt.datetime(2018, 1, 1, 0, 0, 0)
    tend = dt.datetime(2018, 12, 31, 23, 59, 0)
    trange = pd.date_range(start=tini, end=tend, freq='T')

    df_out = pd.DataFrame(index=trange, columns=df_userdata.columns)
    df_out.loc[df_userdata.index, :] = df_userdata

    df_out.to_csv(join(OFOL, 'load_kw_{}.csv'.format(i)), index_label='time')

    gc.collect()

connection.close()