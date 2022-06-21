"""
NREL. (2020). PVDAQ (PV Data Acquisition). Retrieved September 30, 2020, from
https://developer.nrel.gov/docs/solar/pvdaq-v3/

1-minute based active power supplied by
system id:1423
"""
import numpy as np
import pandas as pd
from global_definitions import SNAM_DGPMAX
from case_generation.ts_base_demand import load_ts_data


def ts_dgpmax(folder_name, fname_pattern, seed_field_name='dgpmax',
              seed_index_name='time'):

    df_seed_dgpmax = load_ts_data(folder_name, fname_pattern, seed_field_name,
                                  index_col=seed_index_name)

    return df_seed_dgpmax





