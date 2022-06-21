from datetime import timedelta
import numpy as np
import pandas as pd
from global_definitions import *
import data_structure as ds
from front_end_utilities import down_sample_setpoints
from pandas.tseries.frequencies import to_offset
from errors import InputError

# TODO:
#   - Verify that the functions apply when Dt is grater than pdata.df_data
#     (implicitly given by df_sim or time_map(0))


def losses_pu(pdata: ds.Adn, df_sim: pd.DataFrame, v0=complex(1., 0.)):
    # Assertions and preliminaries
    assert df_sim.shape[0] > 1
    assert isinstance(df_sim.index, pd.DatetimeIndex)

    if pdata.A_brf is None or pdata.A_brt is None:
        pdata.make_connectivity_mats()
    l_buses = pdata.l_buses
    l_cols_v = [SNAM_V + str(i) for i in l_buses]
    l_cols_vang = [SNAM_VANG + str(i) for i in l_buses]
    n_t = df_sim.shape[0]
    ii = complex(0, 1)

    r = pdata.branches['r'].values
    x = pdata.branches['x'].values
    z = r + ii * x
    y = 1 / z

    A = (pdata.A_brf - pdata.A_brt).values

    v = df_sim.loc[:, l_cols_v].values * np.exp(ii * df_sim.loc[:, l_cols_vang].values)
    v = np.concatenate([v0 * np.ones((n_t, 1), dtype='complex128'), v], axis=1)

    i = np.diag(y) @ A @ v.transpose()

    losses = r.reshape(1, len(l_buses)) @ np.absolute(i) ** 2

    return losses.flatten()


def utility_p_injection(pdata, df_sim, df_data=None, v0=complex(1., 0.)):
    """
    :param pdata: {data_structure.Adn}
    :param df_sim: {pd.DataFrame}
    :param df_data: {pd.DataFrame}
    :param v0: {complex} Slack bus complex voltage in p.u..
    :return: {np.ndarray} An array of total utility p injection per period, i.e. of shape=
        (df_sim.shape[0],).
    """
    df_data = pdata.df_data.loc[df_sim.index, :] if df_data is None else df_data

    l_cols_loadp = [i for i in df_data.columns if i.startswith(SNAM_LOADP)]

    ar_losses = losses_pu(pdata, df_sim, v0)
    ar_p_injection = p_injection_pu(df_sim)
    ar_loadp = df_data.loc[df_sim.index, l_cols_loadp].sum(axis=1).values
    ar_utility_p_injection = ar_losses.flatten() + ar_loadp - ar_p_injection

    return ar_utility_p_injection


def p_injection_pu(df_result):
    l_cols_dgp = [i for i in df_result.columns if i.startswith(SNAM_DGP)]
    l_cols_b2bp = [i for i in df_result.columns if i.startswith(SNAM_B2BFP)]
    l_cols_b2bp += [i for i in df_result.columns if i.startswith(SNAM_B2BTP)]
    l_cols_p = l_cols_dgp + l_cols_b2bp + l_cols_b2bp
    return df_result.loc[:, l_cols_p].sum(axis=1).values


def accum_losses_mwh(pdata: ds.Adn, df_result: pd.DataFrame, v0=complex(1., 0.)):
    """
    Calculates the total losses in MWh
    :param pdata: {data_structure.Adn}
    :param df_result: {pandas.DataFrame}
    :param v0: {complex}
    :return: total_losses_mwh {float}
    """
    losses = losses_pu(pdata, df_result, v0)
    total_losses = losses.sum()

    Dt = df_result.index[1] - df_result.index[0]
    time_ratio = Dt / timedelta(hours=1)

    total_losses_mwh = time_ratio * total_losses * pdata.sbase_mva

    return total_losses_mwh


def accum_demand_mwh(pdata: ds.Adn, trange=None):
    assert isinstance(pdata.df_data.index, pd.DatetimeIndex)
    if trange is None:
        df_data = pdata.df_data
    else:
        df_data = pdata.df_data.loc[trange, :]
    Dt = pdata.time_config['dt']
    n_hours_per_dt = Dt / timedelta(hours=1)
    l_cols_loadp = [i for i in df_data.columns if i.startswith(SNAM_LOADP)]
    sum_loadp = df_data.loc[:, l_cols_loadp].sum().sum()
    return sum_loadp * n_hours_per_dt


def accum_utility_mwh(pdata: ds.Adn, df_sim, trange=None):
    trange = df_sim.index if trange is None else trange
    dt = pd.to_timedelta(to_offset(pd.infer_freq(df_sim.index)))
    assert dt
    df_data = pdata.df_data.loc[trange, :]
    n_hours_per_dt = dt / timedelta(hours=1)

    return n_hours_per_dt * utility_p_injection(pdata, df_sim, df_data).sum() * pdata.sbase_mva


def vreg_pus(df_result: pd.DataFrame, vr: float = 0.05):
    assert isinstance(df_result.index, pd.DatetimeIndex)
    assert df_result.shape[0] > 1

    Dt = df_result.index[1] - df_result.index[0]
    seconds_per_dt = Dt / timedelta(seconds=1)

    l_cols_v = [i for i in df_result.columns if i.startswith(SNAM_V)]

    df_result_v = df_result[l_cols_v].values
    aux_zeros = np.zeros(df_result_v.shape, dtype='float64')
    accum_vio_ub = np.maximum(aux_zeros, (df_result_v - (1 + vr)))
    accum_vio_lb = np.maximum(aux_zeros, ((1-vr) - df_result_v))

    vreg = (accum_vio_ub + accum_vio_lb).sum() * seconds_per_dt

    return float(vreg)


def nvv(df_result: pd.DataFrame, Dt_interval: timedelta, Vr=0.05):
    assert isinstance(df_result.index, pd.DatetimeIndex)
    assert df_result.shape[0] > 1

    l_cols_v = [i for i in df_result.columns if i.startswith(SNAM_V)]
    df_v = df_result.loc[:, l_cols_v]
    df_v_mean = df_v.groupby((df_v.index - df_v.index[0]) // pd.Timedelta(Dt_interval)).mean()
    n_ub_vio = (df_v_mean > 1 + Vr).sum().sum()
    n_lb_vio = (df_v_mean < 1 - Vr).sum().sum()

    return n_lb_vio + n_ub_vio


def nvv_updown(df_result: pd.DataFrame, Dt_interval: timedelta, Vr=0.05):
    assert isinstance(df_result.index, pd.DatetimeIndex)
    assert df_result.shape[0] > 1

    l_cols_v = [i for i in df_result.columns if i.startswith(SNAM_V)]
    df_v = df_result.loc[:, l_cols_v]
    df_v_mean = df_v.groupby((df_v.index - df_v.index[0]) // pd.Timedelta(Dt_interval)).mean()
    n_ub_vio = (df_v_mean > 1 + Vr).sum().sum()
    n_lb_vio = (df_v_mean < 1 - Vr).sum().sum()

    return n_ub_vio, n_lb_vio


def calculate_dv(df_sim: pd.DataFrame, df_insample: pd.DataFrame):
    """
    Calculates positive and negative voltage deltas
    :param df_sim: Simulation results
    :param df_insample: Window setpoints
    :return: {pd.DataFrame} Voltage differences
    """
    l_cols_v = [i for i in df_sim.columns if i.startswith(SNAM_V)]
    # Down sample df_insample
    dt_sim = pd.to_timedelta(to_offset(pd.infer_freq(df_sim.index)))
    dt_insample = pd.to_timedelta(to_offset(pd.infer_freq(df_insample.index)))
    assert dt_sim is not None
    assert dt_insample is not None

    df_v_insample_down = down_sample_setpoints(df_insample.loc[:, l_cols_v], dt_sim)
    return df_sim.loc[:, l_cols_v] - df_v_insample_down.loc[df_sim.index, :]


def calculate_v_outofbounds(df_sim, vlim_r=0.05):
    l_cols_v = [i for i in df_sim.columns if i.startswith(SNAM_V)]
    ar_v = df_sim.loc[:, l_cols_v].values
    e_pos = np.maximum(0., ar_v - (1. + vlim_r))
    e_neg = - np.minimum(0., ar_v - (1. - vlim_r))
    return pd.DataFrame(e_pos + e_neg, index=df_sim.index, columns=l_cols_v)


def calculate_vv_pu(df_sim, vlim_r=0.05):
    l_cols_v = [i for i in df_sim.columns if i.startswith(SNAM_V)]
    ar_v = df_sim.loc[:, l_cols_v].values
    e_pos = np.maximum(0., ar_v - (1. + vlim_r))
    e_neg = - np.minimum(0., ar_v - (1. - vlim_r))
    return e_pos + e_neg


def calculate_avg_vv_pu(df_sim, vlim_r=0.05):
    l_cols_v = [i for i in df_sim.columns if i.startswith(SNAM_V)]
    ar_v = df_sim.loc[:, l_cols_v].values
    e_pos = np.maximum(0., ar_v - (1. + vlim_r))
    e_neg = - np.minimum(0., ar_v - (1. - vlim_r))
    return (e_pos + e_neg).mean()


# TODO: this does not consider back-to-backs
def calculate_pv_curtailment(pdata: ds.Adn, df_sim: pd.DataFrame):
    l_dgs = pdata.l_dgs
    l_cols_dgp = [SNAM_DGP + str(i) for i in l_dgs]
    l_cols_dgpmax = [SNAM_DGPMAX + str(i) for i in l_dgs]
    trange_sim = df_sim.index
    ar_dgpmax = pdata.df_data.loc[trange_sim, l_cols_dgpmax].values
    ar_dgp = df_sim.loc[trange_sim, l_cols_dgp].values
    assert ar_dgpmax.shape == ar_dgp.shape
    ar_dg_curt = ar_dgpmax - ar_dgp

    # TODO2: fix negative pv_curtailments in ar_df_curt
    #   assert ar_dg_curt.min() >= 0.

    return pd.DataFrame(ar_dg_curt, index=trange_sim, columns=l_dgs)


def calculate_pv_curtailment_ar_pu(pdata: ds.Adn, df_sim: pd.DataFrame):
    l_dgs = pdata.l_dgs
    l_cols_dgp = [SNAM_DGP + str(i) for i in l_dgs]
    l_cols_dgpmax = [SNAM_DGPMAX + str(i) for i in l_dgs]
    trange_sim = df_sim.index
    ar_dgpmax = pdata.df_data.loc[trange_sim, l_cols_dgpmax].values
    ar_dgp = df_sim.loc[trange_sim, l_cols_dgp].values
    assert ar_dgpmax.shape == ar_dgp.shape
    ar_dg_curt = ar_dgpmax - ar_dgp
    return ar_dg_curt.sum(axis=1).flatten()


def avg_pv_curtailment_pu(pdata: ds.Adn, df_sim: pd.DataFrame):
    return calculate_pv_curtailment(pdata, df_sim).mean().mean()


def calculate_cvar(ar_fobj, ar_weights, gamma):
    # Validate input
    if not ((ar_weights.sum() >= 1. - 1e-12) and (ar_weights.sum() <= 1. + 1e-12)):
        raise InputError('ar_weights do not sum 1!')
    if not (len(ar_fobj.shape) == 1):
        raise InputError('ar_fobj must be a 1-dimensional ndarray!')
    if ar_fobj.shape != ar_weights.shape:
        raise InputError('ar_fobj and ar_weights must have the same shape!')

    # Copy and sort
    ar_fobj = np.copy(ar_fobj)
    ar_weights = np.copy(ar_weights)
    idx_sort = np.argsort(ar_fobj)
    ar_fobj = ar_fobj[idx_sort]
    ar_weights = ar_weights[idx_sort]

    # Calculate
    ar_w_accum = ar_weights.cumsum()
    k_gamma = np.argmax(ar_w_accum >= gamma)
    term1 = (ar_w_accum[k_gamma] - gamma) * ar_fobj[k_gamma]
    term2 = (ar_weights[k_gamma + 1:] * ar_fobj[k_gamma + 1:]).sum()
    cvar_value = (1 / (1 - gamma)) * (term1 + term2)

    return cvar_value

