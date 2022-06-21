"""
Clement-Nyns, K., Haesen, E., & Driesen, J. (2010). The Impact of Charging Plug-In Hybrid Electric
Vehicles on a Residential Distribution Grid. IEEE Transactions on Power Systems, 25(1), 371–380.
https://doi.org/10.1109/TPWRS.2009.2036481

Simulating uncontrolled domestic charging scenario (worst scenario)
"""
import numpy as np
from scipy.stats import lognorm, truncnorm
import pandas as pd
from global_definitions import *

# Minute based quantities
PDF_START_CHARG_MEAN = 1260
PDF_START_CHARG_LB = -60*3
PDF_START_CHARG_UB = 60*3
PDF_START_CHARG_STD = 60

PDF_DRIVEN_LB = 0
PDF_DRIVEN_MEAN = 22.5
PDF_DRIVEN_STD = 12.2

N_EVS = 1000

ALPHA = 2.
DR = 80.

BATTERY_PROF_TRAPA = 15  # [min]
BATTERY_PROF_TRAPB = 255  # [min]
BATTERY_PROF_TRAPC = 30  # [min]


default_config = {
    'start_charg_mean': PDF_START_CHARG_MEAN,
    'start_charg_lb': PDF_START_CHARG_LB,
    'start_charg_ub': PDF_START_CHARG_UB,
    'start_charg_std': PDF_START_CHARG_STD,
    'driven_mean': PDF_DRIVEN_MEAN,
    'driven_std': PDF_DRIVEN_STD,
    'n_evs': N_EVS,
    'alpha': ALPHA,
    'dR': DR,
    'battery_prof_trapa': BATTERY_PROF_TRAPA,
    'battery_prof_trapb': BATTERY_PROF_TRAPB,
    'battery_prof_trapc': BATTERY_PROF_TRAPC
}


# ISSUE:
#   - All generated samples are almost equal
#   - Only minute-sampled
def ev_demand(n_days, tini, n_samples=None, l_buses_loads=None, config=None):
    # Preamble: args assertions
    assert n_samples is not None or l_buses_loads is not None

    if n_samples is not None:
        l_buses_loads = range(n_samples)

    if config is not None:
        assert default_config.keys() <= config.keys()
    else:
        config = default_config

    n_evs = int(config['n_evs'])
    l_cols_loadp = [SNAM_LOADP + str(i) for i in l_buses_loads]
    trange = pd.date_range(start=tini, freq='T', periods=n_days * 1440)

    df_ev_loadp = pd.DataFrame(index=trange, columns=l_cols_loadp)

    # Generating random start times
    ar_start_charg_noise = truncnorm.rvs(
        config['start_charg_lb'],
        config['start_charg_ub'],
        loc=0.,
        scale=config['start_charg_std'],
        size=n_days * n_evs
    ).astype('int64').reshape(n_days, n_evs)

    ar_start_charg_mean = np.asarray(
        [[1440*i + config['start_charg_mean'] for i in range(n_days)]] * n_evs,
        dtype='int64'
    ).transpose()

    ar_start_charg = ar_start_charg_mean + ar_start_charg_noise

    # Generating dayly driven distance
    sigma = float(config['driven_std'])
    mhu = float(config['driven_mean'])

    mhu_log = np.log(mhu ** 2 / np.sqrt(sigma ** 2 + mhu ** 2))
    sigma_log = np.log(1 + (sigma ** 2 / mhu ** 2))

    s = sigma_log
    scale = np.exp(mhu_log)

    d = lognorm.rvs(scale=scale, s=s, size=n_days * n_evs)

    alpha = config['alpha']
    dR = config['dR']
    assert d.max() < dR

    s_Eini = (1 - alpha * d / dR)
    while s_Eini.min() < 0.:
        l_neg = s_Eini < 0.
        n_neg = round(l_neg.sum())
        s_Eini[l_neg] = (1 - alpha * lognorm.rvs(scale=scale, s=s, size=n_neg) / dR)

    s_Eini = s_Eini.reshape(n_evs, n_days)

    # Setting deterministic battery charge profile
    trap_a = 15
    trap_c = 30
    trap_b = 255
    n_profile_original = trap_a + trap_b + trap_c

    x = np.linspace(0, n_profile_original + 1, n_profile_original)
    original_profile = np.piecewise(
        x, [x < trap_a, np.logical_and(x >= 15, x <= trap_a + trap_b), x > trap_a + trap_b],
        [lambda xx: (1. / trap_a) * xx, lambda xx: 1.,
         lambda xx: 1. - (1. / (trap_c - 1)) * (xx - (trap_a + trap_b))]
    )

    Atot = 0.5 * (trap_a + trap_c) + trap_b

    s_delta_b = (Atot * s_Eini).astype(int)
    assert s_delta_b.max().max() < 255

    for k in l_cols_loadp:
        ev_load = np.zeros(shape=((n_days + 1) * 1440,), dtype=float)

        for i in range(n_evs):
            for j in range(n_days):
                yprofile = np.concatenate(
                    [
                        original_profile[:trap_a + 1],
                        original_profile[trap_a + s_delta_b[i, j]:]
                    ]
                )
                n_profile = yprofile.shape[0]
                ev_load[ar_start_charg[j, i]: ar_start_charg[j, i] + n_profile] += yprofile
        df_ev_loadp.loc[:, k] = ev_load[:1440 * n_days]

    df_ev_loadp = df_ev_loadp / df_ev_loadp.max(axis=0)
    return df_ev_loadp
