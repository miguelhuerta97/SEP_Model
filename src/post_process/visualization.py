from cycler import cycler
from global_definitions import *
import numpy as np
import pandas as pd
import seaborn as sns
from post_process.performance_analyzer import losses_pu, p_injection_pu, utility_p_injection
from errors import InputError

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()



plt.rcParams.update({
    "text.usetex": True,
    'text.latex.preamble': [r'\usepackage{amsmath}', r'\usepackage{bm}',
                            r'\newcommand{\mb}{\mathbf}'],
    "font.family": "serif",
    "lines.linewidth": 0.8
    })



LETTER_WIDTH = 8.5  # [in]
LETTER_HEIGHT = 11  # [in]
CM2INCH = 0.393701
TEXT_WIDTH_MONO = (LETTER_WIDTH - 8 * CM2INCH)
TEXT_HEIGHT = (LETTER_HEIGHT - 6.5 * CM2INCH)


def init_figure(rel_width, rel_height):
    fig, axs = plt.subplots()
    fig.set_size_inches(TEXT_WIDTH_MONO * rel_width, TEXT_HEIGHT * rel_height)
    fig.set_dpi(300)
    return fig, axs


def plot_loadp(ax, df_data):
    l_cols_loadp = [i for i in df_data.columns if i.startswith(SNAM_LOADP)]

    for i in l_cols_loadp:
        ax.plot(df_data.index, df_data.loc[:, i], label=i)


def plot_voltages_maxmin(ax, df_sim):
    l_cols_v = [i for i in df_sim.columns if i.startswith(SNAM_V)]
    se_max = df_sim.loc[:, l_cols_v].max(axis=1)
    se_min = df_sim.loc[:, l_cols_v].min(axis=1)

    ax.plot(se_max.index, se_max)
    ax.plot(se_max.index, se_min)


def plot_loaddgp(ax, df_data):
    l_cols_loaddgp = [i for i in df_data.columns if i.startswith(SNAM_LOADDGP)]

    for i in l_cols_loaddgp:
        ax.plot(df_data.index, df_data.loc[:, i], label=i)


def plot_dgp(ax, df_sim):
    l_cols_dgp = [i for i in df_sim.columns if i.startswith(SNAM_DGP)]

    for i in l_cols_dgp:
        ax.plot(df_sim.index, df_sim.loc[:, i], label=i)


def plot_dgq(ax, df_sim):
    l_cols_dgq = [i for i in df_sim.columns if i.startswith(SNAM_DGQ)]

    for i in l_cols_dgq:
        ax.plot(df_sim.index, df_sim.loc[:, i], label=i)


def plot_dgpmax(ax, df_data):
    l_cols_dgpmax = [i for i in df_data.columns if i.startswith(SNAM_DGPMAX)]

    for i in l_cols_dgpmax:
        ax.plot(df_data.index, df_data.loc[:, i], label=i)


def plot_loadev(ax, df_data):
    l_cols_loadev = [i for i in df_data.columns if i.startswith(SNAM_LOADEVP)]

    for i in l_cols_loadev:
        ax.plot(df_data.index, df_data.loc[:, i], label=i)


def plot_utility_injection(ax, pdata, df_sim, df_data=None, v0=complex(1., 0.)):
    ax.plot(df_sim.index, utility_p_injection(pdata, df_sim, df_data, v0), label='utility p inj.')


def plot_debug_panel(pdata, df_sim, df_data=None, Vr=0.05, v0=complex(1., 0.), xformatter='%d-%H',
                     trange_ticker=None, title=None):
    df_data = pdata.df_data.loc[df_sim.index, :] if df_data is None else df_data
    assert df_sim.shape[0] == df_data.shape[0]

    fig, axs = plt.subplots(4, 2)
    rel_width = 2.1
    rel_height = 1.1
    fig.set_size_inches(TEXT_WIDTH_MONO * rel_width, TEXT_HEIGHT * rel_height)
    fig.set_dpi(200)

    # Plot voltages
    l_cols_v = [i for i in df_sim.columns if i.startswith(SNAM_V)]
    for i in l_cols_v:
        axs[0, 0].plot(df_sim.index, df_sim.loc[:, i], label=i)
    max_v = df_sim.loc[:, l_cols_v].max().max()
    min_v = df_sim.loc[:, l_cols_v].min().min()

    ub_v_deviation = max(max_v + 0.01, abs(v0) + Vr + 0.01) - abs(v0)
    lb_v_deviation = abs(v0) - min(min_v - 0.01, abs(v0) - Vr - 0.01)
    v_deviation = max(ub_v_deviation, lb_v_deviation)
    v_deviation = np.ceil(v_deviation * 100) / 100

    axs[0, 0].set_ylim([abs(v0) - v_deviation, abs(v0) + v_deviation])
    axs[0, 0].set_title('Voltages')
    axs[0, 0].set_ylabel('$V$')

    axs[0, 0].axhline(1. + Vr, color='r', ls='--')
    axs[0, 0].axhline(1. - Vr, color='r', ls='--')

    plot_voltages_maxmin(axs[0, 1], df_sim)
    axs[0, 1].set_ylim([abs(v0) - v_deviation, abs(v0) + v_deviation])
    axs[0, 1].set_title('Voltages bounds')
    axs[0, 1].set_ylabel('$V$')

    axs[0, 1].axhline(1. + Vr, color='r', ls='--')
    axs[0, 1].axhline(1. - Vr, color='r', ls='--')

    # Plot solar power injection
    l_cols_dgpmax = [i for i in df_data.columns if i.startswith(SNAM_DGPMAX)]
    max_dgp = df_data.loc[:, l_cols_dgpmax].max().max()
    max_dgp = np.ceil(max_dgp * 1.02 * 1000) / 1000

    plot_dgp(axs[1, 0], df_sim)
    axs[1, 0].set_ylim([0., max_dgp])
    axs[1, 0].set_ylabel('$p_{DG}$')
    axs[1, 0].set_title('PV active power injection')

    plot_dgpmax(axs[1, 1], df_data)
    axs[1, 1].set_ylim([0., max_dgp])
    axs[1, 1].set_title('PV active power availability')
    axs[1, 1].set_ylabel('$\hat{P}_{DG}$')

    # Plot reactive power injection
    plot_dgq(axs[2, 0], df_sim)
    axs[2, 0].set_title('DG reactive power injection')
    axs[2, 0].set_ylabel('$q_{DG}$')

    # Plot utility power injection
    plot_utility_injection(axs[3, 0], pdata, df_sim, df_data, v0)
    axs[3, 0].set_title('Utility power injection')
    axs[3, 0].set_ylabel('$p_{UT}$')

    # Plot total load
    plot_loadp(axs[3, 1], df_data)
    axs[3, 1].set_title('Loads active power')
    axs[3, 1].set_ylabel('$p_L$')

    # Plot loadev
    plot_loadev(axs[2, 1], df_data)
    axs[2, 1].set_title('Electric vehicle load')
    axs[2, 1].set_ylabel('$P_{EV}$')

    if trange_ticker is None:
        trange_ticker = pd.date_range(
            start=df_sim.index[0],
            periods=8,
            end=df_sim.index[-1]
        )

    for ax2 in axs:
        for ax in ax2:
            ax.xaxis_date()
            ax.xaxis.set_major_formatter(mdates.DateFormatter(xformatter))
            ax.set_xticks(trange_ticker)
            ax.set_xlabel('Time {}'.format(xformatter))
            ax.grid()

    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()
    return fig, axs


def plot_benchmark_vlim(df_metric, title, ylims=None):
    # Plot init and config
    bar_width = 0.1
    rel_width = 1.0
    rel_height = 0.4
    fig, axs = plt.subplots()
    fig: plt.Figure
    axs: plt.Axes
    fig.set_size_inches(TEXT_WIDTH_MONO * rel_width, TEXT_HEIGHT * rel_height)
    fig.set_dpi(300)

    l_strategies = df_metric.index.to_list()
    l_cases = df_metric.columns.to_list()
    ar_idx_cases = np.arange(len(l_cases))

    nbars = 0
    for i in l_strategies:
        axs.bar(ar_idx_cases + nbars * bar_width, df_metric.loc[i, :].to_list(), bar_width,  label=i)
        nbars += 1

    axs.xaxis.set_ticks(ar_idx_cases + 1.5*bar_width)
    axs.xaxis.set_ticklabels([i.replace('_', '\_') for i in l_cases])
    axs.legend()
    axs.set_title(title)
    axs.grid(color='gray', linestyle='dashed', lw=0.3)
    axs.set_axisbelow(True)
    fig.tight_layout()
    return fig, axs


def plot_bars(l_strategies, l_metric_vals, name_metric):
    title = name_metric
    # Plot init and config
    bar_width = 0.1
    rel_width = 1.0
    rel_height = 0.4
    fig, axs = plt.subplots()
    fig: plt.Figure
    axs: plt.Axes
    fig.set_size_inches(TEXT_WIDTH_MONO * rel_width, TEXT_HEIGHT * rel_height)
    fig.set_dpi(300)

    ar_idx_strategies = np.arange(len(l_strategies))

    axs.bar(ar_idx_strategies, l_metric_vals, label=name_metric)

    axs.xaxis.set_ticks(ar_idx_strategies + 1.5 * bar_width)
    axs.xaxis.set_ticklabels([i.replace('_', '\_') for i in l_strategies])
    axs.legend()
    axs.set_title(title)
    axs.grid(color='gray', linestyle='dashed', lw=0.3)
    axs.set_axisbelow(True)
    fig.tight_layout()
    return fig, axs


def plot_compare_dv(df_dv:pd.DataFrame):
    fig, axs = plt.subplots(4, 2)
    rel_width = 1.1
    rel_height = 0.45
    fig.set_size_inches(TEXT_WIDTH_MONO * rel_width, TEXT_HEIGHT * rel_height)
    fig.set_dpi(200)

    fig, ax = plt.subplots()

    max_abs_dv = df_dv.abs().max().max()
    sns.kdeplot(data=df_dv)

    ax.set_xlim([-max_abs_dv, max_abs_dv])

    fig.show()

    return fig, axs


def plot_sim(pdata, df_sim, df_data=None, legend=True):
    df_data = pdata.df_data if df_data is None else df_data
    assert df_sim.shape[0] == df_data.shape[0]
    fig, axs = plt.subplots(5, 2)
    rel_width = 2.2
    rel_height = 1.1
    fig.set_size_inches(TEXT_WIDTH_MONO * rel_width, TEXT_HEIGHT * rel_height)

    l_cols_v = [i for i in df_sim.columns if i.startswith(SNAM_V)]
    l_cols_dgp = [i for i in df_sim.columns if i.startswith(SNAM_DGP)]
    l_cols_dgq = [i for i in df_sim.columns if i.startswith(SNAM_DGQ)]
    l_cols_loadp = [i for i in pdata.df_data if i.startswith(SNAM_LOADP)]
    l_cols_dgpmax = [i for i in pdata.df_data if i.startswith(SNAM_DGPMAX)]

    for i in l_cols_v:
        axs[0, 0].plot(df_sim.index, df_sim.loc[:, i], label=i)

    axs[0, 0].set_title('Bus Voltages p.u.')

    for i in l_cols_dgp:
        axs[1, 0].plot(df_sim.index, df_sim.loc[:, i], label=i)

    axs[1, 0].set_title('DG Power Injection p.u.')
    axs[1, 0].set_ylim([0., .6])

    for i in l_cols_dgq:
        axs[2, 0].plot(df_sim.index, df_sim.loc[:, i], label=i)

    axs[2, 0].set_title('Reactive Power Injection p.u.')

    for i in l_cols_dgpmax:
        axs[1, 1].plot(df_data.index, df_data.loc[:, i])

    axs[1, 1].set_title('DG Available Power p.u.')
    axs[1, 1].set_ylim([0., .6])

    losses = losses_pu(pdata, df_sim)
    p_inj = p_injection_pu(df_sim)
    loadp = df_data.loc[df_sim.index, l_cols_loadp].sum(axis=1).values

    axs[3, 0].plot(df_sim.index, losses)
    axs[3, 0].set_title('Losses p.u.')
    axs[3, 0].set_ylim([0., .4])

    axs[4, 0].plot(df_sim.index, loadp + losses - p_inj)
    axs[4, 0].set_title('Utility Power Injection p.u.')

    Vr = 0.05
    Vr_margin = 0.03
    axs[0, 0].set_ylim([1 - (Vr + Vr_margin), 1 + (Vr + Vr_margin)])
    axs[0, 0].axhline(1. + Vr, color='r')
    axs[0, 0].axhline(1. - Vr, color='r')
    for i in axs:
        i[0].xaxis.set_major_formatter(mdates.DateFormatter('%d-%H'))

    if legend:
        for i in axs:
            i[0].legend(loc='upper right')

    plt.tight_layout()
    return fig, axs


def plot_putility_boxplot(ar_utility_p_injection, ar_hours):
    rel_width = 1.0
    rel_height = 0.4
    fig, ax = plt.subplots()
    fig: plt.Figure
    ax: plt.Axes
    fig.set_size_inches(TEXT_WIDTH_MONO * rel_width, TEXT_HEIGHT * rel_height)
    fig.set_dpi(300)

    df_data = pd.DataFrame({'hour': ar_hours, 'p_utility': ar_utility_p_injection})
    l_data = []
    for i in range(24):
        l_data.append(df_data['p_utility'][df_data['hour'] == i].values)

    flierprops = dict(marker='o', markersize=5, markeredgewidth=0.3) # markeredgecolor='red'

    ax.boxplot(l_data, labels=range(24), flierprops=flierprops)

    #ax.boxplot(pd.pivot_table(df_data, index=df_data.index, columns=['hour']), labels=list(range(24)))

    ax.set_ylabel('$P_{\\text{utility}}$ p.u.')
    ax.set_xlabel('Hour')

    """
    sns.set_style("ticks")
    sns.set(rc={'text.usetex': True, "font.family": "serif"})
    #PROPS = {
    #    'boxprops': {'facecolor': 'none', 'edgecolor': 'k'},
    #    'medianprops': {'color': 'k'},
    #    'whiskerprops': {'color': 'k'},
    #    'capprops': {'color': 'k'}
    #}
    PROPS = {
        'boxprops': {'facecolor': 'none'},
        'flierprops': {'color': 'k', 'linewidth': 0.3},
        'fliersize': 0.3,
        'fliercolor': 'black'
    }

    sns_plot = sns.boxplot(x=ar_hours, y=ar_utility_p_injection, ax=ax, **PROPS)
    """

    plt.tight_layout()
    return fig, ax


def plot_df_data_day(trange: pd.DatetimeIndex, ar_loadp_sample, ar_dgpmax_sample, label_load, label_dgpmax,
                     xformatter='%I:%M %p', l_units=None):
    if l_units is None:
        l_units = ['p.u.', 'p.u.']
    fig, axs = plt.subplots(2, 1)
    fig: plt.Figure
    fig.set_size_inches(3.5*1.3, 2.9*1.3)
    fig.set_dpi(300)

    for ax in axs:
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter(xformatter))

    axs[0].plot(trange, ar_loadp_sample, lw=0.7)
    axs[0].set_title(label_load)
    axs[0].set_ylabel(l_units[0])
    axs[0].set_ylim([-1.05, 1.05])
    axs[1].plot(trange, ar_dgpmax_sample, lw=0.7)
    axs[1].set_ylabel(l_units[1])
    axs[1].set_title(label_dgpmax)
    axs[1].set_ylim([-0.05, 1.55])

    dt = trange.freq
    trange_ticks = pd.date_range(start=trange[0], end=trange[-1] + dt, periods=5)
    for ax in axs:
        ax.set_xticks(trange_ticks)

    plt.tight_layout()

    return fig, axs