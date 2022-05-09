import numpy as np
import pandas as pd
import densities
from scipy.integrate import quad, trapz


def kl_div(data: pd.DataFrame, grid: np.ndarray, sn_flag: str):
    """Compute Kullback-Leibler divergence on densities P and Q.

    Args:
        @data: input dataframe of remnant mass.
        @grid: grid of values to fit on.
        @bh_flag: True if working on black holes.
        @ns_double: True if working on double neutrons.

    Returns:
        Kullback-Leibler divergence value.
    """
    if sn_flag == "black_holes":
        bandwidth = 1.0
        p_y = densities.pdf_blackhole(grid)
    elif sn_flag == "ns_isolated":
        bandwidth = 0.1
        p_y = densities.pdf_neutron_isolated(grid)
    elif sn_flag == "ns_double":
        bandwidth = 0.1
        p_y = densities.pdf_neutron_double(grid)

    q_y = densities.pdf_kde(data, grid, bandwidth)
    norm_p = trapz(p_y, x=grid)
    p_y /= norm_p
    flr = max([10 ** -13, min(p_y), min(q_y)])
    p_y[p_y < flr] = flr
    q_y[q_y < flr] = flr
    return p_y * np.log2(p_y / q_y)


def tv_dist(data: pd.DataFrame, grid: np.ndarray, sn_flag: str):
    """Compute Total Variation distance on densities P and Q.

    Args:
        @data: input dataframe of remnant mass.
        @grid: grid of values to fit on.
        @bh_flag: True if working on black holes.
        @ns_double: True if working on double neutrons.

    Returns:
        Absolute Total Variation distance between P and Q.
    """
    if sn_flag == "black_holes":
        bandwidth = 1.0
        p_y = densities.pdf_blackhole(grid)
    elif sn_flag == "ns_isolated":
        bandwidth = 0.1
        p_y = densities.pdf_neutron_isolated(grid)
    elif sn_flag == "ns_double":
        bandwidth = 0.1
        p_y = densities.pdf_neutron_double(grid)

    q_y = densities.pdf_kde(data, grid, bandwidth)
    norm_p = trapz(p_y, x=grid)
    p_y /= norm_p
    return abs(p_y - q_y)
