import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import trapz

import densities
import utils
import preprocess
from metrics import kl_div, tv_dist


def compute_bh(data: pd.DataFrame, state: int, grid: np.ndarray, sn_flag: str, *kwargs):
    """BHwd."""
    print(f"{'-'*31}\nDistance Metrics: [Black Holes]\n{'-'*31}")
    eos_rmass = data[data['outcome'] != 0]

    # Compute Kullback-Leibler Divergence
    kl_values = kl_div(eos_rmass, grid, sn_flag)
    d_kl = trapz(kl_values, x=grid)

    # Compute Total Variation
    tv_values = tv_dist(eos_rmass, grid, sn_flag)
    d_tv = 0.5 * trapz(tv_values, x=grid)

    print(f"- KL Divergence for EOS {state}: {d_kl:.4f}")
    print(f"- Total Variation for EOS {state}: {d_tv:.4f}\n")
    return [d_kl, d_tv]


def compute_nss(data: pd.DataFrame, state: int, grid: np.ndarray, sn_flag: str):
    """BHwd."""
    print(f"{'-'*37}\nDistance Metrics: [Neutrons Single Slow]\n{'-'*37}")
    eos_rmass = data[data['outcome'] == 0]

    # Compute Kullback-Leibler Divergence
    kl_values = kl_div(eos_rmass, grid, sn_flag)
    d_kl = trapz(kl_values, x=grid)

    # Compute Total Variation
    tv_values = tv_dist(eos_rmass, grid, sn_flag)
    d_tv = 0.5 * trapz(tv_values, x=grid)

    print(f"- KL Divergence for EOS {state}: {d_kl:.4f}")
    print(f"- Total Variation for EOS {state}: {d_tv:.4f}\n")
    return [d_kl, d_tv]


def compute_nsd(data: pd.DataFrame, state: int, grid: np.ndarray, sn_flag: str):
    """BHwd."""
    print(f"{'-'*35}\nDistance Metrics: [Neutrons Double]\n{'-'*35}")
    eos_rmass = data[data['outcome'] == 0]

    # Compute Kullback-Leibler Divergence
    kl_values = kl_div(eos_rmass, grid, sn_flag)
    d_kl = trapz(kl_values, x=grid)

    # Compute Total Variation
    tv_values = tv_dist(eos_rmass, grid, sn_flag)
    d_tv = 0.5 * trapz(tv_values, x=grid)

    print(f"- KL Divergence for EOS {state}: {d_kl:.4f}")
    print(f"- Total Variation for EOS {state}: {d_tv:.4f}\n")
    return [d_kl, d_tv]