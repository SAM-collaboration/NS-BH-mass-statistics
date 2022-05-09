import numpy as np
import pandas as pd
from KDEpy import FFTKDE
from scipy.interpolate import interp1d
from scipy.integrate import quad, trapz


def S(m, m_min, dm):
    out = (m - m_min) / dm
    out[m < m_min] = 0
    out[m > m_min + dm] = 1
    return out

def power_law(m, mmax, alpha):
    return m**(-alpha)

def gaussian(m, mu, sigma):
    return np.exp(-(m-mu)**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))

def pdf_blackhole(m, l=0.3, alpha=7.1, mmax=75, mu=29.8, sigma=6.4, mmin=6.8, dm=3):
    out = np.zeros_like(m)
    
    inv_A = quad(lambda m: power_law(m, mmax, alpha), mmin, mmax)[0]
    out += (1-l)*(1./inv_A)*power_law(m, mmax, alpha)
    out[m > mmax] = 0
    out[m <= 10e-2] = 0
    
    out += l * gaussian(m, mu, sigma) * 0
  
    out *= S(m, mmin, dm)

    norm_factor = trapz(out, x=m)
    out /= norm_factor

    return out


def pdf_neutron_double(grid: np.ndarray) -> np.ndarray:
    """Compute the PDF of double neutron stars and interpolate.

    Args:
        @data: input dataframe of remnant mass.
        @grid: grid of values to fit on.

    Returns:
        Interpolated values over the grid.
    """
    p_y = (1 / (np.sqrt(2 * np.pi * pow(0.09, 2))) *
           np.exp((-pow(grid - 1.33, 2)) / (2 * pow(0.09, 2))))
    return p_y


def pdf_neutron_isolated(grid: np.ndarray) -> np.ndarray:
    """Compute the PDF of isolated neutron stars and interpolate.

    Args:
        @data: input dataframe of remnant mass.
        @grid: grid of values to fit on.

    Returns:
        Interpolated values over the grid.
    """
    p_y = (1 / (np.sqrt(2 * np.pi * pow(0.219689, 2))) *
           np.exp((-pow(grid - 1.54, 2)) / (2 * pow(0.219689, 2))))
    return p_y


def pdf_kde(data: pd.DataFrame, grid: np.ndarray, bandwidth: float) -> np.array:
    """Perform the weighted kernel desnity estimation.

    Args:
        @data: input dataframe of remnant mass.
        @grid: grid of values to fit on.

    Returns:
        Estimated kernel density.
    """
    kde = FFTKDE(bw=bandwidth).fit(data['remnant_mass_noisy'].to_numpy(),
                                   weights=data['weights'].to_numpy())
    y = kde.evaluate(grid)
    return y
