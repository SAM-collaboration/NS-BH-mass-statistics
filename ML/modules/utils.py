import numpy as np
import pandas as pd


def generate_noise(data: pd.DataFrame) -> pd.DataFrame:
    """Compute the noise term for each remnant mass.

    Args:
        @data: input dataframe.

    Returns:
        New dataframe with added noise term.
    """
    data['noise'] = 0
    sigma_bh = 0.120213 * data['remnant_mass'] + 0.355936
    sigma_ns = 0.12

    # Compute neutron star noise
    ns_mask = (data['outcome'] == 0)
    ns = data[ns_mask]
    data.loc[ns_mask, 'noise'] = pd.Series(np.random.normal(0,
                                           sigma_ns, len(ns_mask)))

    # Compute black hole noise
    bh_mask = (data['outcome'] != 0)
    bh = data[bh_mask]
    data.loc[bh_mask, 'noise'] = pd.Series(np.random.normal(0, 
                                           sigma_bh, len(bh_mask)))

    data['remnant_mass_noisy'] = data['remnant_mass'] + data['noise']
    data.loc[data['remnant_mass_noisy'] < 0, 'remnant_mass_noisy'] = 10e-2
    return data


def generate_weights(data: pd.DataFrame) -> np.array:
    """haha."""
    m = data['ZAMS_mass'].to_numpy()
    condlist = [m < 0.5, np.logical_and(0.5 <= m, m < 1.0), m >= 1.0]
    funclist = [lambda m: 0.035 * pow(m, -1.3),
                lambda m: 0.019 * pow(m, -2.2),
                lambda m: 0.019 * pow(m, -2.7)]
    return np.piecewise(m, condlist, funclist)


def generate_grid(data: pd.DataFrame, num_samples: int) -> np.ndarray:
    """Generate a linear space grid.

    Args:
        @data: input dataframe.
        @num_samples: number of data points to generate.

    Returns:
        A linear grid.
    """
    grid = np.linspace(10e-15, 30, num=num_samples)
    return grid

def generate_bootstrap(state_data, num_noise):
    sample = state_data.sample(frac=1.0, replace=True, ignore_index=True)
    bootstrap = pd.concat([sample] * num_noise, ignore_index=True)
    bootstrap = generate_noise(bootstrap)
    bootstrap['weights'] = generate_weights(bootstrap)

    return bootstrap

def generate_sample(state_data, num_noise):
    sample = pd.concat([state_data] * num_noise, ignore_index=True)
    sample = generate_noise(sample)
    sample['weights'] = generate_weights(sample)

    return sample
