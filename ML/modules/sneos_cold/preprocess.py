import numpy as np
import pandas as pd


def process_data(file_path: str) -> pd.DataFrame:
    """Read in data file and preprocess it.

    Args:
        @file_path: relative path to input file.

    Returns:
        Processed and cleaned up dataframe.
    """
    data = pd.read_csv(file_path)
    data.columns = ['EOS', 'ZAMS_mass', 'remnant_mass', 'outcome']
    return data


def split_states(data: pd.DataFrame) -> list:
    """Split the dataframe into individual states.

    Args:
        @data: input dataframe of EOS.

    Return:
        List of individual states.
    """
    states_data = []
    states = sorted(data['EOS'].unique())

    for state in states:
        states_data.append(data[data['EOS'] == state].reset_index())
    return states, states_data
