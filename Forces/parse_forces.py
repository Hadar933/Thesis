import pandas as pd
from typing import Tuple, Dict, List
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def read_forces_csv(
        csv_filename: str,
        header_row_count: int = 21
) -> Tuple[pd.DataFrame, Dict[str, List]]:
    """
    reads the csv_filename and extracts the header information alongside a forces dataframe
    :param csv_filename: forces csv file to read
    :param header_row_count: the size of the header
    :return: a tuple of header dictionary and forces dataframe
    """
    # Extract relevant information from the header:
    with open(csv_filename, 'r') as file:
        header_lines = file.readlines()[:header_row_count]
    header_dict = {}
    for line in header_lines:
        sp = line.split(':,')
        if len(sp) == 2:
            value_split = sp[1].strip('\n').split(',')
            header_dict[sp[0]] = value_split

    # skip the header and parse the forces to a df:
    f1_col_ind = 1
    f4_col_ind = f1_col_ind + 4
    forces_df = pd.read_csv(csv_filename, skiprows=header_row_count).iloc[:-1, f1_col_ind:f4_col_ind]
    forces_df.columns = ['F1', 'F2', 'F3', 'F4']
    forces_df = forces_df.applymap(lambda x: float(x.replace(' N', '')))  # remove ` N` from every cell

    return forces_df, header_dict


def plot_forces(df: pd.DataFrame, header: Dict[str, List], convert_to_time: bool = True):
    sample_rate = int(header['Sampling rate'][0].split('/')[0])  # extracting int value frequency
    start_time = " ".join(header['Start (Date Time)'])
    if convert_to_time:
        df.index = np.arange(0, len(df)) / sample_rate
    mpl.use("TkAgg")
    df.plot(title=f"Forces [{sample_rate} sample rate], [{start_time}]",
            xlabel="Time [sec]" if convert_to_time else f"Sample [#]",
            grid=True)
    plt.show()
