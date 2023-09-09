from typing import List, Optional
import numpy as np
import pandas as pd
from scipy import special


class Encoder:
    def __init__(self, encoding_tasks: Optional[List[str]] = None):
        self.encoding_tasks = encoding_tasks

    @staticmethod
    def encode_derivatives(df: pd.DataFrame, cols: Optional[List[str]] = None) -> pd.DataFrame:
        if not cols: cols = df.columns
        time = df.index.to_series().diff().dt.total_seconds().values[:, None]

        col_dot_names = [f"{col}_dot" for col in cols]
        df[col_dot_names] = df[cols].diff() / time

        col_ddot_names = [f"{col}_ddot" for col in cols]
        df[col_ddot_names] = df[col_dot_names].diff() / time
        return df

    @staticmethod
    def encode_hermite(df: pd.DataFrame, cols: Optional[List[str]] = None, orders: Optional[List[int]] = None):
        if not cols: cols = df.columns
        if not orders: orders = [2, 3, 4]
        for n in orders:
            col_names = [f"{col}_hermite_{n}" for col in cols]
            poly_h = special.hermite(n)
            df[col_names] = poly_h(df)
        return df

    @staticmethod
    def encode_angle(df: pd.DataFrame, cols: Optional[List[str]] = None):
        if not cols: cols = df.columns
        sin_col_names = [f"sin_{col}" for col in cols]
        cos_col_names = [f"cos_{col}" for col in cols]
        arg = 2 * np.pi * df[cols] / df[cols].max()
        df[sin_col_names] = np.sin(arg)
        df[cos_col_names] = np.cos(arg)
        return df

    def encode_torque(self, f1, f2, f3, f4, radius):
        amplitude = radius / np.sqrt(2)
        torque_x = amplitude * ((f1 + f2) - (f3 + f4))
        torque_y = amplitude * ((f2 + f3) - (f1 + f4))
        return torque_x, torque_y


if __name__ == '__main__':
    start_time = '2023-03-19 12:00:00'
    experiment_time = 1  # second
    end_time = pd.to_datetime(start_time) + pd.Timedelta(seconds=experiment_time)
    time_index = pd.date_range(start=start_time, end=end_time, freq='1ms')
    time_values = (time_index - time_index[0]).astype('timedelta64[ms]').astype(float) / 1000
    A = 1.0
    f = 10.0
    phi = 1.0

    # Generate the sine wave
    df1 = pd.DataFrame(data={'sin': A * np.sin(2 * np.pi * f * time_values + phi)},
                       index=time_index)
