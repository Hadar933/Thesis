from typing import List, Optional
import numpy as np
import pandas as pd
from scipy import special
from loguru import logger
import matplotlib.pyplot as plt

#
# class Encoder:
#     def __init__(self, encoding_tasks: Optional[List[str]] = None):
#         self.encoding_tasks = encoding_tasks or []
#         self.task_map = {
#             method: getattr(self, method) for method in
#             [
#                 func for func in dir(self) if
#                 callable(getattr(self, func))
#                 and not func.startswith("_")
#                 and func != 'encode'
#             ]
#         }
#
#     def encode(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
#         for task in self.encoding_tasks:
#             func = self.task_map.get(task)
#             if func:
#                 df = func(df, **kwargs)
#             else:
#                 logger.warning(f"Could not find encoder function {func}, moving on.")
#         return df
#
#     def encode_fourier(self,df: pd.DataFrame, **kwargs) -> pd.DataFrame:
#         if cols is None:
#             cols = df.columns
#
#         transformed_data = {}
#         for col in cols:
#             transformed_data[f"{col}_fourier"] = np.fft.fft(df[col])
#
#         return pd.DataFrame(transformed_data)
#     @staticmethod
#     def encode_derivatives(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
#         if not cols: cols = df.columns
#         time = df.index.to_series().diff().dt.total_seconds().values[:, None]
#
#         col_dot_names = [f"{col}_dot" for col in cols]
#         df[col_dot_names] = df[cols].diff() / time
#
#         col_ddot_names = [f"{col}_ddot" for col in cols]
#         df[col_ddot_names] = df[col_dot_names].diff() / time
#         return df
#
#     @staticmethod
#     def encode_hermite(df: pd.DataFrame, **kwargs):
#         if not cols: cols = df.columns
#         if not orders: orders = [2, 3, 4]
#         for n in orders:
#             col_names = [f"{col}_hermite_{n}" for col in cols]
#             poly_h = special.hermite(n)
#             df[col_names] = poly_h(df)
#         return df
#
#     @staticmethod
#     def encode_angle(df: pd.DataFrame, **kwargs):
#         if not cols: cols = df.columns
#         sin_col_names = [f"sin_{col}" for col in cols]
#         cos_col_names = [f"cos_{col}" for col in cols]
#         arg = 2 * np.pi * df[cols] / df[cols].max()
#         df[sin_col_names] = np.sin(arg)
#         df[cos_col_names] = np.cos(arg)
#         return df
#
#     def encode_torque(self, df, **kwargs):
#         amplitude = radius / np.sqrt(2)
#         torque_x = amplitude * ((f1 + f2) - (f3 + f4))
#         torque_y = amplitude * ((f2 + f3) - (f1 + f4))
#         return torque_x, torque_y
#
#
# if __name__ == '__main__':
#     start_time = '2023-03-19 12:00:00'
#     experiment_time = 1  # second
#     end_time = pd.to_datetime(start_time) + pd.Timedelta(seconds=experiment_time)
#     time_index = pd.date_range(start=start_time, end=end_time, freq='1ms')
#     time_values = (time_index - time_index[0]).astype('timedelta64[ms]').astype(float) / 1000
#     A = 1.0
#     f = 10.0
#     phi = 1.0
#
#     # Generate the sine wave
#     df1 = pd.DataFrame(data={'sin': A * np.sin(2 * np.pi * f * time_values + phi)},
#                        index=time_index)


if __name__ == '__main__':
    samplingFrequency = 100
    samplingInterval = 1 / samplingFrequency
    beginTime = 0
    endTime = 10
    signal1Frequency = 41
    signal2Frequency = 17
    time = np.arange(beginTime, endTime, samplingInterval)
    amplitude1 = np.sin(2 * np.pi * signal1Frequency * time)
    amplitude2 = np.sin(2 * np.pi * signal2Frequency * time)

    amplitude = amplitude1 + amplitude2
    fourierTransform = np.fft.fft(amplitude) / len(amplitude)  # Normalize amplitude

    fourierTransform = fourierTransform[range(int(len(amplitude) / 2))]  # Exclude sampling frequency
    fourierTransform = np.abs(fourierTransform)
    tpCount = len(amplitude)
    values = np.arange(int(tpCount / 2))
    timePeriod = tpCount / samplingFrequency
    frequencies = values / timePeriod

    sorted_indices = np.argsort(fourierTransform)
    max_peak1_index = sorted_indices[-1]
    max_peak2_index = sorted_indices[-2]

    plt.plot(frequencies, abs(fourierTransform))
    plt.scatter(frequencies[max_peak1_index], fourierTransform[max_peak1_index], color='red',
                label=f'Peak at {frequencies[max_peak1_index]:.2f} Hz')
    plt.scatter(frequencies[max_peak2_index], fourierTransform[max_peak2_index], color='green',
                label=f'Peak at {frequencies[max_peak2_index]:.2f} Hz')
    plt.legend()
    plt.show()
