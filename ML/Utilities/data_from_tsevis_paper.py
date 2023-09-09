from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

wing = {
    "psi": [90, 53, 90, -53],
    "theta": [0.0, 1.0, 0.0, 0.0],
    "phi": [90, 65, -90, -65],
    "delta_psi": -90,
    "delta_theta": 90,
    "C": 2.4,
    "K": 0.7,
    "hingeR": [0.0001, 0.0, 0.0001],
    "hingeL": [0.0001, 0.0, 0.0001],
    "ACloc": [0.00175, 0.0, 0.0],
    "bps": 10,
    "span": 0.0025,
    "chord": 0.0007,
    "speedCalc": [0.0025, 0.0, 0.0]}

wing['omega'] = wing['bps'] * 2 * np.pi
wing['T'] = 1 / wing['bps']


def wing_angles(psi, theta, phi, omega, delta_psi, delta_theta, c, k, t):
    """
    computes the wing angles given a set of variables, described in (see Whithead et al, "Pitch perfect: how fruit flies
     control their body pitch angle." 2015, appendix 1)
    """
    # t = t.timestamp() - REF_TIME
    psi_w = psi[0] + psi[1] * np.tanh(c * np.sin(omega * t + delta_psi)) / np.tanh(c)  # angle of attack
    theta_w = theta[0] + theta[1] * np.cos(2 * omega * t + delta_theta)  # elevation
    phi_w = phi[0] + phi[1] * np.arcsin(k * np.sin(omega * t)) / np.arcsin(k)  # motor rotation angle

    psi_dot = -(c * omega * psi[1] * np.cos(delta_psi + omega * t) * (
            np.tanh(c * (np.sin(delta_psi + omega * t))) ** 2 - 1)) / np.tanh(c)
    theta_dot = -2 * omega * theta[1] * np.sin(delta_theta + 2 * omega * t)
    phi_dot = k * (omega * phi[1] * np.cos(omega * t)) / (
            np.arcsin(k) * (1 - k ** 2 * (np.sin(omega * t)) ** 2) ** (1 / 2))

    return psi_w, theta_w, phi_w, psi_dot, theta_dot, phi_dot


def make_data_realer(df: pd.DataFrame, add_noise: bool = True, add_nans: bool = True,
                     nan_probability: Optional[float] = 0.0):
    if add_nans:
        df = df.mask(np.random.random(df.shape) < nan_probability)
    if add_noise:
        mu = 0
        for col in df.columns:
            sigma = max([1, df[col].median() // 10])  # sigma = 10% of median value
            noise = np.random.normal(mu, sigma, len(df))
            df[col] += noise
    return df


def generate_dummy_data(t_start: str, t_end: str, phi_frequencies: List[float], phi_amplitudes: List[float],
                        time_delta: str, add_NaN: Optional[bool] = True, add_noise: Optional[bool] = True,
                        nan_probability: Optional[float] = 0.02):
    """
    performs a series of runs, each represent a single experiment  with different phi frequency and amplitude
    :param phi_amplitudes: list of amplitudes to consider
    :param phi_frequencies: list of frequencies to consider
    :param t_start: start time of a single run
    :param t_end:  end time of a single run
    :param time_delta: timem delta between two consecutive discrete time values
    :param add_NaN: add randomly located nans with some small probability
    :param add_noise: adds gaussian noise to every column in the data
    :param nan_probability:

    :return: a matrix X with rows for history and columns for features
    """
    df = pd.DataFrame(columns=['psi', 'theta', 'phi'])
    time = pd.date_range(start=t_start, end=t_end, freq=time_delta)
    np_time = np.array([t.timestamp() - REF_TIME for t in time])
    C, K = wing["C"], wing["K"]
    delta_psi, delta_theta = wing["delta_psi"], wing["delta_theta"]
    Bias_psi, A_psi = 90.0, 53.0
    Bias_theta, A_theta = 0.0, 1.0
    Bias_phi = 90.0
    for f_phi in phi_frequencies:
        for A_phi in phi_amplitudes:
            omega = 2 * np.pi * f_phi
            psi, theta, phi, psi_dot, theta_dot, phi_dot = wing_angles([Bias_psi, A_psi], [Bias_theta, A_theta],
                                                                       [Bias_phi, A_phi], omega, delta_psi, delta_theta,
                                                                       C, K, np_time)
            curr_df = pd.DataFrame({'psi': psi, 'theta': theta, 'phi': phi}, index=time)
            curr_df = make_data_realer(curr_df, add_noise, add_NaN, nan_probability)
            df = pd.concat([df, curr_df])
    df.to_pickle('dummy_data')
    return df


def generate_dummy_target(t_start, t_end, sample_freq: str, add_NaN: Optional[bool] = True,
                          add_noise: Optional[bool] = True, nan_probability: Optional[float] = 0.02):
    time = pd.date_range(start=t_start, end=t_end, freq=sample_freq)
    np_time = np.array([t.timestamp() - REF_TIME for t in time])
    df = pd.DataFrame({'lift_force': np.sin(2 * np.pi * np_time)})
    df = make_data_realer(df, add_noise, add_NaN, nan_probability)
    df.index = time
    df.to_pickle('dummy_target')
    return df


if __name__ == '__main__':
    start = '2023-01-01 00:00:00'
    end = '2023-01-01 00:00:05'
    REF_TIME = pd.Timestamp(start).timestamp()

    data_freq = 'ms'
    target_freq = '2ms'
    generate_dummy_data(start, end, [1, 3, 4], [0, np.pi / 4, np.pi / 2], data_freq)
    generate_dummy_target(start, end, target_freq)
