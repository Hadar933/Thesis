from Forces import parse_forces
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
import pandas as pd


def find_force_start(threshold, force_df, verbose=False):
    surpass_threshold_array = []
    for f in force_df.values.T:
        first_surpass_index = np.argmax(f > threshold)  # if f<=threshold, argmax = 0
        if first_surpass_index > 0:
            surpass_threshold_array.append(first_surpass_index)
    start = np.min(surpass_threshold_array) if len(surpass_threshold_array) > 0 else 0
    if verbose: plot_start(start, force_df)
    return start


def find_camera_start_based_on_points(threshold, trajectory_3d):
    # test of static start up to N(0,0.01)
    # static_start = np.repeat(trajectory_3d[:, 0, :][:, np.newaxis, :], 10, axis=1) + 0.01 * np.random.randn(3, 10, 3)
    # trajectory_3d = np.concatenate([static_start, trajectory_3d], axis=1)

    for t in range(trajectory_3d.shape[1] - 1):
        p0curr, p1curr, p2curr = trajectory_3d[0, t, :], trajectory_3d[1, t, :], trajectory_3d[2, t, :]
        p0next, p1next, p2next = trajectory_3d[0, t + 1, :], trajectory_3d[1, t + 1, :], trajectory_3d[2, t + 1, :]
        dist1, dist2, dist3 = norm(p0next - p0curr), norm(p1next - p1curr), norm(p2next - p2curr)
        if dist1 > threshold or dist2 > threshold or dist3 > threshold:
            print(f"Start frame: {t}")
            return t
    raise ValueError('Did not find start...')


def plot_start(start, df) -> None:
    df.plot()
    plt.axvline(start, color='black', linestyle='--')
    plt.text(x=0.8 * start, y=0.9 * df.max().max(), s='Start')
    plt.show()


def find_camera_start(threshold, angles_df, verbose=False):
    for t in range(len(angles_df) - 1):
        if any((angles_df.iloc[t + 1] - angles_df.iloc[t]).values > threshold):
            start = t
            break
    if verbose: plot_start(start, angles_df)
    return start


def calc_start_of_experiment(force_start, camera_start, force_sample_freq, camera_sample_freq, force_df, angles_df):
    angles_df = angles_df[camera_start:]
    angles_df.index = pd.timedelta_range(start='0', periods=len(angles_df), freq=camera_sample_freq)
    force_df = force_df[force_start:]
    force_df.index = pd.timedelta_range(start='0', periods=len(force_df), freq=force_sample_freq)
    return angles_df, force_df


def merge_data(
        date,
        angles_df: pd.DataFrame, forces_df: pd.DataFrame,
        camera_freq, force_freq,
        camera_threshold, force_threshold
):
    trajectory_path = f"Camera\\experiments\\{date}\\trajectory.npy"
    trajectory_3d = np.load(trajectory_path)

    camera_start = find_camera_start_based_on_points(camera_threshold, trajectory_3d)
    forces_start = find_force_start(force_threshold, forces_df)
    angles_df, forces_df = calc_start_of_experiment(forces_start, camera_start, force_freq, camera_freq, forces_df,
                                                    angles_df)
    data = angles_df.join(forces_df)
    return data
