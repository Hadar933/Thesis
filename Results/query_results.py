import os
import re
import time

import matplotlib.pyplot as plt
import numpy as np

from Camera import camera_utils


def filter_results(exp_date: str, f=None, a=None, k=None):
    parent_dir = rf'E:\Hadar\experiments\{exp_date}\results'
    # Create the search patterns based on provided values
    f_pattern = f"F={f}" if f is not None else ""
    a_pattern = f"A=M_PIdiv{a}" if a is not None else ""
    k_pattern = f"K={k}" if k is not None else ""
    relevant_result_dirs = []
    for subdir in os.listdir(parent_dir):
        subdir = os.path.join(parent_dir, subdir)
        if f_pattern in subdir and a_pattern in subdir and k_pattern in subdir:
            relevant_result_dirs.append(subdir)
    return relevant_result_dirs


def plot_filtered_results(paths, what_to_plot):
    for item in paths:
        if 'angles' in what_to_plot:
            data = np.load(os.path.join(item, f"angles.npy"))
            title_addition = re.search(r'\[(.*)\]', item).group(1)
            camera_utils.plot_angles(data, n_samples=10_000, add_to_title=title_addition)
        if 'trajectories' in what_to_plot:
            data = np.load(os.path.join(item, f"trajectories.npy"))
            title_addition = re.search(r'\[(.*)\]', item).group(1)
            camera_utils.plot_trajectories(data, wing_plane_jmp=200, add_to_title=title_addition)
        # time.sleep(1)


if __name__ == '__main__':
    plot_filtered_results(
        paths=filter_results('22_09_2023', a=3),
        what_to_plot='trajectories angles'
    )
