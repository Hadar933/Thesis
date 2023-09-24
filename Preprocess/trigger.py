import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
import pandas as pd
import Camera.tracker


def find_force_start(threshold, force_df, verbose=False):
    surpass_threshold_array = []
    for f in force_df.values.T:
        first_surpass_index = np.argmax(f > threshold)  # if f<=threshold, argmax = 0
        if first_surpass_index > 0:
            surpass_threshold_array.append(first_surpass_index)
    start = np.min(surpass_threshold_array) if len(surpass_threshold_array) > 0 else 0
    if verbose: plot_start(start, force_df)
    return start


def find_camera_start_based_on_trajectory(threshold, trajectory_3d):
    for t in range(trajectory_3d.shape[1] - 1):
        p0curr, p1curr, p2curr = trajectory_3d[0, t, :], trajectory_3d[1, t, :], trajectory_3d[2, t, :]
        p0next, p1next, p2next = trajectory_3d[0, t + 1, :], trajectory_3d[1, t + 1, :], trajectory_3d[2, t + 1, :]
        dist1, dist2, dist3 = norm(p0next - p0curr), norm(p1next - p1curr), norm(p2next - p2curr)
        if dist1 > threshold or dist2 > threshold or dist3 > threshold:
            return t
    raise ValueError('Did not find camera start...')


def find_camera_start_based_on_images(images_path, tracking_params, threshold=150.0, verbose=True):
    images = os.listdir(images_path)
    images = sorted([os.path.join(images_path, f) for f in images if f.endswith(".jpg")])

    prev_image = cv2.imread(images[0], cv2.IMREAD_GRAYSCALE)
    blob_detector = Camera.tracker.NaiveBlobFinder(tracking_params['NumBlobs'], **tracking_params)
    blobs = blob_detector.run(prev_image, verbose=verbose)
    p0 = np.array([[*b.pt, b.size / 2] for b in blobs])
    plot_every_n = 500

    def get_region_avg_diff(img1, img2, x, y, radius):
        region1 = img1[int(y - radius):int(y + radius) + 1, int(x - radius):int(x + radius) + 1]
        region2 = img2[int(y - radius):int(y + radius) + 1, int(x - radius):int(x + radius) + 1]
        return np.abs(region1 - region2).mean()

    def plot_points(p0):
        plt.figure(figsize=(12, 12))  # Adjusted to make the plots a bit smaller

        for idx, (x, y, radius) in enumerate(p0, start=1):
            # Extract region for the current point from both images
            region1 = prev_image[int(y - radius):int(y + radius) + 1, int(x - radius):int(x + radius) + 1]
            region2 = curr_image[int(y - radius):int(y + radius) + 1, int(x - radius):int(x + radius) + 1]
            diff = np.abs(region1 - region2)

            # Plot region from the first image
            plt.subplot(3, 3, idx)
            plt.imshow(region1, cmap='gray')
            plt.title(f"Point {idx} - Region1", fontsize=14)  # Increased fontsize

            # Plot region from the second image
            plt.subplot(3, 3, idx + 3)
            plt.imshow(region2, cmap='gray')
            plt.title(f"Point {idx} - Region2", fontsize=14)  # Increased fontsize

            # Plot absolute difference
            plt.subplot(3, 3, idx + 6)
            plt.imshow(diff, cmap='gray')
            plt.title(f"Point {idx} - Abs Diff", fontsize=14)  # Increased fontsize

        plt.tight_layout()
        plt.show()

    for idx, image_path in enumerate(images[1:], start=1):
        curr_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Calculate region average differences for each point
        motion_magnitudes = [get_region_avg_diff(prev_image, curr_image, x, y, r) for x, y, r in p0]
        if idx % plot_every_n == 0:
            plot_points(p0)
        # Check if any region's average difference exceeds the threshold
        if any(magnitude > threshold for magnitude in motion_magnitudes):
            return idx  # This image index is where motion starts
        prev_image = curr_image
    return None  # No motion detected


def plot_start(camera_start, forces_start, camera_df, forces_df) -> None:
    ax = camera_df.plot()
    plt.axvline(camera_start, color='red', linestyle='--')
    plt.text(x=0.8 * camera_start, y=0.9 * camera_df.max().max(), s='Camera\n start')

    forces_df.plot(ax=ax)
    plt.axvline(forces_start, color='blue', linestyle='--')
    plt.text(x=0.8 * forces_start, y=0.9 * forces_df.max().max(), s='Forces\n start')

    plt.show()


def trim_and_join(force_start, camera_start, force_sample_freq, camera_sample_freq, force_df, angles_df):
    angles_df = angles_df[camera_start:]
    angles_df.index = pd.timedelta_range(start='0', periods=len(angles_df), freq=camera_sample_freq)
    force_df = force_df[force_start:]
    force_df.index = pd.timedelta_range(start='0', periods=len(force_df), freq=force_sample_freq)
    return angles_df, force_df


def merge_data(
        trajectory_3d,
        angles_df: pd.DataFrame,
        forces_df: pd.DataFrame,
        camera_freq, force_freq,
        camera_threshold,
        force_threshold,
        show_start_indicators
):
    camera_start = find_camera_start_based_on_trajectory(camera_threshold, trajectory_3d)
    forces_start = find_force_start(force_threshold, forces_df)
    if show_start_indicators:
        plot_start(camera_start, forces_start, angles_df, forces_df)
    angles_df, forces_df = trim_and_join(forces_start, camera_start, force_freq, camera_freq, forces_df,
                                         angles_df)
    data = angles_df.join(forces_df)
    return data


if __name__ == '__main__':
    tracking_params = {'NumBlobs': 3, 'minArea': 100, 'winSize': (15, 15), 'maxLevel': 2,
                       'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)}
    images_path = "E:\\Hadar\\experiments\\22_09_2023\\cam2\\Photos[F=20_A=M_PIdiv3_K=0.9]"
    s = find_camera_start_based_on_images(images_path, tracking_params)
    x = 1
