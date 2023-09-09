import os.path
import cv2
import numpy as np
import pandas as pd

from Camera import to_3d, camera_utils, tracker


def get_angles(date: str, use_hard_drive=True, show_tracker=False, show_results=False):
    optic_flow = tracker.OpticalFlow('blobs', show_tracker)

    params = {'NumBlobs': 3, 'minArea': 100, 'winSize': (15, 15), 'maxLevel': 2,
              'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)}

    photos_sub_dirname = 'photos'
    crop_params_filename = 'crop_params.pkl'
    trajectory_path = f"Camera\\experiments\\{date}\\trajectory.npy"

    trajectories = {'cam2': None, 'cam3': None}
    if not os.path.exists(trajectory_path):
        for cam in ['cam2', 'cam3']:
            camera_dirname = f"D:\\Hadar\\experiments\\{date}\\{cam}" if use_hard_drive else f'experiments\\{date}\\{cam}'
            cam_crop_params = f"{camera_dirname}\\{crop_params_filename}"
            trajectory = optic_flow.run(camera_dirname, photos_sub_dirname=photos_sub_dirname, **params)
            trajectory = camera_utils.shift_image_based_on_crop(trajectory, cam_crop_params)
            trajectories[cam] = trajectory

        proj1_path = f"Camera\\calibrations\\{date}\\cameraMatrix1.mat"
        proj2_path = f"Camear\\calibrations\\{date}\\cameraMatrix2.mat"

        trajectory_3d = to_3d.triangulate(proj1_path, proj2_path, trajectories['cam2'], trajectories['cam3'])
        np.save(trajectory_path, trajectory_3d)
    else:
        trajectory_3d = np.load(trajectory_path)
    angles = to_3d.xyz2euler(trajectory_3d)
    if show_results:
        camera_utils.plot_trajectories(trajectory_3d, wing_plane_jmp=100)
        camera_utils.plot_angles(angles, camera_freq=10_000)
    return pd.DataFrame(angles.T, columns=['theta', 'phi', 'psi'])
