import os
import torch
import pandas as pd
import cv2

import Camera.main
import Forces.main
import Preprocess.trigger
import Preprocess.preprocess
import ML.main

if __name__ == '__main__':

    exp_date = '22_09_2023'
    assert all(os.path.exists(f"Camera\\calibrations\\{exp_date}\\cameraMatrix{i}.mat") for i in [1, 2]), \
        'run calib.m first!'

    use_hard_drive = True
    camera_n_samples = 10_000
    force_n_samples = 5_000
    camera_start_threshold: float = 0.01
    forces_start_threshold: float = 0.1
    angles: list[torch.Tensor] = []
    forces: list[torch.Tensor] = []

    tracking_params = {'NumBlobs': 3, 'minArea': 100, 'winSize': (15, 15), 'maxLevel': 2,
                       'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)}

    crop_params_filename = 'crop_params.pkl'
    parent_dirname = "E:\\Hadar" if use_hard_drive else ''
    first_image_name = '000000.jpg'
    add_manual_crop = False
    show_wing_tracker = False
    show_angle_results = True
    show_force_results = False
    num_rows_in_force_data_header = 21

    # the subdirectories of experiment names have the same names for cam2, cam3 we use cam2 w.log
    photos_subdirs_name = sorted(os.listdir(f'{parent_dirname}\\experiments\\{exp_date}\\cam2'))
    for curr_subdir_name in photos_subdirs_name:
        angles, trajectory_3d = Camera.main.get_angles(
            exp_date=exp_date,
            parent_dirname=parent_dirname,
            crop_params_filename=crop_params_filename,
            photos_sub_dirname=curr_subdir_name,
            first_image_name=first_image_name,
            tracking_params=tracking_params,
            add_manual_crop=add_manual_crop,
            show_wing_tracker=show_wing_tracker,
            show_angle_results=show_angle_results
        )
        angles_df = pd.DataFrame(angles.T, columns=['theta', 'phi', 'psi'])
        forces_df = Forces.main.get_forces(
            csv_filename=curr_subdir_name,
            show_force_results=show_force_results,
            header_row_count=num_rows_in_force_data_header
        )
        df = Preprocess.trigger.merge_data(
            parent_dirname=parent_dirname,
            exp_date=exp_date,
            angles_df=angles_df,
            forces_df=forces_df,
            camera_freq=f"{1 / camera_n_samples}S",
            force_freq=f"{1 / force_n_samples}S",
            camera_threshold=camera_start_threshold,
            force_threshold=forces_start_threshold
        )
        df = Preprocess.preprocess.interpolate(df)
        df = Preprocess.preprocess.resample(df)

        angles.append(torch.tensor(df[['theta', 'phi', 'psi']].values))
        forces.append(torch.tensor(df[['F1', 'F2', 'F3', 'F4']].values))

    trainer = ML.main.init_trainer(forces=torch.stack(forces), kinematics=torch.stack(angles))
    trainer.fit()
    trainer.predict()
