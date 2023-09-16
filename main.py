import os.path
import torch

import Camera.main
import Forces.main
import Preprocess.trigger
import Preprocess.preprocess
import ML.main

if __name__ == '__main__':

    # (!) make sure to first run calib.m for the calibration matrices
    camera_freq: str = f"{1 / 10_000}S"
    force_freq: str = f"{1 / 5_000}S"
    camera_start_threshold: float = 0.01
    forces_start_threshold: float = 0.1
    angles: list[torch.Tensor] = []
    forces: list[torch.Tensor] = []

    for exp_time in ['22_08_2023']:
        assert all(os.path.exists(f"Camera\\calibrations\\{exp_time}\\cameraMatrix{i}.mat") for i in
                   [1, 2]), 'run calib.m first!'

        angles_df = Camera.main.get_angles(exp_time, use_hard_drive=False, show_tracker=False, show_results=False)
        forces_df = Forces.main.get_forces(exp_time, show_results=False)
        df = Preprocess.trigger.merge_data(exp_time, angles_df, forces_df, camera_freq, force_freq,
                                           camera_start_threshold, forces_start_threshold)
        df = Preprocess.preprocess.interpolate(df)
        df = Preprocess.preprocess.resample(df)

        angles.append(torch.tensor(df[['theta', 'phi', 'psi']].values))
        forces.append(torch.tensor(df[['F1', 'F2', 'F3', 'F4']].values))

    trainer = ML.main.init_trainer(forces=torch.stack(forces), kinematics=torch.stack(angles))
    trainer.fit()
    trainer.predict()
