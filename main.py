import torch

import Camera.main
import Forces.main
import Preprocess.trigger
import Preprocess.preprocess
import ML.main

if __name__ == '__main__':
    # (!) make sure to first run calib.m for the calibration matrices
    exp_time = '22_08_2023'
    camera_freq = 10_000
    force_freq = 5_000
    angles_df = Camera.main.get_angles(exp_time, use_hard_drive=False, show_tracker=False, show_results=False)
    forces_df = Forces.main.get_forces(exp_time, show_results=False)
    df = Preprocess.trigger.merge_data(exp_time,
                                       angles_df, forces_df,
                                       f"{1 / camera_freq}S", f"{1 / force_freq}S",
                                       0.01, 0.1)
    df = Preprocess.preprocess.interpolate(df)
    df = Preprocess.preprocess.resample(df)

    angles_tensor = torch.tensor(df[['theta', 'phi', 'psi']].values).repeat(100, 1, 1)
    forces_tensor = torch.tensor(df[['F1', 'F2', 'F3', 'F4']].values).repeat(100, 1, 1)
    trainer = ML.main.init_trainer(angles_tensor, forces_tensor)
    trainer.fit()
    trainer.predict()
