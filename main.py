import os
import torch
import pandas as pd
import cv2
from tqdm import tqdm

import Camera.main
import Forces.main
import DataHandler.trigger
import DataHandler.preprocess
from DataHandler import encoders, preprocess

if __name__ == '__main__':

	exp_date = '19_10_2023'
	assert (
			os.path.exists(rf"Camera\calibrations\{exp_date}\cameraMatrix1.mat") and
			os.path.exists(rf"Camera\calibrations\{exp_date}\cameraMatrix2.mat")
	), 'run calib.m first!'
	assert (
			os.path.exists(fr"E:\Hadar\experiments\{exp_date}\preset_20302.adj") and
			os.path.exists(fr"E:\Hadar\experiments\{exp_date}\preset_20302.adj")
	), 'preset camera configuration files missing'

	use_hard_drive = True
	human_verification_runtime = True
	human_verification_tensors = False
	add_manual_crop = False
	show_wing_tracker = False
	show_angle_results = False
	show_force_results = False
	show_start_heuristics = True

	camera_n_samples = 10_000
	force_n_samples = 5_000

	angles_lst: list[torch.Tensor] = []
	forces_lst: list[torch.Tensor] = []
	tare = []  # we can use tare to specify force column names we want to shift to zero ( df[tare] - df[tare].iloc[0] )

	crop_params_filename = 'crop_params.pkl'
	parent_dirname = "E:\\Hadar" if use_hard_drive else ''
	first_image_name = '000000.jpg'

	force_data_n_header_rows = 21

	crop_this_many_samples_from_prefix = 2000
	camera_start_threshold: float = 0.15
	forces_start_threshold: float = 0.001
	smooth_method_for_trigger = 'median'
	smooth_kernel_size_for_trigger = 10

	bad_dirs = []

	tracking_params = {
		'NumBlobs': 3, 'minArea': 100, 'winSize': (15, 15), 'maxLevel': 2,
		'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
	}
	preprocessor = preprocess.Preprocess(['interpolate', 'resample'])
	# encoder = encoders.Encoder(['torque'])

	# the subdirectories of experiment names have the same names for cam2, cam3 we use cam2 w.l.o.g
	photos_subdirs_name = sorted(os.listdir(f'{parent_dirname}\\experiments\\{exp_date}\\cam2'))
	for curr_subdir_name in tqdm(photos_subdirs_name):
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
			exp_date=exp_date,
			parent_dirname=parent_dirname,
			photos_sub_dirname=curr_subdir_name,
			show_force_results=show_force_results,
			header_row_count=force_data_n_header_rows,
			tare=tare
		)

		trajectory_3d = trajectory_3d[:, crop_this_many_samples_from_prefix:, :]
		angles_df = angles_df[crop_this_many_samples_from_prefix:].reset_index(drop=True)
		forces_df = forces_df[crop_this_many_samples_from_prefix:].reset_index(drop=True)

		df = DataHandler.trigger.merge_data(
			trajectory_3d=trajectory_3d,
			angles_df=angles_df,
			forces_df=forces_df,
			camera_freq=f"{1 / camera_n_samples}S",
			force_freq=f"{1 / force_n_samples}S",
			camera_threshold=camera_start_threshold,
			force_threshold=forces_start_threshold,
			smooth_method=smooth_method_for_trigger,
			smooth_kernel_size=smooth_kernel_size_for_trigger,
			show_start_indicators=show_start_heuristics
		)
		if human_verification_runtime:
			if input('Is this one bad? [y/Any]') == 'y':
				continue

		df = preprocessor.run(df)
		# df = encoder.run(df)

		angles_lst.append(torch.tensor(df[['theta', 'phi', 'psi']].values))
		forces_lst.append(torch.tensor(df[['F1', 'F2', 'F3', 'F4']].values))

	trim_len = sorted([len(f) for f in angles_lst])[2]
	trimmed_angles = []
	trimmed_forces = []
	for angles, forces in zip(angles_lst, forces_lst):
		angles = angles[:trim_len]
		forces = forces[:trim_len]
		if len(angles) == trim_len and len(forces) == trim_len:
			trimmed_angles.append(angles)
			trimmed_forces.append(forces)

	kinematics = torch.stack(trimmed_angles)
	forces = torch.stack(trimmed_forces)
	print('k shape:', kinematics.shape)
	print('f shape: ', forces.shape)

	torch.save(forces, f"{parent_dirname}\\experiments\\{exp_date}\\forces.pt")
	torch.save(kinematics, f"{parent_dirname}\\experiments\\{exp_date}\\kinematics.pt")
