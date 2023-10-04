from ML.Zoo import seq2seq
from ML.Core.trainer import Trainer
import torch
import torch.nn as nn

if __name__ == '__main__':
	exp_time = '22_09_2023'
	train_percent = 0.85
	val_percent = 0.1
	feature_win = 120
	target_win = 1
	intersect = 0
	batch_size = 256
	n_epochs = 20
	seed = 3407
	criterion = 'L1Loss'
	optimizer = 'Adam'
	patience = 10
	patience_tolerance = 0.005
	features_norm = 'zscore'
	features_global_norm = True
	targets_norm = 'identity'
	targets_global_norm = True
	flip_history = True

	forces_path = '/home/hadar/Downloads/forces_small.pt'
	kinematics_path = '/home/hadar/Downloads/kinematics_small.pt'

	forces, kinematics = torch.load(forces_path), torch.load(kinematics_path)
	input_size, output_size = forces.shape[-1], kinematics.shape[-1]
	input_dim = (batch_size, feature_win, input_size)
	del forces
	del kinematics

	model_name = 'Seq2Seq'

	for emb_size in [8, 16, 32, 64, 128, 256]:
		for hid_size in [8, 16, 32, 64, 128, 256]:
			for fwin in [64, 128, 256, 512]:
				for nlayers in [1, 2, 3]:
					exp_name = f""
					model_args = dict(
						input_dim=input_dim,
						target_lag=target_win,
						enc_embedding_size=emb_size,
						enc_hidden_size=hid_size,
						enc_num_layers=nlayers,
						enc_bidirectional=True,
						dec_embedding_size=emb_size,
						dec_hidden_size=hid_size,
						dec_output_size=output_size
					)

					trainer = Trainer(
						features_path=forces_path,
						targets_path=kinematics_path,
						train_percent=train_percent,
						val_percent=val_percent,
						feature_win=feature_win,
						target_win=target_win,
						intersect=intersect,
						batch_size=batch_size,
						model_name=model_name,
						model_args=model_args,
						exp_name=exp_name,
						optimizer=optimizer,
						criterion=criterion,
						patience=patience,
						patience_tolerance=patience_tolerance,
						n_epochs=n_epochs,
						seed=seed,
						features_norm_method=features_norm,
						features_global_normalizer=features_global_norm,
						targets_norm_method=targets_norm,
						targets_global_normalizer=targets_global_norm,
						flip_history=flip_history
					)
					trainer.fit()

	trainer.predict()
