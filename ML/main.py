import os

from ML.Core.trainer import Trainer
import torch

if __name__ == '__main__':

	t = Trainer.from_model_dirname(
		r"G:\My Drive\Master\Lab\Thesis\ML\saved_models\Input(256, 120, 3)seq2seq[1,2,2,1,True,2,2,2]__2023-10-10_14-05-42"
	)
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

	use_hard_drive = False
	parent_dirname = r"E:\\Hadar\\experiments" if use_hard_drive else '..\\Results'
	forces_path = os.path.join(parent_dirname, exp_time, 'forces.pt')
	kinematics_path = os.path.join(parent_dirname, exp_time, 'kinematics.pt')

	forces, kinematics = torch.load(forces_path), torch.load(kinematics_path)
	input_size, output_size = forces.shape[-1], kinematics.shape[-1]
	input_dim = (batch_size, feature_win, input_size)
	del forces
	del kinematics

	seq2seq_name = 'Seq2Seq'

	for emb_size in [2]:
		for hid_size in [2]:
			for fwin in [2]:
				for nlayers in [1]:
					exp_name = f""
					seq2seq_args = dict(
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
						model_class_name=seq2seq_name,
						model_args=seq2seq_args,
						exp_name=exp_name,
						optimizer_name=optimizer,
						criterion_name=criterion,
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
