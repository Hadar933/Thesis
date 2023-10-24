import os

from ML.Core.trainer import Trainer
import torch

from ML.Utilities import utils

if __name__ == '__main__':
	exp_time = '23_10_2023'
	train_percent = 0.85
	val_percent = 0.1
	feature_win = 256
	target_win = 1
	intersect = 0
	batch_size = 512
	n_epochs = 20
	seed = 3407
	criterion = 'L1Loss + pv_l1'
	regularization_factor = 10
	optimizer = 'Adam'
	patience = 10
	patience_tolerance = 0.005
	features_norm = 'zscore'
	features_global_norm = True
	targets_norm = 'identity'
	targets_global_norm = True
	flip_history = True

	use_hard_drive = False
	parent_dirname = r"E:\\Hadar\\experiments" if use_hard_drive else '../Results'
	forces_path = os.path.join(parent_dirname, exp_time, 'forces_list.pt')
	kinematics_path = os.path.join(parent_dirname, exp_time, 'kinematics_list.pt')

	forces, kinematics = torch.load(forces_path), torch.load(kinematics_path)
	if isinstance(forces, list) and isinstance(kinematics, list):
		input_size, output_size = forces[0].shape[-1], kinematics[0].shape[-1]
	else:
		input_size, output_size = forces.shape[-1], kinematics.shape[-1]
	input_dim = (batch_size, feature_win, input_size)

	seq2seq_name = 'Seq2Seq'
	mlp_name = 'MLP'
	rnn_name = 'RNN'

	for emb_size in [1]:
		for hid_size in [1]:
			for nlayers in [1]:
				exp_name = f"pv_l1_test"
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
				mlp_args = dict(
					input_size=input_size,
					history_size=feature_win,
					output_size=output_size,
					hidden_dims_list=[30, 50, 30]
				)
				rnn_args = dict(
					type='gru',
					input_size=input_size,
					output_size=output_size,
					hidden_dim=hid_size,
					num_layers=nlayers,
					dropout=0.05,
					bidirectional=False
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
					model_class_name=rnn_name,
					model_args=rnn_args,
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
					flip_history=flip_history,
					regularization_factor=regularization_factor
				)
				trainer.fit()

	preds = trainer.predict('val')
	utils.plot_df_with_plotly(preds)
