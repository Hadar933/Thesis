import os

from ML.Core.trainer import Trainer
import torch

from Utilities import utils

if __name__ == '__main__':

    exp_time = '22_11_2023'
    train_percent = 0.85
    val_percent = 0.1
    feature_win = 256
    target_win = 1
    intersect = 0
    batch_size = 512
    n_epochs = 2
    seed = 3407
    criterion = 'L1Loss'
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
    forces_path = os.path.join(parent_dirname, exp_time, 'f19+f23_list_clean.pt')
    kinematics_path = os.path.join(parent_dirname, exp_time, 'k19+k23_list_clean.pt')

    forces, kinematics = torch.load(forces_path), torch.load(kinematics_path)
    if isinstance(forces, list) and isinstance(kinematics, list):
        input_size, output_size = forces[0].shape[-1], kinematics[0].shape[-1]
    else:
        input_size, output_size = forces.shape[-1], kinematics.shape[-1]
    input_dim = (batch_size, feature_win, input_size)

    seq2seq_name = 'Seq2Seq'
    mlp_name = 'MLP'
    rnn_name = 'RNN'
    ltsf_linear_name = 'LTSF\\Linear'

    for emb_size in [5]:
        for hid_size in [30]:
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
                mlp_args = dict(
                    input_size=input_size,
                    history_size=feature_win,
                    output_size=output_size,
                    hidden_dims_list=[1]
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
                ltsf_linear_args = dict(
                    seq_len=feature_win,
                    pred_len=target_win,
                    enc_in=input_size,
                    individual=False
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
                    flip_history=flip_history,
                    regularization_factor=regularization_factor
                )
                trainer.fit()

    valpreds = trainer.predict('val', True)
    testpreds = trainer.predict('test', True)
    utils.plot_df_with_plotly(testpreds)
