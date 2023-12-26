import os

from ML.Core.trainer import Trainer
import torch
from ML import ml_utils

if __name__ == '__main__':

    # mostly unchanged parameters:
    exp_time = '22_11_2023'
    train_percent = 0.85
    val_percent = 0.1
    intersect = 1
    batch_size = 512
    n_epochs = 30
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

    seq2seq_name = 'Seq2seq'
    mlp_name = 'Mlp'
    rnn_name = 'Rnn'
    ltsf_linear_name = os.path.join('LTSF', 'Linear')
    ltsf_informer_name = os.path.join('LTSF', 'Informer')
    ltsf_transformer_name = os.path.join('LTSF', 'Transformer')

    parent_dirname = r"E:\\Hadar\\experiments" if use_hard_drive else '../Results'
    forces_path = os.path.join(parent_dirname, exp_time, 'f19+f23_list_clean.pt')
    kinematics_path = os.path.join(parent_dirname, exp_time, 'k19+k23_list_clean.pt')
    forces, kinematics = torch.load(forces_path), torch.load(kinematics_path)
    if isinstance(forces, list) and isinstance(kinematics, list):
        input_size, output_size = forces[0].shape[-1], kinematics[0].shape[-1]
    else:
        input_size, output_size = forces.shape[-1], kinematics.shape[-1]

    # mlp_args = dict(input_size=input_size, history_size=feature_win, output_size=output_size, hidden_dims_list=[1])
    # rnn_args = dict(type='gru', input_size=input_size, output_size=output_size, hidden_dim=hid_size, num_layers=nlayers,
    #                 dropout=0.05, bidirectional=False)
    # ltsf_linear_args = dict(seq_len=feature_win, pred_len=target_win, channels=input_size, individual=False,
    #                         output_size=output_size)
    # ltsf_informer_args = dict(pred_len=target_win, label_len=0, output_attention=False, enc_in=input_size,
    #                           d_model=hid_size, dropout=0.05, dec_in=output_size, embed_type=3, factor=1, d_ff=hid_size,
    #                           e_layers=nlayers, activation='gelu', n_heads=2, d_layers=1, c_out=output_size,
    #                           distil=True)
    # ltsf_transformer_args = {key: value for key, value in ltsf_informer_args.items() if key != 'distil'}

    seq2seq_params = ml_utils.generate_hyperparam_combinations(
        global_args=dict(
            feature_lag=[128, 256, 512],
            batch_size=[512]
        ),
        model_args=dict(
            target_lag=[16, 32, 64, 128],
            enc_embedding_size=[2, 4, 8, 16],
            enc_hidden_size=[8, 16, 32, 64],
            enc_num_layers=[1],
            enc_bidirectional=[True],
            output_size=[output_size]
        ),
        model_shared_pairs={
            'dec_hidden_size': 'enc_hidden_size',
            'dec_embedding_size': 'enc_embedding_size'
        }
    )
    for hyperparams in seq2seq_params:
        input_dim = (hyperparams['batch_size'], hyperparams['feature_lag'], input_size)

        hyperparams.update({"input_dim": input_dim})

        trainer = Trainer(
            features_path=forces_path,
            targets_path=kinematics_path,
            train_percent=train_percent,
            val_percent=val_percent,
            feature_win=input_dim[1],
            target_win=hyperparams['model_args']['target_lag'],
            intersect=intersect,
            batch_size=hyperparams['batch_size'],

            model_class_name=seq2seq_name,
            model_args=hyperparams['model_args'],

            exp_name='_'.join(f'{k}{v}_' if not isinstance(v, dict) else ''.join(f'{sk}{sv}' for sk, sv in v.items()) for k, v in hyperparams.items()),
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
#
# valpreds = trainer.predict('val', True)
# testpreds = trainer.predict('test', True)
#
# with open('valpreds_prssm.pkl', 'wb') as handle:
# 	pickle.dump(valpreds, handle)
# with open('tstpreds_prssm.pkl', 'wb') as handle:
# 	pickle.dump(testpreds, handle)
