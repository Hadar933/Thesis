import os
import json
from typing import Literal

from ML.Core.trainer import Trainer
import torch
from ML import ml_utils
from Utilities import utils

if __name__ == '__main__':
    data: Literal['ours', 'prssm'] = 'ours'
    # mostly unchanged parameters:
    exp_time = '22_11_2023' if data == 'ours' else '10_10_2023'
    train_percent = 0.85
    val_percent = 0.1
    intersect = 1
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
    ltsf_autoformer_name = os.path.join('LTSF', 'Autoformer')
    ltsf_fedformer_name = os.path.join('LTSF', 'FedFormer')

    parent_dirname = r"E:\\Hadar\\experiments" if use_hard_drive else '../Results'
    force_filename = 'f19+f23_list_clean.pt' if data == 'ours' else 'forces_prssm.pt'
    kinematics_filename = 'k19+k23_list_clean.pt' if data == 'ours' else 'kinematics_prssm.pt'
    forces_path = os.path.join(parent_dirname, exp_time, force_filename)
    kinematics_path = os.path.join(parent_dirname, exp_time, kinematics_filename)
    forces, kinematics = torch.load(forces_path), torch.load(kinematics_path)
    if isinstance(forces, list) and isinstance(kinematics, list):
        input_size, output_size = forces[0].shape[-1], kinematics[0].shape[-1]
    else:
        input_size, output_size = forces.shape[-1], kinematics.shape[-1]

    model_args_key = 'model_args'
    feature_lags = [512]
    batch_sizes = [512]
    target_lags = [1]
    embedding_sizes = [5, 10, 15, 20, 25, 30, 35, 40]
    hidden_sizes = [10 * i for i in range(1, 13)]
    label_lens = [0]
    layers = [1]
    bidirs = [True]
    dropouts = [0.05]
    activations = ['gelu']
    output_attentions = [False]
    embed_types = [3]
    factors = [1]
    n_heads = [2]
    e_layers = [1, 2]
    d_layers = [1, 2]
    distil = [True]
    moving_avg = [25]
    fedformer_version = ['Fourier']
    fedformer_mode_select = ['random']
    fedformer_n_modes = [32]
    individual = [True, False]

    seq2seq_params = ml_utils.generate_hyperparam_combinations(
        global_args=dict(feature_lags=feature_lags, batch_size=batch_sizes),
        model_args=dict(target_lags=target_lags, enc_embedding_size=[30], enc_hidden_size=[35],
                        enc_num_layers=layers, enc_bidirectional=bidirs, dec_output_size=[output_size]),
        model_shared_pairs={'dec_hidden_size': 'enc_hidden_size', 'dec_embedding_size': 'enc_embedding_size'},
        model_args_key=model_args_key
    )
    transformer_params = ml_utils.generate_hyperparam_combinations(
        global_args=dict(feature_lag=feature_lags, batch_size=batch_sizes),
        model_args=dict(pred_len=target_lags, label_len=label_lens, output_attention=output_attentions,
                        enc_in=[input_size], d_model=hidden_sizes, dropout=dropouts, dec_in=[output_size],
                        embed_type=embed_types, factor=factors, e_layers=e_layers, activation=activations,
                        n_heads=n_heads, d_layers=d_layers, c_out=[output_size]),
        model_shared_pairs={'d_ff': 'd_model'},
        model_args_key=model_args_key
    )
    informer_params = ml_utils.generate_hyperparam_combinations(
        global_args=dict(feature_lags=feature_lags, batch_size=batch_sizes),
        model_args=dict(pred_len=target_lags, label_len=label_lens, output_attention=output_attentions,
                        enc_in=[input_size], d_model=hidden_sizes, dropout=dropouts, dec_in=[output_size],
                        embed_type=embed_types, factor=factors, e_layers=e_layers, activation=activations,
                        n_heads=n_heads, d_layers=d_layers, c_out=[output_size], distil=distil),
        model_shared_pairs={'d_ff': 'd_model'},
        model_args_key=model_args_key
    )
    autoformer_params = ml_utils.generate_hyperparam_combinations(
        global_args=dict(feature_lag=feature_lags, batch_size=batch_sizes),
        model_args=dict(pred_len=target_lags, label_len=label_lens, output_attention=output_attentions,
                        enc_in=[input_size], d_model=hidden_sizes, dropout=dropouts, dec_in=[input_size],
                        embed_type=embed_types, factor=factors, e_layers=e_layers, activation=activations,
                        n_heads=n_heads, d_layers=d_layers, c_out=[input_size], moving_avg=moving_avg,
                        output_size=[output_size]),
        model_shared_pairs={'d_ff': 'd_model'},
        model_args_key=model_args_key
    )
    fedformer_params = ml_utils.generate_hyperparam_combinations(
        global_args=dict(batch_size=batch_sizes),
        model_args=dict(feature_lags=feature_lags, target_lags=target_lags, label_len=label_lens,
                        output_attention=output_attentions, enc_in=[input_size], d_model=hidden_sizes, dropout=dropouts,
                        dec_in=[input_size], embed_type=embed_types, e_layers=e_layers, activation=activations,
                        n_heads=[8], d_layers=d_layers, c_out=[input_size], moving_avg=moving_avg,
                        version=fedformer_version, mode_select=fedformer_mode_select, modes=fedformer_n_modes,
                        output_size=[output_size]),
        model_shared_pairs={'d_ff': 'd_model'},
        model_args_key=model_args_key
    )
    linear_params = ml_utils.generate_hyperparam_combinations(
        global_args=dict(batch_size=batch_sizes),
        model_args=dict(seq_len=feature_lags, pred_len=target_lags, channels=[input_size], individual=individual,
                        output_size=[output_size]),
        model_args_key=model_args_key
    )
    for hyperparams in seq2seq_params:
        print(json.dumps(hyperparams, sort_keys=True, indent=4))
        model_class_name = seq2seq_name
        input_dim = (hyperparams['batch_size'], hyperparams['feature_lags'], input_size)
        if model_class_name == seq2seq_name:
            hyperparams[model_args_key]['input_dim'] = input_dim

        trainer = Trainer(
            features_path=forces_path,
            targets_path=kinematics_path,
            train_percent=train_percent,
            val_percent=val_percent,
            feature_win=input_dim[1],
            target_win=hyperparams[model_args_key]['target_lags'],
            intersect=intersect,
            batch_size=hyperparams['batch_size'],

            model_class_name=model_class_name,
            model_args=hyperparams[model_args_key],

            exp_name=f"[{data},T={hyperparams[model_args_key]['target_lags']}",
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
            regularization_factor=regularization_factor,
            hyperparams=utils.flatten_dict(hyperparams)
        )
        trainer.fit()
        _, loss_df = trainer.predict('test', False)
