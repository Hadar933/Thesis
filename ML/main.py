import os
import json
from ML.Core.trainer import Trainer
import torch
from ML import ml_utils
from Utilities import utils

if __name__ == '__main__':

    # mostly unchanged parameters:
    exp_time = '22_11_2023'
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


    parent_dirname = r"E:\\Hadar\\experiments" if use_hard_drive else '../Results'
    forces_path = os.path.join(parent_dirname, exp_time, 'f19+f23_list_clean.pt')
    kinematics_path = os.path.join(parent_dirname, exp_time, 'k19+k23_list_clean.pt')
    forces, kinematics = torch.load(forces_path), torch.load(kinematics_path)
    if isinstance(forces, list) and isinstance(kinematics, list):
        input_size, output_size = forces[0].shape[-1], kinematics[0].shape[-1]
    else:
        input_size, output_size = forces.shape[-1], kinematics.shape[-1]

    model_args_key = 'model_args'
    feature_lags = [128, 256, 512]
    batch_sizes = [512]
    target_lags = [1, 16, 32, 64, 128]
    embedding_sizes = [2, 4, 8, 16]
    hidden_sizes = [4, 8, 16, 32, 64]
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
    individual = [True, False]

    seq2seq_params = ml_utils.generate_hyperparam_combinations(
        global_args=dict(feature_lag=feature_lags, batch_size=batch_sizes),
        model_args=dict(target_lag=target_lags, enc_embedding_size=hidden_sizes, enc_hidden_size=embedding_sizes,
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
        global_args=dict(feature_lag=feature_lags, batch_size=batch_sizes),
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
                        enc_in=[input_size], d_model=hidden_sizes, dropout=dropouts, dec_in=[output_size],
                        embed_type=embed_types, factor=factors, e_layers=e_layers, activation=activations,
                        n_heads=n_heads, d_layers=d_layers, c_out=[output_size], moving_avg=moving_avg),
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
        print(json.dumps(hyperparams,sort_keys=True,indent=4))
        model_class_name = seq2seq_name
        input_dim = (hyperparams['batch_size'], hyperparams['feature_lag'], input_size)
        if model_class_name == seq2seq_name:
            hyperparams[model_args_key]['input_dim'] = input_dim

        trainer = Trainer(
            features_path=forces_path,
            targets_path=kinematics_path,
            train_percent=train_percent,
            val_percent=val_percent,
            feature_win=input_dim[1],
            target_win=hyperparams[model_args_key]['target_lag'],
            intersect=intersect,
            batch_size=hyperparams['batch_size'],

            model_class_name=model_class_name,
            model_args=hyperparams[model_args_key],

            exp_name=f"ResidualReLU[ours,T={hyperparams[model_args_key]['target_lag']}]",
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
        # testpreds, test_loss = trainer.predict('test', True)
        # print(test_loss.to_string())
#
# #
# with open('valpreds_prssm.pkl', 'wb') as handle:
# 	pickle.dump(valpreds, handle)
# with open('tstpreds_prssm.pkl', 'wb') as handle:
# 	pickle.dump(testpreds, handle)
