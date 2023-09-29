from ML.Zoo import seq2seq
from ML.Core.trainer import Trainer
import torch
import torch.nn as nn
from torchinfo import summary

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
    criterion = nn.L1Loss()
    patience = 10
    patience_tolerance = 0.005
    features_norm = 'zscore'
    features_global_norm = True
    targets_norm = 'identity'
    targets_global_norm = True

    forces = torch.load('/home/hadar/Downloads/forces_small.pt')
    kinematics = torch.load('/home/hadar/Downloads/kinematics_small.pt')

    input_size, output_size = forces.shape[-1], kinematics.shape[-1]
    input_dim = (batch_size, feature_win, input_size)

    for emb_size in [8, 16, 32, 64, 128, 256]:
        for hid_size in [8, 16, 32, 64, 128, 256]:
            for fwin in [64, 128, 256, 512]:
                for nlayers in [1, 2, 3]:
                    exp_name = f""
                    model = seq2seq.Seq2Seq(
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
                    print(summary(model, input_size=(batch_size, feature_win, input_size)))
                    optimizer = torch.optim.Adam(model.parameters())

                    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                    trainer = Trainer(
                        features=forces,
                        targets=kinematics,
                        train_percent=train_percent,
                        val_percent=val_percent,
                        feature_win=feature_win,
                        target_win=target_win,
                        intersect=intersect,
                        batch_size=batch_size,
                        model=model,
                        exp_name=exp_name,
                        optimizer=optimizer,
                        criterion=criterion,
                        device=device,
                        patience=patience,
                        patience_tolerance=patience_tolerance,
                        n_epochs=n_epochs,
                        seed=seed,
                        features_norm_method=features_norm,
                        features_global_normalizer=features_global_norm,
                        targets_norm_method=targets_norm,
                        targets_global_normalizer=targets_global_norm
                    )
                    trainer.fit()

    trainer.predict()
