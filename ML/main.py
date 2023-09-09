from ML.Zoo import rnn, seq2seq
from ML.Core.trainer import Trainer
from ML.Utilities import utils
import torch
import torch.nn as nn
from torchinfo import summary


def init_trainer(
        forces: torch.Tensor,
        kinematics: torch.Tensor,
        train_percent=0.85,
        val_percent=0.1,
        feature_win=120,
        target_win=1,
        intersect=0,
        batch_size=64,
        n_epochs=30,
        seed=3407,
        criterion=nn.L1Loss(),
        patience=10,
        patience_tolerance=0.005,
        features_norm='zscore',
        features_global_norm=True,
        targets_norm='identity',
        targets_global_norm=True
) -> Trainer:
    input_size, output_size = forces.shape[-1], kinematics.shape[-1]
    input_dim = (batch_size, feature_win, input_size)
    model = seq2seq.Seq2Seq(input_dim, target_win, 5, 5, 1, True, 5, 5, output_size)
    optimizer = torch.optim.Adam(model.parameters())
    print(summary(model, input_size=(batch_size, feature_win, input_size)))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(forces, kinematics, train_percent, val_percent, feature_win, target_win, intersect, batch_size,
                      model, model.name, optimizer, criterion, device, patience, patience_tolerance, n_epochs, seed,
                      features_norm, targets_norm, features_global_norm, targets_global_norm)
    return trainer


if __name__ == '__main__':
    # 64 x 120 x 5
    kinematics, forces = utils.load_data_from_prssm_paper()
    x = 2
