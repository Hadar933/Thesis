from typing import Tuple
from torch.utils.data import Dataset
import torch


class MultiTimeSeries(Dataset):
    def __init__(self, features: torch.Tensor, targets: torch.Tensor,
                 feature_win: int, target_win: int, feature_target_intersection: int):
        """
        denote FL as feature lag, TL as target lag, intersection as I and history size as H. Assume we want to predict
        at time t+FL+1, the time windows we provide are:

                ALL WINDOW: [0, 1, 2, 3,  ..., t-1, t, t+1,           ...               H-4, H-3, H-2, H-1]
                FEATURE WINDOW:                    [t, t+1, ..., t-FL-1, t+FL]
                TARGET WINDOW:                               [t+FL-I ,...t+FL, t+FL+1 , ..., t+FL+TL]

        as per the number of windows per ds - for every t we create a window, but also leaving room for the first
        window (FL timestamps) and the last window (TL timestamps), minus their intersection + 1 (cut t_0=0)

        :param features: a (normalized) tensor with shape (N,H,F), where:
                         - N: number of datasets
                         - H: number of samples per dataset
                         - F: number of features per sample
        :param targets: a (normalized) tensor with shape (N,H,T) where T is the number of targets to predict
        :param feature_win: feature history to consider
        :param target_win: target future to predict
        :param feature_target_intersection: intersection between target and features
        """
        self.feature_win: int = feature_win
        self.target_win: int = target_win
        self.feature_target_intersect: int = feature_target_intersection
        self.normalized_features: torch.Tensor = features
        self.normalized_targets: torch.Tensor = targets
        self.n_datasets, self.n_samples_per_ds, self.n_features = self.normalized_features.shape
        self.n_windows_per_ds = self.n_samples_per_ds - self.feature_win + self.feature_target_intersect - self.target_win + 1

    def __len__(self) -> int:
        return self.n_datasets * self.n_windows_per_ds

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: maybe add random start index.
        ds_idx = idx // self.n_windows_per_ds
        win_idx = idx % self.n_windows_per_ds
        features_window = self.normalized_features[ds_idx, win_idx: win_idx + self.feature_win]
        target_window = self.normalized_targets[ds_idx, win_idx + self.feature_win - self.feature_target_intersect:
                                                        win_idx + self.feature_win - self.feature_target_intersect + self.target_win]
        return features_window, target_window


if __name__ == '__main__':
    k = torch.Tensor([[[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]],
                      [[8, 8], [9, 9], [10, 10], [11, 11], [12, 12]]])

    f = torch.Tensor([[[1], [2], [3], [4], [5]],
                      [[8], [9], [10], [11], [12]]])
    ds = MultiTimeSeries(k, f, 3, 2, 1)
    dl = torch.utils.data.DataLoader(ds)
    print(len(ds))
    for x, y in dl:
        z = 2
