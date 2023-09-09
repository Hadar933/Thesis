import os.path
from typing import Optional, Union, Dict, Tuple, Literal, List
import plotly.offline as pyo

import imageio
import numpy as np
import scipy
import json

from matplotlib import pyplot as plt
from plotly_resampler import FigureResampler
import pandas as pd
import plotly.graph_objects as go
import torch
from tqdm import tqdm

from ML.Core.datasets import MultiTimeSeries
from deprecated import deprecated

from ML.Zoo.seq2seq import Attention, Seq2Seq

TIME_FORMAT = '%Y-%m-%d_%H-%M-%S'


def tensor_mb_size(v: torch.Tensor):
    """ returns the size (in MB) of a given torch tensor"""
    return v.nelement() * v.element_size() / 1_000_000


def update_json(yaml_path: str, new_data):
    """ updates a given yaml file with new data dictionary """
    if os.path.exists(yaml_path):
        with open(yaml_path, "r", encoding='utf-8') as f:
            old_data = json.load(f)
    else:
        old_data = {}
    with open(yaml_path, "w", encoding='utf-8') as f:
        json.dump({**old_data, **new_data}, f, ensure_ascii=False, indent=4)


def set_time_index(df: pd.DataFrame, freq: float = 25.0):
    """ takes in a df and converts its index to time given the provided frequency"""
    start_time = pd.Timestamp('2023-04-24 00:00:00')
    end_time = start_time + pd.Timedelta(seconds=len(df) / freq)
    time_index = pd.date_range(start=start_time, end=end_time, periods=len(df))
    time_values = time_index.strftime('%H:%M:%S.%f').tolist()
    df.index = pd.to_datetime(time_values, format='%H:%M:%S.%f')
    return df


def plot(df: pd.DataFrame,
         title: Optional[str] = "Data vs Time",
         x_title: Optional[str] = "time / steps",
         y_title: Optional[str] = "Data",
         save_path: Optional[str] = None) -> None:
    """plots a df with plotly resampler"""
    fig = FigureResampler(go.Figure())
    for col in df.columns:
        fig.add_trace(go.Scattergl(name=col, showlegend=True), hf_x=df.index, hf_y=df[col])
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title,
                      margin=dict(l=20, r=20, t=30, b=0), height=700)
    if save_path is not None:
        fig.write_html(save_path)
    pyo.plot(fig, filename='temp-plot.html', auto_open=True)


def read_data(data_path: str) -> pd.DataFrame:
    """	reads a gzip file with possible 'ts' column and saves the df as the preprocessed data. """
    try:
        data = pd.read_csv(data_path, index_col=0, parse_dates=['ts'], compression="gzip")
    except ValueError:
        data = pd.read_csv(data_path, index_col=0, compression="gzip")
    return data


def load_data_from_prssm_paper(
        path: str = "G:\\My Drive\\Master\\Lab\\Thesis\\Preprocess\\flapping_wing_aerodynamics.mat",
        kinematics_key: Literal['ds_pos', 'ds_u_raw', 'ds_u'] = "ds_pos",
        forces_key: Literal['ds_y_raw', 'ds_y'] = "ds_y_raw",
        return_all: bool = False,
        forces_to_take: Optional[List[int]] = None) -> \
        Union[Dict, Tuple[torch.Tensor, torch.Tensor]]:
    """
    loads the data from the PRSSM paper, based on a string that represents the request
    :param path: path to the .mat file
    :param kinematics_key: one of the following options:
                        - 'ds_pos' (default): raw stroke, deviation and rotation
                        - 'ds_u_raw': The 7 kinematic variables derived from ds_pos
                        - 'ds_u': The standardized* versions of ds_u_raw
    :param forces_key: one of the following options:
                        - 'ds_y_raw' (default): raw targets Fx (normal), Fy (chord-wise), Mx, My, Mz
                        - 'ds_y': the standardized* version of ds_y_raw
    :param return_all: if true, returns the entire data dictionary
    :param forces_to_take: the forces columns to consider. If not specified, takes all of them (5)
    * beware - this is normalized w.r.t an unknown training data
    :return:
    """
    try:
        mat: Dict = scipy.io.loadmat(path)
    except FileNotFoundError:
        mat: Dict = scipy.io.loadmat("Preprocess/flapping_wing_aerodynamics.mat")
    if return_all: return mat
    kinematics = torch.Tensor(mat[kinematics_key])
    forces = torch.Tensor(mat[forces_key])
    forces = forces if forces_to_take is None else forces[:, :, forces_to_take]
    return kinematics, forces


def train_val_test_split(features: np.ndarray, targets: np.ndarray,
                         train_percent: float, val_percent: float,
                         feature_lag: int, target_lag: int, intersect: int,
                         batch_size: int,
                         features_normalizer, targets_normalizer) -> Dict:
    """
    creates a time series train-val-test split for multiple multivariate time series
    :param features: the data itself
    :param targets: the target(s)
    :param train_percent: percentage of data to be considered as training data
    :param val_percent: same, just for validation
    :param feature_lag: the number of samples in every window (history size)
    :param target_lag: the number of samples in the predicted value (usually one)
    :param intersect: an intersection between the feature and target lags
    :param batch_size: the batch size for the training/validation sets.
    :param features_normalizer: a normalizer object for the features of the training data
    :param targets_normalizer: a normalizer object for the targets of the training data
    :return: for the training and validation - regular pytorch loader. for the test - a loader for every dataset.
    """
    n_exp, hist_size, n_features = features.shape
    train_size = int(train_percent * n_exp)
    val_size = int(val_percent * n_exp)

    features_train, targets_train = features[:train_size], targets[:train_size]
    features_train = features_normalizer.fit_transform(features_train)  # statistics are set from training data
    targets_train = targets_normalizer.fit_transform(targets_train)  # same

    features_val, targets_val = features[train_size:train_size + val_size], targets[train_size:train_size + val_size]
    features_val = features_normalizer.transform(features_val)
    targets_val = targets_normalizer.transform(targets_val)

    features_test, targets_test = features[train_size + val_size:], targets[train_size + val_size:]
    features_test = features_normalizer.transform(features_test)
    targets_test = targets_normalizer.transform(targets_test)

    train_dataset = MultiTimeSeries(features_train, targets_train, feature_lag, target_lag, intersect)
    val_dataset = MultiTimeSeries(features_val, targets_val, feature_lag, target_lag, intersect)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # since a prediction should be on a single dataset, we create a dataset (and a dataloader) for each of
    # the test datasets in the test tensor. In inference time, we can simply choose the test dataloader
    # using its index:
    all_test_datasets = [MultiTimeSeries(features_test[i].unsqueeze(0), targets_test[i].unsqueeze(0), feature_lag,
                                         target_lag, intersect) for i in range(features_test.shape[0])]
    all_test_dataloaders = [torch.utils.data.DataLoader(all_test_datasets[i], batch_size=1, shuffle=False)
                            for i in range(len(all_test_datasets))]
    return {
        'train': {
            'data': train_dataset,
            'loader': train_loader
        },
        'val': {
            'data': val_dataset,
            'loader': val_loader
        },
        'test': {
            'data': all_test_datasets,
            'loader': all_test_dataloaders
        }
    }


def format_df_torch_entries(df: pd.DataFrame):
    """ takes in a df where every entry is a torch tensor and returns a new df with unpacked tensor values as cols """
    old_cols = df.columns

    def create_new_columns(row, col_name):
        tensor = row[col_name]
        new_cols = pd.Series(tensor.numpy())
        new_col_names = [f"{col_name}_{i}" for i in range(len(tensor))]
        return pd.Series(dict(zip(new_col_names, new_cols)))

    for col_name in df.columns:
        if col_name.startswith('pred_') or col_name.startswith('true_'):
            new_cols = df.apply(create_new_columns, args=(col_name,), axis=1)
            df[new_cols.columns] = new_cols
    df = df.drop(columns=old_cols, axis=1)
    return df


def visualize_attention(model: Seq2Seq, test_loader) -> pd.DataFrame:
    """

    :param model: a sequence to sequence torch model
    :param test_loader: a loader that contains test data
    :return:
    """
    model.train(False)
    device = torch.device('cpu')
    model = model.to(device)
    images = []
    x_start, x_end = -1, test_loader.dataset.feature_win - 1  # moving x axis
    with torch.no_grad():
        for i, (inputs_i, true_i) in tqdm(enumerate(test_loader), total=len(test_loader)):
            x_start += 1
            x_end += 1
            inputs_i = inputs_i.to(device)
            pred_i = model(inputs_i)
            x_vals = torch.arange(x_start, x_end)
            y_vals = inputs_i.squeeze(0)
            weights = model.attention.attn_weights.squeeze(0)
            image = plot_with_background(x_vals, y_vals, weights, 'time', 'value',
                                         f'Attention weights [{i}/{len(test_loader)}]')
            images.append(image)
    imageio.mimsave('attention_weights_over_time.mp4', images, fps=10)
    return images


def plot_with_background(x_values: torch.Tensor, y_values: torch.Tensor, w_values: torch.Tensor,
                         x_label, y_label, title):
    """
    plots a figure given x_values and y_values and colors the background using w_values
    :param x_values: x-axis values (time mostly) with shape (win_size,)
    :param y_values: y-axis values (forces mostly) with shape (win_size, input_size)
    :param w_values: color mapping (attention weights mostly) with shape (win_size,)
    :return: an RGB image that represents the colored window plot
    """
    margin = 0.1
    n_plots = y_values.shape[1]
    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(n_plots):  # plotting the data itself
        ax.plot_image(x_values, y_values[:, i])
    cmap = plt.get_cmap('coolwarm')
    for j in range(len(x_values) - 1):  # plotting the background
        color = cmap(w_values[j].item())
        ax.fill_betweenx([y_values.min() - margin, y_values.max() + margin], x_values[j], x_values[j + 1], color=color)
    # Add a colorbar as a color index
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Adjust position of the colorbar
    norm = plt.Normalize(w_values.min(), w_values.max())  # Normalize colorbar to original 'w' values
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label('w values')
    ax.set_xlim(x_values.min(), x_values.max())
    ax.set_ylim(y_values.min() - margin, y_values.max() + margin)
    ax.set_xlabel(x_label), ax.set_ylabel(y_label), ax.set_title(title)
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return image
