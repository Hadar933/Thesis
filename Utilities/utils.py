import os.path
import re
import subprocess

import dash
import matplotlib
import torch
import imageio
import scipy
import json

import plotly.offline as pyo
import numpy as np
import pandas as pd
from dash import html, dcc, Output, Input

from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from plotly import graph_objects as go
from plotly_resampler import FigureResampler
from tqdm import tqdm
from typing import Literal, Optional

from ML.Core.datasets import FixedLenMultiTimeSeries, VariableLenMultiTimeSeries
from ML.Zoo.Seq2seq import Seq2seq

TIME_FORMAT = '%Y-%m-%d_%H-%M-%S'


# ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║                                               PLOTTING                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

def plot_fourier_reconstruction(x, fft, weights, reconstructed_x, fftfreq):
    """
    run when inside ASL forawrd, using:
    plot_fourier_reconstruction(x[0], fft[0], weights.squeeze(1).permute(0, 2, 1)[0], reconstructed_x[0], self.fftfreq)
    Plot the input signal, its Fourier transform, learned weights, and reconstructed signal."""
    # Convert tensors to NumPy arrays
    x_np = x.detach().cpu().numpy()
    fft_np = fft.detach().cpu().numpy()
    weights_np = weights.detach().cpu().numpy()
    reconstructed_x_np = reconstructed_x.detach().cpu().numpy()

    # Create Pandas DataFrames
    x_df = pd.DataFrame(x_np, columns=[fr'$x_{i}$' for i in range(1, x_np.shape[1] + 1)],
                        index=np.arange(len(x_np)) * (1 / 5000))
    fft_df = pd.DataFrame(np.abs(fft_np), columns=[fr'$|FFT[x]|_{i}$' for i in range(1, fft_np.shape[1] + 1)],
                          index=fftfreq.numpy())
    weights_df = pd.DataFrame(weights_np, columns=[fr'$W_{i}$' for i in range(1, weights_np.shape[1] + 1)],
                              index=fftfreq.numpy())
    reconstructed_x_df = pd.DataFrame(reconstructed_x_np,
                                      columns=[fr'$x_{i}$' for i in range(1, reconstructed_x_np.shape[1] + 1)],
                                      index=np.arange(len(reconstructed_x_np)) * (1 / 5000))

    # Plot using Pandas
    fig, axs = plt.subplots(4, 1, figsize=(10, 10))

    x_df.plot(ax=axs[0], title=r'Input $x$')
    axs[0].set_xlabel('Time [sec]')
    axs[0].set_ylabel('Normalized Force [N]')

    fft_df.plot(ax=axs[1], title=r'$|FFT[x]|$')
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('Magnitude')
    # axs[1].set_yscale('log')
    weights_df.plot.bar(ax=axs[2], title=r'Learned weights $W$')  # Changed to bar plot
    axs[2].set_xlabel('Frequency (Hz)')
    axs[2].set_ylabel('Magnitude')
    axs[2].tick_params(axis='x', rotation=45)  # Rotate x-axis labels by 45 degrees
    labels = [item.get_text() for item in axs[2].get_xticklabels()]
    # Beat them into submission and set them back again
    axs[2].set_xticklabels([str(round(float(label), 2)) for label in labels])
    reconstructed_x_df.plot(ax=axs[3], title=r'Reconstructed $x$')
    axs[3].set_xlabel('Time [sec]')
    axs[3].set_ylabel('Normalized Force [N]')
    for ax in axs:
        ax.legend(loc='upper right')
        ax.grid(True)
    plt.tight_layout()
    plt.show()


def plot_model_weights(trainer, attention_plot=False, fourier_plot=False, in_time=True, cmap='Blues'):
    """
    plot the weights and the test batch in the same figure
    """
    test_batch = next(iter(trainer.data_dict['test']['loader']))[0]  # (B,T,F)
    seq2seq = trainer.model.to('cpu')
    if in_time:
        _ = seq2seq(test_batch)
        fft_weights = seq2seq.adl1.fft_weights.detach().numpy().squeeze()  # (B,N_freqs,F)
        fft_freqs = seq2seq.adl1.fftfreq.detach().numpy().squeeze()  # (N_freqs,)
        attn_weights = seq2seq.attention.attn_weights.detach().numpy().squeeze()  # (B,T)
        # note that for test, batch samples are consecutive in time.
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        # Plot the Fourier weights
        im1 = axs[0].imshow(fft_weights[:, :, 3], cmap=cmap, interpolation='nearest', aspect='auto')
        axs[0].set_xlabel('Frequency (Hz)')
        axs[0].set_ylabel('Window')
        axs[0].set_title('Fourier Weights for Shifted Windows')
        fig.colorbar(im1, ax=axs[0])
        axs[0].set_xticks(range(len(fft_freqs)))
        axs[0].set_xticklabels(fft_freqs.astype(np.int16))
        axs[0].tick_params(axis='x', rotation=45)
        # for label in axs[0].xaxis.get_ticklabels()[::2]:
        #     label.set_visible(False)
        # Plot the attention weights
        im2 = axs[1].imshow(attn_weights, cmap=cmap, interpolation='nearest', aspect='auto',
                            norm=matplotlib.colors.LogNorm(vmin=attn_weights.min(), vmax=attn_weights.max()))
        axs[1].set_xlabel('Time Step')
        axs[1].set_ylabel('Window')
        axs[1].set_title('Attention Weights for Shifted Windows')
        fig.colorbar(im2, ax=axs[1])
        plt.tight_layout()
        plt.show()

    else:
        i = 0  # sample index
        x = test_batch[[i]]  # (1,T,F)
        _ = seq2seq(x)
        fft_weights = seq2seq.adl1.fft_weights.detach().numpy().squeeze()  # (N_freqs,F)
        fft_magnitude = seq2seq.adl1.fft_magnitude.detach().numpy().squeeze()  # (N_freqs,F)
        fft_freqs = seq2seq.adl1.fftfreq.detach().numpy().squeeze()  # (N_freqs,)
        attn_weights = seq2seq.attention.attn_weights.detach().numpy().squeeze()  # (T,)
        if fourier_plot:
            fig, axs = plt.subplots(1, fft_weights.shape[1], figsize=(16, 5))
            for i in range(fft_weights.shape[1]):  # Loop over each feature F
                im = axs[i].imshow(np.vstack(fft_weights[:, i]).T,
                                   extent=(0, len(fft_weights), fft_magnitude[:, i].min(), fft_magnitude[:, i].max()),
                                   aspect='auto', cmap=cmap)
                axs[i].plot(fft_magnitude[:, i])
                axs[i].set_title(fr'$|FFT[Input=F_{i + 1}]|$', fontsize=14)
                axs[i].set_xticks(range(len(fft_freqs)))
                axs[i].set_xticklabels(fft_freqs.astype(np.int16))
                axs[i].tick_params(axis='x', rotation=45)
                for label in axs[i].xaxis.get_ticklabels()[::2]:
                    label.set_visible(False)
            fig.text(0.5, 0.02, 'Frequency (Hz)', ha='center', va='center', fontsize=14)
            fig.text(0.01, 0.5, 'Amplitude', ha='center', va='center', rotation='vertical', fontsize=14)
            cbar = fig.colorbar(im, ax=axs[i])
            cbar.set_label('Weight Magnitude', fontsize=14)
            plt.tight_layout()
            plt.show()

        if attention_plot:
            plt.figure(figsize=(15, 5))
            plt.imshow(np.vstack(attn_weights).T, extent=(0, len(attn_weights), x.min(), x.max()), aspect='auto',
                       cmap=cmap)
            plt.plot(x.squeeze().numpy())
            plt.colorbar()
            plt.title(f'Attention-Weighted Input Window')
            plt.xlabel('Frame #')
            plt.ylabel('Amplitude')
            plt.legend(['F1', 'F2', 'F3', 'F4'])
            plt.show()


def plot_model_predictions(
        pred_dict: dict[int, pd.DataFrame],
        datasets: Optional[list[int]] = None,
        trim: Optional[int] = None,
        fontsize: int = 14
) -> None:
    """
    takes in a dictionary of datasets predictions and plots them side by side
    :param pred_dict: dictionary where each value is a df with inputs, true labels and predictions.
                      These are generated using the `.predict` method of a `Trainer` instance
    :param datasets: a list of dataset keys to plot (the dictionary will be trimmed accordingly)
    :param trim: an integer that slices each dataframe (all rows > trim are dropped)
    :param fontsize: fontsize for labels and legends
    """
    if datasets is not None:
        pred_dict = {key: value for key, value in pred_dict.items() if key in datasets}
    num_dataframes = len(pred_dict)
    fig, axs = plt.subplots(2, num_dataframes, sharex=True, sharey='row', figsize=(len(pred_dict) * num_dataframes, 10))
    force_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:brown']
    pred_true_colors = ['tab:purple', 'tab:cyan', 'tab:pink']
    for idx, (df_name, df) in enumerate(pred_dict.items()):
        df = df.iloc[:trim, :] if trim is not None else df
        inputs_cols = [col for col in df.columns if 'input' in col.lower()]
        true_labels_cols = [col for col in df.columns if 'true' in col.lower()]
        predictions_cols = [col for col in df.columns if 'pred' in col.lower()]
        inputs = df[inputs_cols].values
        true_labels = df[true_labels_cols].values
        predictions = df[predictions_cols].values
        # Plotting inputs in the top row:
        input_names = ['Fx', 'Fy', 'Fz', 'My', 'Mz'] if len(inputs_cols) == 5 else ['F1', 'F2', 'F3', 'F4']
        for i, label in zip(range(inputs.shape[1]), input_names):
            axs[0, idx].plot(inputs[:, i], label=label, color=force_colors[i])
        # Plotting targets and predictions in the bottom row:
        labels = [r"$\theta$", r"$\phi$", r"$\psi$"]
        nan_index = np.where(np.isnan(true_labels[:, 0]))[0][0]
        for i in range(true_labels.shape[1]):
            axs[1, idx].plot(true_labels[:nan_index, i], label=f'True {labels[i]}', linestyle='dashed',
                             color=pred_true_colors[i])
            axs[1, idx].plot(predictions[:nan_index, i], label=f'Pred {labels[i]}', linestyle='solid',
                             color=pred_true_colors[i])
        axs[1, idx].set_xlim(0, nan_index)
    axs[0, 0].set_ylabel('Force and Torque Values [N] or [Nm]', fontsize=fontsize)
    axs[0, idx].legend(loc='lower right', fontsize=fontsize)
    axs[1, idx].legend(loc='lower right', fontsize=fontsize)
    axs[1, 0].set_ylabel('Angle Targets & Predictions [rad]', fontsize=fontsize)
    trim_suffix = f" (Trimmed to {trim} samples)" if trim is not None else ""
    fig.suptitle(f"Forces, Kinematics, and Predictions across Multiple Datasets {trim_suffix}", fontsize=fontsize + 2)
    fig.text(0.5, 0.04, 'Frames', ha='center', fontsize=fontsize)
    plt.tight_layout(rect=[0, 0.06, 1, 0.99])
    for ax in axs.flatten():        ax.grid(True)
    plt.show()


def results_plotter(
        kinematics_path: str,
        forces_path: str
) -> None:
    """ an interactive dash plot for the kinematics and forces torch tensors """
    # matplotlib.use('TkAgg')

    kinematics = torch.load(kinematics_path)
    forces = torch.load(forces_path)
    app = dash.Dash(__name__)
    app.layout = html.Div([
        html.H1("Kinematics and Forces Data Visualization"),
        dcc.Dropdown(
            id='dataset-dropdown',
            options=[{'label': f'Dataset {i}', 'value': i} for i in
                     range(len(kinematics) if isinstance(kinematics, list) else kinematics.size(0))],
            value=0  # Default value
        ),
        dcc.Graph(id='combined-graph')
    ])

    @app.callback(
        Output('combined-graph', 'figure'),
        [Input('dataset-dropdown', 'value')]
    )
    def update_graph(selected_dataset):
        """ updates the graph based on the dataset index chosen in the dropdown menu """
        phi_trace = go.Scatter(y=kinematics[selected_dataset][:, 0].numpy(), mode='lines', name=r'phi')
        psi_trace = go.Scatter(y=kinematics[selected_dataset][:, 1].numpy(), mode='lines', name=r'psi')
        theta_trace = go.Scatter(y=kinematics[selected_dataset][:, 2].numpy(), mode='lines', name=r'theta')
        force_traces = [go.Scatter(y=forces[selected_dataset][:, i].numpy(), mode='lines', name=fr'F_{i + 1}')
                        for i in range(4)]
        traces = [phi_trace, psi_trace, theta_trace] + force_traces
        layout = go.Layout(
            title=f'Kinematics and Forces for Experiment {selected_dataset}',
            xaxis=dict(title='Sample'),
            yaxis=dict(title='Value')
        )
        figure = {'data': traces, 'layout': layout}
        return figure

    app.run_server(debug=True)


def plot_df_with_plotly(
        df_dict: dict[int, pd.DataFrame],
        ignore_cols: list[str] | None = None,
        title: str = "Data vs Time",
        x_title: str = "time / steps",
        y_title: str = "Data",
        save_path: str | None = None
) -> None:
    """plots a df with plotly resampler"""
    df = pd.concat(df_dict.values(), axis=1)
    if ignore_cols is None:
        ignore_cols = []
    fig = FigureResampler(go.Figure())
    cols_to_plot = [col for col in df.columns if col not in ignore_cols]
    if isinstance(df.index, pd.TimedeltaIndex):
        x_axis = df.index.total_seconds()
        x_title = 'time [sec]'
    else:
        x_axis = df.index
    for col in cols_to_plot:
        fig.add_trace(go.Scattergl(name=col, showlegend=True), hf_x=x_axis, hf_y=df[col])
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, margin=dict(l=20, r=20, t=30, b=0),
                      height=700)
    if save_path is not None:
        fig.write_html(save_path)
    pyo.plot(fig, filename='temp-plot.html', auto_open=True)


def plot_dict_with_plotly(
        data: dict[str, list],
        title: str = "Data vs Time",
        x_title: str = "time / steps",
        y_title: str = "Data",
        save_path: str | None = None
) -> None:
    """plots a df with plotly resampler"""
    fig = FigureResampler(go.Figure())
    for key in data.keys():
        fig.add_trace(go.Scattergl(name=key, showlegend=True), hf_x=data.index, hf_y=data[key])
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, margin=dict(l=20, r=20, t=30, b=0),
                      height=700)
    if save_path is not None:
        fig.write_html(save_path)
    pyo.plot(fig, filename='temp-plot.html', auto_open=True)


def visualize_attention(
        model: Seq2seq,
        test_loader
) -> list[np.ndarray]:
    """
    @deprecated
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
            _ = model(inputs_i)
            x_vals = torch.arange(x_start, x_end)
            y_vals = inputs_i.squeeze(0)[0]
            weights = model.attention.attn_weights.squeeze(0)[0]
            image = plot_with_background(x_vals, y_vals, weights, 'time', 'value',
                                         f'Attention weights [{i}/{len(test_loader)}]')
            images.append(image)
            plt.imshow(image)
            plt.show()
    imageio.mimsave('attention_weights_over_time.mp4', images, fps=10)
    return images


def plot_with_background(
        x_values: torch.Tensor,
        y_values: torch.Tensor,
        w_values: torch.Tensor,
        x_label: str,
        y_label: str,
        title: str
) -> np.ndarray:
    """
    @deprecated
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
        ax.plot(x_values, y_values[:, i])
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


# ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║                                      DATA SPLITTING AND LOADING                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

def load_data_from_prssm_paper(
        path: str = "G:\\My Drive\\Master\\Lab\\Thesis\\DataHandler\\flapping_wing_aerodynamics.mat",
        kinematics_key: Literal['ds_pos', 'ds_u_raw', 'ds_u'] = "ds_pos",
        forces_key: Literal['ds_y_raw', 'ds_y'] = "ds_y_raw",
        return_all: bool = False,
        forces_to_take: list[int] | None = None
) -> dict | tuple[torch.Tensor, torch.Tensor]:
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
        mat: dict = scipy.io.loadmat(path)
    except FileNotFoundError:
        mat: dict = scipy.io.loadmat("DataHandler/flapping_wing_aerodynamics.mat")
    if return_all: return mat
    kinematics = torch.Tensor(mat[kinematics_key])
    forces = torch.Tensor(mat[forces_key])
    forces = forces if forces_to_take is None else forces[:, :, forces_to_take]
    return kinematics, forces


def train_val_test_split(
        use_variable_length_dataset: bool,
        features: torch.Tensor,
        targets: torch.Tensor,
        train_percent: float,
        val_percent: float,
        feature_window_size: int,
        target_window_size: int,
        intersect: int,
        batch_size: int,
        features_normalizer,
        targets_normalizer
) -> dict:
    """
    creates a time series train-val-test dataset-split (not history) for multiple multivariate time series.
    since a prediction should be on a single dataset, we create a dataset (and a dataloader) for each of
    the datasets. In inference time, we can simply choose the dataloader using its index.
    :param use_variable_length_dataset: if true, uses a dataset that is suited for list of tensors [t1,t2,...,tn],
                                        with each ti shaped as (hi,f). Otherwise, uses a dataset that handles a
                                        multi-dim tensor with shape (n,h,f), i.e fixed history for all n tensors.
    :param features: the data itself
    :param targets: the target(s)
    :param train_percent: percentage of data to be considered as training data
    :param val_percent: same, just for validation
    :param feature_window_size: the number of samples in every window (history size)
    :param target_window_size: the number of samples in the predicted value (usually one)
    :param intersect: the intersection size between the feature and target windows
    :param batch_size: the batch size for the training/validation sets.
    :param features_normalizer: a normalizer object for the features of the training data
    :param targets_normalizer: a normalizer object for the targets of the training data
    :return: for the training and validation - regular pytorch loader. for the test - a loader for every dataset.
    """
    DatasetClass = VariableLenMultiTimeSeries if use_variable_length_dataset else FixedLenMultiTimeSeries
    n_datasets = len(features) if use_variable_length_dataset else features.shape[0]
    train_size = int(train_percent * n_datasets)
    val_size = int(val_percent * n_datasets)

    # train loaders and per-dataset class:
    features_train, targets_train = features[:train_size], targets[:train_size]
    features_train = features_normalizer.fit_transform(features_train)  # statistics are set from training data
    targets_train = targets_normalizer.fit_transform(targets_train)  # same
    train_dataset = DatasetClass(features_train, targets_train, feature_window_size, target_window_size, intersect)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    all_train_datasets = [
        DatasetClass(features_train[i].unsqueeze(0), targets_train[i].unsqueeze(0), feature_window_size,
                     target_window_size, intersect) for i in range(train_size)
    ]
    all_train_dataloaders = [
        torch.utils.data.DataLoader(all_train_datasets[i], batch_size=1, shuffle=False) for i in
        range(len(all_train_datasets))
    ]

    # validation loaders and per-dataset class:
    features_val, targets_val = features[train_size:train_size + val_size], targets[train_size:train_size + val_size]
    features_val = features_normalizer.transform(features_val)
    targets_val = targets_normalizer.transform(targets_val)
    val_dataset = DatasetClass(features_val, targets_val, feature_window_size, target_window_size, intersect)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    all_val_datasets = [
        DatasetClass(features_val[i].unsqueeze(0), targets_val[i].unsqueeze(0), feature_window_size,
                     target_window_size, intersect) for i in range(val_size)
    ]
    all_val_dataloaders = [
        torch.utils.data.DataLoader(all_val_datasets[i], batch_size=1, shuffle=False) for i in
        range(len(all_val_datasets))
    ]

    # test loaders and per-dataset class:
    features_test, targets_test = features[train_size + val_size:], targets[train_size + val_size:]
    test_size = len(features_test) if isinstance(features_test, list) else features_test.shape[0]
    features_test = features_normalizer.transform(features_test)
    targets_test = targets_normalizer.transform(targets_test)
    test_dataset = DatasetClass(features_test, targets_test, feature_window_size, target_window_size, intersect)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    all_test_datasets = [
        DatasetClass(features_test[i].unsqueeze(0), targets_test[i].unsqueeze(0),
                     feature_window_size, target_window_size, intersect) for i in range(test_size)
    ]
    all_test_dataloaders = [
        torch.utils.data.DataLoader(all_test_datasets[i], batch_size=1, shuffle=False)
        for i in range(len(all_test_datasets))
    ]
    return {
        'train': {
            'data': train_dataset,
            'loader': train_loader,
            'all_datasets': all_train_datasets,
            'all_dataloaders': all_train_dataloaders
        },
        'val': {
            'data': val_dataset,
            'loader': val_loader,
            'all_datasets': all_val_datasets,
            'all_dataloaders': all_val_dataloaders
        },
        'test': {
            'data': test_dataset,
            'loader': test_loader,
            'all_datasets': all_test_datasets,
            'all_dataloaders': all_test_dataloaders
        }

    }


# ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║                                                  OTHER                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
def get_gpu_usage_percentage() -> float:
    """
    returns the GPU percentage usage, used inside a forward, as part ot tqdm logs
    :return: gpu percentage
    """
    output = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits']
    ).decode('utf-8').strip()
    gpu_memory_usage, gpu_memory_total = map(int, output.split(', '))
    gpu_memory_usage_percent = (gpu_memory_usage / gpu_memory_total) * 100
    return round(gpu_memory_usage_percent, 3)


def flatten_dict(d, parent_key='', sep='_'):
    """
    Flatten a nested dictionary.

    Args:
        d (dict): Nested dictionary.
        parent_key (str): Prefix to use for keys.
        sep (str): Separator between parent and child keys.

    Returns:
        dict: Flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def save_np_to_matlab_mat(
        data: np.ndarray,
        save_path: str,
        key_name: Optional[str] = None
) -> None:
    scipy.io.savemat(save_path, {key_name if key_name else 'data': data})


def mpl():
    """ boilerplate code for matplotlib importing and tkagg usage """
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('TkAgg')


def update_json(yaml_path: str, new_data):
    """ updates a given yaml file with new data dictionary """
    if os.path.exists(yaml_path):
        with open(yaml_path, "r", encoding='utf-8') as f:
            old_data = json.load(f)
    else:
        old_data = {}
    with open(yaml_path, "w", encoding='utf-8') as f:
        json.dump({**old_data, **new_data}, f, ensure_ascii=False, indent=4)


def _format_df_torch_entries(
        df: pd.DataFrame
) -> pd.DataFrame:
    """ takes in a df where every entry is a torch tensor and returns a new df with unpacked tensor values as cols """
    old_cols = df.columns.tolist()
    new_df = pd.DataFrame()

    def create_new_columns(row, col_name):
        tensor = row[col_name]
        if tensor.dim() == 0:  # tensor is scalar
            new_col_names = [col_name]
            new_cols = [tensor.item()]
        else:
            new_col_names = [f"{col_name}_{i}" for i in range(len(tensor))]
            new_cols = tensor.numpy().tolist()
        return pd.Series(new_cols, index=new_col_names)

    for col_name in old_cols:
        if col_name.startswith('pred_') or col_name.startswith('true_'):
            new_cols = df.apply(lambda row: create_new_columns(row, col_name), axis=1)
            new_df = pd.concat([new_df, new_cols], axis=1)

    return new_df


def remove_and_trim_datasets(
        kinematics_path: str,
        forces_path: str,
        save: bool = True,
        basic_plot: bool = True,
        keep_as_list: bool = True
) -> None:
    """
    a basic function that plots the data and allows to slice it and/or remove it altogether.
    :param kinematics_path: path to tensor or list of tensors kinematics
    :param forces_path: path to tensor or list of tensors forces
    :param save: if true, saves the results
    :param basic_plot: if true, plot a jpg, otherwise plots an interactive plot on html
    :param keep_as_list: if false, stacks as torch tensor, otherwise keeps values in list (for var-len dataset)
    """
    forces_to_keep = []
    kinematics_to_keep = []
    forces = torch.load(forces_path)
    kinematics = torch.load(kinematics_path)
    for dataset_idx in tqdm(range(len(forces))):
        curr = torch.cat((forces[dataset_idx], kinematics[dataset_idx]), dim=1)
        df = pd.DataFrame(curr, columns='f1 f2 f3 f4 theta phi psi'.split())
        if basic_plot:
            df.plot()
            xticks_location = np.linspace(df.index.min(), df.index.max(), num=20)  # 20 ticks
            plt.xticks(xticks_location, rotation=45)
            plt.grid()
            plt.show()
        else:
            plot_df_with_plotly(df)
        if input('Is this one bad? [y/Any]') == 'y':
            continue
        curr_force = forces[dataset_idx]
        curr_kinematics = kinematics[dataset_idx]
        trim_suffix_index = input('add suffix index: ')
        if trim_suffix_index:
            curr_force = curr_force[:int(trim_suffix_index)]
            curr_kinematics = curr_kinematics[:int(trim_suffix_index)]
        forces_to_keep.append(curr_force)
        kinematics_to_keep.append(curr_kinematics)

    if not keep_as_list:
        forces_to_keep = torch.stack(forces_to_keep)
        kinematics_to_keep = torch.stack(kinematics_to_keep)
    if save:
        torch.save(forces_to_keep, forces_path.replace('.pt', '_cleaned.pt'))
        torch.save(kinematics_to_keep, kinematics_path.replace('.pt', '_cleaned.pt'))


def from_loss_logs_to_df(all_logs_combined: str, plot,split_by):
    loss_logs_list = all_logs_combined.split(split_by)
    dfs = []
    for loss_logs in loss_logs_list:
        lines = loss_logs.split('\n')
        names = lines[0].replace('data #', 'data_#')
        space_remover = lambda s: re.sub(r'\s+', ' ', s)
        names = space_remover(names).split(' ')[1:]
        vals = space_remover(lines[1][1:])[1:].split(' ')
        vals = [float(v) for v in vals]
        if vals[0] == 0.0: vals = vals[1:]
        df = pd.DataFrame([vals], columns=names)
        dfs.append(df)
    # concat all dfs:
    df = pd.concat(dfs, axis=0)
    df.index = ['Seq2Seq', 'Seq2Seq+ASL', 'Transformer', 'Linear', 'AutoFormer', 'FedFormer', 'NLinear']
    if plot:
        plt.figure(figsize=(10, 6))
        for i, model in enumerate(df.index):
            # Get the color of the scatter plot for the current model
            color = plt.gca()._get_lines.get_next_color()

            plt.scatter(range(len(df.columns)), df.loc[model],
                        label=f'{model} (Mean MAE: {df.loc[model, "mean"]:.4f})', color=color)

            # Plot a horizontal line at the mean loss for each model with the same color as the scatter plot
            plt.axhline(y=df.loc[model, "mean"], color=color, linestyle='--')

            # Add the mean loss as text next to the horizontal line

        plt.title('HUJI dataset full results')
        plt.xlabel('Dataset #')
        plt.ylabel('Test MAE Loss')
        plt.legend()
        # plt.grid(linestyle='-.', linewidth=0.5)
        ax = plt.gca()
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.xaxis.set_minor_locator(AutoMinorLocator(1))
        ax.yaxis.set_minor_locator(AutoMinorLocator(1))
        ax.grid(which='major', color='#CCCCCC', linestyle='--')
        ax.grid(which='minor', color='#CCCCCC', linestyle=':')
        plt.show()
    return df


if __name__ == '__main__':
    # results_plotter(
    #     kinematics_path=r'G:\My Drive\Master\Lab\Thesis\Results\22_11_2023\k19+k23_list_clean.pt',
    #     forces_path=r'G:\My Drive\Master\Lab\Thesis\Results\22_11_2023\f19+f23_list_clean.pt'
    # )
    prssm_s = """     data #0   data #1   data #2   data #3  data #4   data #5   data #6   data #7  data #8   data #9  data #10  data #11  data #12  data #13  data #14  data #15  data #16  data #17  data #18  data #19  data #20  data #21  data #22  data #23  data #24  data #25  data #26  data #27  data #28  data #29  data #30  data #31  data #32  data #33  data #34  data #35  data #36  data #37  data #38  data #39  data #40  data #41  data #42  data #43  data #44  data #45  data #46  data #47  data #48  data #49  data #50  data #51  data #52  data #53  data #54  data #55  data #56  data #57  data #58  data #59  data #60  data #61  data #62  data #63  data #64  data #65  data #66  data #67  data #68  data #69  data #70  data #71  data #72  data #73  data #74  data #75  data #76  data #77  data #78  data #79  data #80  data #81  data #82      mean
    0  0.178047  0.095764  0.180324  0.116165  0.05972  0.111333  0.104855  0.137168  0.07822  0.080676  0.061711  0.066297  0.084862  0.112144   0.11375  0.111365  0.083885  0.115646  0.101228  0.103461  0.093994  0.215954    0.0698  0.055788    0.1177  0.069576   0.17767   0.19849  0.142092  0.078826   0.11851  0.106562  0.096415  0.128049  0.124021  0.101642  0.081163  0.080024  0.201846  0.100258  0.068952  0.248882  0.194381  0.076436  0.088656  0.105836  0.151157  0.348589  0.194523  0.074229  0.140929  0.217614  0.061466  0.084536  0.073826  0.080148  0.121691   0.06551  0.137895  0.191746   0.19094  0.095801  0.070023  0.191113  0.158693  0.175163  0.157419  0.069935  0.145975   0.05193  0.128912  0.077256  0.091133  0.203262  0.151977  0.063592  0.061561  0.161658  0.259844  0.111664  0.041969  0.069997  0.094978  0.120564


        data #0   data #1   data #2   data #3  data #4   data #5   data #6   data #7   data #8   data #9  data #10  data #11  data #12  data #13  data #14  data #15  data #16  data #17  data #18  data #19  data #20  data #21  data #22  data #23  data #24  data #25  data #26  data #27  data #28  data #29  data #30  data #31  data #32  data #33  data #34  data #35  data #36  data #37  data #38  data #39  data #40  data #41  data #42  data #43  data #44  data #45  data #46  data #47  data #48  data #49  data #50  data #51  data #52  data #53  data #54  data #55  data #56  data #57  data #58  data #59  data #60  data #61  data #62  data #63  data #64  data #65  data #66  data #67  data #68  data #69  data #70  data #71  data #72  data #73  data #74  data #75  data #76  data #77  data #78  data #79  data #80  data #81  data #82      mean
    0  0.20791  0.081664  0.209284  0.089274  0.05883  0.121112  0.085132  0.101649  0.043795  0.056567  0.067325  0.092109  0.084692  0.094667  0.085846  0.087085  0.066862   0.13182  0.156417  0.064155  0.083774  0.211993   0.05706  0.063275  0.089015    0.0638  0.150921  0.161651  0.169465  0.076565  0.127879  0.097718   0.10299    0.0732  0.067439  0.079699  0.109214  0.066921  0.171481   0.09876  0.057326  0.309726  0.149901  0.096408  0.084661  0.085216   0.20113   0.28086  0.161503    0.0815  0.119811  0.255166  0.055522  0.069332  0.076502  0.053237  0.091607  0.075226  0.090782  0.139398  0.187925   0.08312  0.051984  0.129674  0.216921  0.188106  0.173118  0.066245  0.133781  0.071204  0.111184   0.10874  0.089847  0.212182  0.148429  0.061615  0.054833  0.096425  0.252435   0.09083  0.033266  0.101091  0.076229  0.113048


         data #0   data #1  data #2   data #3   data #4   data #5   data #6   data #7   data #8   data #9  data #10  data #11  data #12  data #13  data #14  data #15  data #16  data #17  data #18  data #19  data #20  data #21  data #22  data #23  data #24  data #25  data #26  data #27  data #28  data #29  data #30  data #31  data #32  data #33  data #34  data #35  data #36  data #37  data #38  data #39  data #40  data #41  data #42  data #43  data #44  data #45  data #46  data #47  data #48  data #49  data #50  data #51  data #52  data #53  data #54  data #55  data #56  data #57  data #58  data #59  data #60  data #61  data #62  data #63  data #64  data #65  data #66  data #67  data #68  data #69  data #70  data #71  data #72  data #73  data #74  data #75  data #76  data #77  data #78  data #79  data #80  data #81  data #82      mean
    0  0.122503  0.123435  0.18562  0.080379  0.061239  0.078981  0.087159  0.093659  0.067068  0.046348  0.057992  0.099522  0.107748  0.114049  0.106229  0.099402  0.100387  0.141185  0.201585  0.091504   0.12509  0.171201  0.065931  0.050305  0.180684  0.054766  0.132368  0.158341  0.145832  0.146208  0.085681  0.071938  0.102309  0.104489  0.100541  0.106391  0.109937  0.102104  0.097065  0.129912   0.08454  0.191641  0.145066  0.084645  0.087462  0.120575  0.153349  0.382267  0.155903  0.099555  0.120397  0.099531  0.050807  0.078977  0.096465    0.1032  0.142622  0.064706  0.115186  0.118132  0.187377  0.100054  0.074401  0.088341  0.151629  0.167424   0.15754   0.07608  0.072416  0.047998  0.079781  0.068397  0.154957  0.155312  0.124004  0.088504  0.077313  0.125295  0.145788  0.092133  0.067533  0.123974  0.096351  0.112346


         data #0   data #1   data #2   data #3   data #4   data #5   data #6   data #7   data #8   data #9  data #10  data #11  data #12  data #13  data #14  data #15  data #16  data #17  data #18  data #19  data #20  data #21  data #22  data #23  data #24  data #25  data #26  data #27  data #28  data #29  data #30  data #31  data #32  data #33  data #34  data #35  data #36  data #37  data #38  data #39  data #40  data #41  data #42  data #43  data #44  data #45  data #46  data #47  data #48  data #49  data #50  data #51  data #52  data #53  data #54  data #55  data #56  data #57  data #58  data #59  data #60  data #61  data #62  data #63  data #64  data #65  data #66  data #67  data #68  data #69  data #70  data #71  data #72  data #73  data #74  data #75  data #76  data #77  data #78  data #79  data #80  data #81  data #82      mean
    0  0.282954  0.501738  0.595101  0.256239  0.136843  0.330183  0.277795  0.357621  0.150389  0.150204  0.146562  0.351188  0.165193  0.297421  0.366744  0.378302  0.293261  0.357638  0.671955  0.306996  0.233845  0.301669  0.333955  0.102953  0.259042  0.196786  0.466356  0.326616   0.48987  0.257676  0.387164  0.327792  0.240212  0.433598  0.367161  0.229464  0.314868  0.339983  0.186179  0.298746  0.230575  0.512518  0.413995  0.333317  0.266317  0.378866    0.4665  0.579339  0.416366  0.354535  0.273695  0.437104  0.186135  0.216837  0.293652  0.195357  0.280866  0.184071  0.408832  0.408913  0.501537  0.156804  0.142156   0.21325  0.472827  0.299477  0.489454  0.258122  0.448537  0.112634  0.209925  0.283617  0.376812  0.312003  0.581441    0.3511   0.13743  0.463447  0.540009  0.263558   0.09577  0.255341  0.476009  0.322257


         data #0   data #1   data #2   data #3   data #4  data #5   data #6   data #7   data #8   data #9  data #10  data #11  data #12  data #13  data #14  data #15  data #16  data #17  data #18  data #19  data #20  data #21  data #22  data #23  data #24  data #25  data #26  data #27  data #28  data #29  data #30  data #31  data #32  data #33  data #34  data #35  data #36  data #37  data #38  data #39  data #40  data #41  data #42  data #43  data #44  data #45  data #46  data #47  data #48  data #49  data #50  data #51  data #52  data #53  data #54  data #55  data #56  data #57  data #58  data #59  data #60  data #61  data #62  data #63  data #64  data #65  data #66  data #67  data #68  data #69  data #70  data #71  data #72  data #73  data #74  data #75  data #76  data #77  data #78  data #79  data #80  data #81  data #82      mean
    0  0.221723  0.467217  0.492516  0.299602  0.211925  0.38257  0.274239  0.486108  0.282366  0.264313  0.183768  0.402499  0.336111  0.377345  0.484888  0.276063  0.399767  0.395742  0.443178  0.277376  0.390534  0.385608  0.333455  0.209183  0.222061  0.337862  0.685958  0.397551  0.458971  0.308929  0.373732   0.30239  0.309881  0.359679  0.372085  0.245653  0.298905  0.359265  0.298145  0.384105  0.253587  0.474498  0.437101  0.382302  0.303426  0.369024  0.500897  0.713499  0.396282   0.25427  0.300817  0.506906  0.183351  0.337125  0.311931  0.214759  0.329701   0.29033  0.497756  0.555592  0.429552  0.247244  0.270991  0.287714  0.572664  0.381544  0.655971  0.250485  0.392697  0.234068   0.24974  0.266753  0.587882  0.377904  0.426863   0.29114  0.227097  0.565735  0.571607  0.395232  0.189791  0.283433  0.336701  0.362641


         data #0   data #1   data #2   data #3   data #4   data #5   data #6   data #7   data #8   data #9  data #10  data #11  data #12  data #13  data #14  data #15  data #16  data #17  data #18  data #19  data #20  data #21  data #22  data #23  data #24  data #25  data #26  data #27  data #28  data #29  data #30  data #31  data #32  data #33  data #34  data #35  data #36  data #37  data #38  data #39  data #40  data #41  data #42  data #43  data #44  data #45  data #46  data #47  data #48  data #49  data #50  data #51  data #52  data #53  data #54  data #55  data #56  data #57  data #58  data #59  data #60  data #61  data #62  data #63  data #64  data #65  data #66  data #67  data #68  data #69  data #70  data #71  data #72  data #73  data #74  data #75  data #76  data #77  data #78  data #79  data #80  data #81  data #82      mean
    0  0.166809  0.287906  0.489825  0.250832  0.157821  0.324156  0.156286  0.382185  0.269114  0.184581  0.232171   0.36749  0.254355  0.370278  0.483038  0.153456  0.324411  0.271949  0.313254  0.287479  0.330518  0.335333  0.293981  0.122068  0.267988  0.276641  0.568445  0.315813  0.313232  0.291757  0.249421  0.276086  0.263943  0.322617  0.407317  0.253275  0.196023  0.202512  0.234661  0.336397  0.205806  0.488453  0.399411  0.365402  0.287836  0.243464  0.430554  0.733181  0.364985  0.332638  0.269979  0.339864  0.177805  0.237939  0.225427  0.214505  0.281234  0.253628  0.413862  0.338618  0.355197  0.131224  0.245964  0.234533  0.470541  0.378905  0.422985  0.223654   0.36123  0.148734  0.172738    0.2064  0.395573  0.273341  0.367262  0.213146  0.125503  0.484674  0.442268  0.362616  0.158366  0.308101  0.253378  0.300317


         data #0   data #1   data #2   data #3   data #4   data #5   data #6   data #7   data #8   data #9  data #10  data #11  data #12  data #13  data #14  data #15  data #16  data #17  data #18  data #19  data #20  data #21  data #22  data #23  data #24  data #25  data #26  data #27  data #28  data #29  data #30  data #31  data #32  data #33  data #34  data #35  data #36  data #37  data #38  data #39  data #40  data #41  data #42  data #43  data #44  data #45  data #46  data #47  data #48  data #49  data #50  data #51  data #52  data #53  data #54  data #55  data #56  data #57  data #58  data #59  data #60  data #61  data #62  data #63  data #64  data #65  data #66  data #67  data #68  data #69  data #70  data #71  data #72  data #73  data #74  data #75  data #76  data #77  data #78  data #79  data #80  data #81  data #82      mean
    0  0.355295  0.447078  0.605292  0.299362  0.119545  0.325483  0.275471  0.313108  0.169931  0.192578  0.149315  0.361516  0.166396  0.324691  0.385442  0.343948  0.268275  0.300183  0.623656  0.293435  0.227285  0.361408  0.336981  0.110956  0.259253  0.214704  0.418059  0.353485  0.477976  0.249789  0.393479  0.342199  0.325514  0.453119  0.372487  0.223497  0.329047  0.327273  0.243475  0.268923  0.254866  0.528281  0.413817  0.298403  0.244043  0.412318  0.486765  0.728958  0.403122  0.286482  0.291869  0.379816  0.176966   0.26987  0.269095  0.191394  0.295463  0.210054  0.375944  0.434953  0.498712  0.168045  0.144558  0.245915  0.454808  0.285835   0.46822  0.256874  0.421938  0.121257  0.228417  0.300292  0.408412  0.337081  0.557003  0.357633  0.176997  0.447374  0.513309   0.24888  0.104155  0.298393   0.43308  0.325766
    """
    ours_s = """     data #0   data #1   data #2   data #3   data #4   data #5   data #6   data #7   data #8  data #9  data #10  data #11  data #12  data #13  data #14  data #15  data #16  data #17  data #18  data #19  data #20  data #21  data #22  data #23      mean
    0  0.305816  0.155288  0.070466  0.125154  0.218538  0.205412  0.212051  0.272537  0.271906  0.11161  0.155805  0.094729  0.082592   0.08464  0.192416  0.142971  0.092425   0.15179  0.122068  0.093538  0.092046  0.099876  0.064718  0.111082  0.147062
    
    
        data #0   data #1   data #2   data #3   data #4   data #5  data #6  data #7   data #8   data #9  data #10  data #11  data #12  data #13  data #14  data #15  data #16  data #17  data #18  data #19  data #20  data #21  data #22  data #23      mean
    0  0.287614  0.131127  0.076225  0.103622  0.158735  0.121515  0.23073  0.20032  0.267133  0.099163  0.136296  0.092848  0.087123  0.081563  0.180881  0.143041  0.085105  0.084382   0.12198  0.057344  0.123773  0.101326  0.082205   0.12175  0.132325
    
    
         data #0   data #1   data #2   data #3   data #4   data #5   data #6   data #7   data #8   data #9  data #10  data #11  data #12  data #13  data #14  data #15  data #16  data #17  data #18  data #19  data #20  data #21  data #22  data #23      mean
    0  0.255793  0.142457  0.157072  0.167493  0.225912  0.196098  0.217046  0.221807  0.182698  0.181625  0.144761   0.07812  0.073766  0.095228  0.204355  0.136366   0.09745  0.115519  0.143501  0.081802  0.120566  0.091509  0.082869   0.12918  0.147625
    
    
         data #0  data #1   data #2   data #3  data #4   data #5   data #6   data #7   data #8   data #9  data #10  data #11  data #12  data #13  data #14  data #15  data #16  data #17  data #18  data #19  data #20  data #21  data #22  data #23     mean
    0  0.222062    0.179  0.181947  0.223942  0.26015  0.203436  0.264365  0.213147  0.156813  0.247477    0.2677  0.264286  0.257081  0.193213  0.294884  0.304424  0.309157   0.34208  0.358637  0.282089  0.322395  0.359355   0.32846  0.352464  0.26619
    
    
         data #0   data #1  data #2   data #3   data #4   data #5   data #6   data #7   data #8   data #9  data #10  data #11  data #12  data #13  data #14  data #15  data #16  data #17  data #18  data #19  data #20  data #21  data #22  data #23      mean
    0  0.222641  0.180802  0.09917  0.268547  0.246523  0.238143  0.213462  0.254878  0.195012  0.210634  0.197698  0.143584  0.145936  0.126995  0.229186   0.22499  0.197595  0.374988  0.352877  0.270272  0.332304  0.334264  0.320893  0.330015  0.237975
    
    
         data #0   data #1   data #2   data #3   data #4   data #5   data #6   data #7   data #8  data #9  data #10  data #11  data #12  data #13  data #14  data #15  data #16  data #17  data #18  data #19  data #20  data #21  data #22  data #23      mean
    0  0.281397  0.291132  0.148901  0.295637  0.309759  0.286713  0.284124  0.292846  0.255892  0.25593  0.264756  0.260663  0.264794  0.208945  0.256013  0.353681  0.351435  0.356512  0.427831  0.341845  0.388048   0.38206  0.368101  0.390148  0.304882
    
    
         data #0   data #1   data #2  data #3   data #4   data #5   data #6   data #7   data #8   data #9  data #10  data #11  data #12  data #13  data #14  data #15  data #16  data #17  data #18  data #19  data #20  data #21  data #22  data #23      mean
    0  0.240396  0.249847  0.124114   0.3144  0.328333  0.282673  0.330821  0.294148  0.242655  0.322048  0.267742  0.248712  0.244695  0.197593   0.29137  0.301961  0.338718  0.320248  0.339631   0.28412  0.281345   0.31703  0.283036  0.273786  0.279976
    """
    WHICH = 'ours'
    df = from_loss_logs_to_df(ours_s if WHICH=='ours' else prssm_s, True,split_by = '\n    \n    \n' if WHICH=='ours' else '\n\n\n')
    print(df.T.to_latex())
