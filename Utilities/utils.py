import os.path

import dash
import torch
import imageio
import scipy
import json

import plotly.offline as pyo
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import html, dcc, Output, Input

from matplotlib import pyplot as plt
from plotly import graph_objects as go
from plotly_resampler import FigureResampler
from tqdm import tqdm
from typing import Literal

from ML.Core.datasets import FixedLenMultiTimeSeries, VariableLenMultiTimeSeries
from ML.Zoo.seq2seq import Seq2Seq

TIME_FORMAT = '%Y-%m-%d_%H-%M-%S'


# ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║                                               PLOTTING                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
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
			options=[{'label': f'Dataset {i}', 'value': i} for i in range(kinematics.size(0))],
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
		phi_trace = go.Scatter(y=kinematics[selected_dataset, :, 0].numpy(), mode='lines', name='phi')
		psi_trace = go.Scatter(y=kinematics[selected_dataset, :, 1].numpy(), mode='lines', name='psi')
		force_traces = [go.Scatter(y=forces[selected_dataset, :, i].numpy(), mode='lines', name=f'F{i + 1}')
						for i in range(4)]
		traces = [phi_trace, psi_trace] + force_traces
		layout = go.Layout(
			title=f'Kinematics and Forces for Experiment {selected_dataset}',
			xaxis=dict(title='Sample'),
			yaxis=dict(title='Value')
		)
		figure = {'data': traces, 'layout': layout}
		return figure

	app.run_server(debug=True)


def plot_df_with_plotly(
		df: pd.DataFrame,
		ignore_cols: list[str] | None = None,
		title: str = "Data vs Time",
		x_title: str = "time / steps",
		y_title: str = "Data",
		save_path: str | None = None
) -> None:
	"""plots a df with plotly resampler"""
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
		model: Seq2Seq,
		test_loader
) -> list[np.ndarray]:
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


def plot_with_background(
		x_values: torch.Tensor,
		y_values: torch.Tensor,
		w_values: torch.Tensor,
		x_label: str,
		y_label: str,
		title: str
) -> np.ndarray:
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

def update_json(yaml_path: str, new_data):
	""" updates a given yaml file with new data dictionary """
	if os.path.exists(yaml_path):
		with open(yaml_path, "r", encoding='utf-8') as f:
			old_data = json.load(f)
	else:
		old_data = {}
	with open(yaml_path, "w", encoding='utf-8') as f:
		json.dump({**old_data, **new_data}, f, ensure_ascii=False, indent=4)


def format_df_torch_entries(
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


if __name__ == '__main__':
	exp_date = '19_10_2023'
	exp_name = '[F=7.886_A=M_PIdiv5.401_K=0.03]'
	filename = 'merged_data_preprocessed_and_encoded.pkl'
	plot_df_with_plotly(
		df=pd.read_pickle(fr"E:\Hadar\experiments\{exp_date}\results\{exp_name}\{filename}"),
		ignore_cols=['p0', 'p1', 'p2', 'center_of_mass']
	)
