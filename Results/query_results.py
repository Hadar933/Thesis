import os
import re
import matplotlib
import numpy as np
from Camera import camera_utils
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import torch

matplotlib.use('TkAgg')


def results_plotter(kinematics_path, forces_path):
	""" an interactive dash plot for the kinematics and forces torch tensors """
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
		force_traces = [go.Scatter(y=forces[selected_dataset, :, i].numpy(), mode='lines', name=f'F{i + 1}') for i in
						range(4)]
		traces = [phi_trace, psi_trace] + force_traces
		layout = go.Layout(title=f'Kinematics and Forces for Experiment {selected_dataset}', xaxis=dict(title='Sample'),
						   yaxis=dict(title='Value'))
		figure = {'data': traces, 'layout': layout}
		return figure

	app.run_server(debug=True)


def filter_results(exp_date: str, f=None, a=None, k=None):
	""" displays all kinematics datasets from the hard drive that adhere the provided arguments """
	parent_dir = rf'E:\Hadar\experiments\{exp_date}\results'
	# Create the search patterns based on provided values
	f_pattern = f"F={f}" if f is not None else ""
	a_pattern = f"A=M_PIdiv{a}" if a is not None else ""
	k_pattern = f"K={k}" if k is not None else ""
	relevant_result_dirs = []
	for subdir in os.listdir(parent_dir):
		subdir = os.path.join(parent_dir, subdir)
		if f_pattern in subdir and a_pattern in subdir and k_pattern in subdir:
			relevant_result_dirs.append(subdir)
	return relevant_result_dirs


def plot_filtered_results(paths, what_to_plot):

	for item in paths:
		if 'angles' in what_to_plot:
			data = np.load(os.path.join(item, f"angles.npy"))
			title_addition = re.search(r'\[(.*)\]', item).group(1)
			camera_utils.plot_angles(data, n_samples=10_000, add_to_title=title_addition)
		if 'trajectories' in what_to_plot:
			data = np.load(os.path.join(item, f"trajectories.npy"))
			title_addition = re.search(r'\[(.*)\]', item).group(1)
			camera_utils.plot_trajectories(data, wing_plane_jmp=200, add_to_title=title_addition)


if __name__ == '__main__':
	results_plotter(r'10_10_2023/kinematics.pt', r'10_10_2023/forces.pt')
