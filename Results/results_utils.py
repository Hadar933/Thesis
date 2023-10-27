import matplotlib
import dash
from dash import dcc, html, Input, Output
import torch
import pandas as pd
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np

from Camera import camera_utils
from Forces import parse_forces, force_utils


def results_plotter(kinematics_path: str, forces_path: str):
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


def playground_plotter_np(exp_name='[F=5.037_A=M_PIdiv4.86_K=0.188]'):
	matplotlib.use('TkAgg')
	date = '23_10_2023'
	traject_path = fr"E:\Hadar\experiments\{date}\results\{exp_name}\trajectories.npy"
	angles_path = fr"E:\Hadar\experiments\{date}\results\{exp_name}\angles.npy"
	force_path = fr"E:\Hadar\experiments\{date}\forces\Forces{exp_name}.csv"

	forces_df, _ = parse_forces.read_forces_csv(csv_filename=force_path, header_row_count=21, tare=[])  # in newtons
	f1, f2, f3, f4 = forces_df['F1'], forces_df['F2'], forces_df['F3'], forces_df['F4']
	forces_df['sum'] = f1 + f2 + f3 + f4
	radius = 6  # millimeters
	forces_df['tau_x=f1+f2-f3-f4'] = (radius / 2) * ((f1 + f2) - (f3 + f4))
	forces_df['tau_y=f2+f3-f1-f4'] = (radius / 2) * ((f2 + f3) - (f1 + f4))

	angles_df = pd.DataFrame(np.load(angles_path).T, columns=['theta', 'phi', 'psi'])

	center_of_mass = camera_utils.calc_wing_center_of_mass(np.load(traject_path), do_plot=True)
	inertial_force = force_utils.calc_inertial_force(center_of_mass)
	forces_df['inertia'] = inertial_force

	df = pd.concat([forces_df, angles_df], axis=1)

	fig = go.Figure()
	for col in df.columns:
		fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=col))
	fig.show()
	fig.write_html(f'plot{exp_name}.html')


def playground_plotter_tensors(i):
	matplotlib.use('TkAgg')
	date = '19_10_2023'
	kpath = rf'{date}/kinematics_cleaned.pt'
	fpath = rf'{date}/forces_cleaned.pt'
	k = pd.DataFrame(torch.load(kpath)[i], columns=['theta', 'phi', 'psi'])
	f = pd.DataFrame(torch.load(fpath)[i], columns=['f1', 'f2', 'f3', 'f4'])
	f1, f2, f3, f4 = f['f1'], f['f2'], f['f3'], f['f4']
	f['sum'] = f1 + f2 + f3 + f4
	f['tau_x=f1+f2-f3-f4'] = (f1 + f2) - (f3 + f4)
	f['tau_y=f2+f3-f1-f4'] = (f2 + f3) - (f1 + f4)
	df = pd.concat([f, k], axis=1)

	fig = go.Figure()
	for col in df.columns:
		fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=col))
	fig.show()
	fig.write_html(f'plot{i}.html')


def remove_and_trim_datasets(kinematics_path, forces_path, save=True, basic_plot=True, keep_as_list=True):
	"""
	a basic function that plots the data and allows to slice it and/or remove it altogether.
	:param kinematics_path: path to tensor or list of tensors kinematics
	:param forces_path: path to tensor or list of tensors forces
	:param save: if true, saves the results
	:param basic_plot: if true, plot a jpg, otherwise plots an interactive plot on html
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
			fig = go.Figure()
			for col in df.columns:
				fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=col))
			fig.show()
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
	# playground_plotter(10)
	playground_plotter_np()
	exp_date = '19_10_2023'
	fpath = fr'G:\My Drive\Master\Lab\Thesis\Results\{exp_date}\forces_list.pt'
	kpath = fr'G:\My Drive\Master\Lab\Thesis\Results\{exp_date}\kinematics_list.pt'
	remove_and_trim_datasets(kpath, fpath)
