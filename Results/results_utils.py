import matplotlib
import dash
from dash import dcc, html, Input, Output
import torch
import pandas as pd
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from tqdm import tqdm


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


def playground_plotter(i):
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


def remove_bad_datasets_from_tensor(kinematics_path, forces_path, save=True, basic_plot=True):
	forces_to_keep = []
	kinematics_to_keep = []
	forces = torch.load(forces_path)
	kinematics = torch.load(kinematics_path)
	for dataset_idx in tqdm(range(len(forces))):
		curr = torch.cat((forces[dataset_idx], kinematics[dataset_idx]), dim=1)
		df = pd.DataFrame(curr, columns='f1 f2 f3 f4 theta phi psi'.split())
		if basic_plot:
			df.plot()
			plt.show()
		else:
			fig = go.Figure()
			for col in df.columns:
				fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=col))
			fig.show()
		if input('Is this one bad? [y/Any]') == 'y':
			continue
		forces_to_keep.append(forces[dataset_idx])
		kinematics_to_keep.append(kinematics[dataset_idx])
	forces_to_keep = torch.stack(forces_to_keep)
	kinematics_to_keep = torch.stack(kinematics_to_keep)
	if save:
		torch.save(forces_to_keep, forces_path.replace('.pt', '_cleaned.pt'))
		torch.save(kinematics_to_keep, kinematics_path.replace('.pt', '_cleaned.pt'))


if __name__ == '__main__':
	# playground_plotter(10)
	exp_date = '23_10_2023'
	fpath = fr'G:\My Drive\Master\Lab\Thesis\Results\{exp_date}\forces.pt'
	kpath = fr'G:\My Drive\Master\Lab\Thesis\Results\{exp_date}\kinematics.pt'
	remove_bad_datasets_from_tensor(kpath, fpath, save=True)
# results_plotter(kpath, fpath)
