import os
import re
import matplotlib
import numpy as np
from Camera import camera_utils
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import torch
import pandas as pd

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


if __name__ == '__main__':
	date = '17_10_2023'
	kpath = rf'{date}/kinematics.pt'
	fpath = rf'{date}/forces.pt'
	k = pd.DataFrame(torch.load(kpath)[0], columns=['theta', 'phi', 'psi'])
	f = pd.DataFrame(torch.load(fpath)[0], columns=['f1', 'f2', 'f3', 'f4'])
	f1, f2, f3, f4 = f['f1'], f['f2'], f['f3'], f['f4']
	f['sum'] = f1 + f2 + f3 + f4
	f['tau_x=f1+f2-f3-f4'] = (f1 + f2) - (f3 + f4)
	f['tau_y=f2+f3-f1-f4'] = (f2 + f3) - (f1 + f4)
	df = pd.concat([f, k], axis=1)
	# df.plot()
	# plt.show()

	import plotly.graph_objects as go

	fig = go.Figure()
	for col in df.columns:
		fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=col))
	fig.show()
	# Optionally, save the figure to an HTML file
	fig.write_html('plot.html')

	z = 2
# results_plotter(r'17_10_2023/kinematics.pt', r'17_10_2023/forces.pt')
