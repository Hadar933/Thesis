import pandas as pd
from scipy.stats.qmc import LatinHypercube
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use('TkAgg')
fig = plt.figure()


def from_01_to_ab(x, a, b):
	"""
	assumes the data in x is in [0,1], and casts it to [a,b]
	:param x: input data
	:param a: min range
	:param b: max range
	:return: x with fixed range
	"""
	return a + x * (b - a)


def latin_hyper_cube_sampling(show_plot: bool, save_to_csv: bool):
	engine = LatinHypercube(d=3, strength=1, seed=42, scramble=True, optimization='lloyd')
	lhc_data = engine.random(n=200)
	lhc_data[:, 0] = from_01_to_ab(lhc_data[:, 0], Aa, Ab)
	lhc_data[:, 1] = from_01_to_ab(lhc_data[:, 1], ka, kb)
	lhc_data[:, 2] = from_01_to_ab(lhc_data[:, 2], fa, fb)
	if show_plot:
		lhc_ax = fig.add_subplot(111, projection='3d')
		lhc_ax.scatter(lhc_data[:, 0], lhc_data[:, 1], lhc_data[:, 2])
		lhc_ax.set_title(f'Latin Hyper Cube Sampler ({len(lhc_data)} samples)')
		lhc_ax.set_xlabel(r'$A$')
		lhc_ax.set_ylabel(r'$K$')
		lhc_ax.set_zlabel(r'$f$')
		plt.show()
	if save_to_csv:
		df = pd.DataFrame(lhc_data, columns=['A', 'K', 'f']).round(3)
		df.to_csv('exp_to_run.csv')
	return lhc_data


if __name__ == '__main__':
	fa, fb = 5.0, 20.0
	ka, kb = 0.01, 1.0
	Aa, Ab = 3.5, 6
	samples = latin_hyper_cube_sampling(True, True)
