import numpy as np
import pandas as pd
import torch
from scipy import special
from loguru import logger


class Encoder:
	def __init__(self, encoding_tasks: list[str] | None):
		self.encoding_tasks = encoding_tasks or []
		self.task_map = {
			method: getattr(self, method) for method in
			[
				func for func in dir(self) if
				callable(getattr(self, func))
				and not func.startswith("_")
				and func != 'run'
			]
		}
		# fallback column names
		self._force_cols = ['F1', 'F2', 'F3', 'F4']
		self._angle_cols = ['phi', 'theta', 'psi']

	def run(
			self,
			data: pd.DataFrame,
			**kwargs
	) -> pd.DataFrame:
		for task in self.encoding_tasks:
			func = self.task_map.get(task)
			if func:
				data = func(data, **kwargs)
			else:
				logger.warning(f"Could not find encoder function {func}, moving on.")
		return data

	def derivatives(
			self,
			df: pd.DataFrame,
			**kwargs
	) -> pd.DataFrame:
		cols = kwargs.get('encoding_cols', self._angle_cols)
		logger.info(f'Encoding derivative for cols = {self._angle_cols}')
		time = df.index.to_series().diff().dt.total_seconds().values[:, None]

		col_dot_names = [f"{col}_dot" for col in cols]
		df[col_dot_names] = df[cols].diff() / time

		col_ddot_names = [f"{col}_ddot" for col in cols]
		df[col_ddot_names] = df[col_dot_names].diff() / time
		return df

	def hermite(
			self,
			df: pd.DataFrame,
			**kwargs
	) -> pd.DataFrame:
		cols = kwargs.get('encoding_cols', self._force_cols)
		orders = kwargs.get('orders', [2, 3, 4])
		logger.info(f'Encoding hermite with cols = {cols} and orders = {orders}')
		for n in orders:
			col_names = [f"hermite_{n}({col})" for col in cols]
			poly_h = special.hermite(n)
			df[col_names] = poly_h(df)
		return df

	def angle(
			self,
			df: pd.DataFrame,
			**kwargs
	) -> pd.DataFrame:
		cols = kwargs.get('encoding_cols', self._angle_cols)
		logger.info(f'Encoding angle with cols = {cols}')
		sin_col_names = [f"sin({col})" for col in cols]
		cos_col_names = [f"cos({col})" for col in cols]
		arg = 2 * np.pi * df[cols] / df[cols].max()
		df[sin_col_names] = np.sin(arg)
		df[cos_col_names] = np.cos(arg)
		return df

	def torque(
			self,
			df: pd.DataFrame,
			**kwargs
	) -> pd.DataFrame:
		radius = kwargs.get('encode_torque_radius', 0.06)  # as measured using to_3d.evaluate_3d_distances
		logger.info(f'Encoding torque with radius = {radius}')
		f1, f2, f3, f4 = df['F1'], df['F2'], df['F3'], df['F4']
		df['torque_x'] = (radius / 2) * ((f1 + f2) - (f3 + f4))
		df['torque_y'] = (radius / 2) * ((f2 + f3) - (f1 + f4))
		return df


if __name__ == '__main__':

	angles = torch.load(r"G:\My Drive\Master\Lab\Thesis\Results\10_10_2023\kinematics.pt")
	forces = torch.load(r"G:\My Drive\Master\Lab\Thesis\Results\10_10_2023\forces.pt")
	df = pd.DataFrame(
		torch.concat([angles[0], forces[0]], dim=1),
		columns=['theta', 'phi', 'psi', 'F1', 'F2', 'F3', 'F4']
	)
	enc = Encoder(['angle', 'torque'])
	df1 = enc.run(df)
	z = 1
