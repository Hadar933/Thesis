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
		self._points_cols = ['p0', 'p1', 'p2']

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

	def center_of_mass(
			self,
			df: pd.DataFrame,
			**kwargs
	) -> pd.DataFrame:
		cols = kwargs.get('center_of_mass_cols', self._points_cols)
		logger.info(f'Encoding center of mass using cols = {cols}')
		p0, p1, p2 = (df[col] for col in cols)
		vec1 = p2 - p0  # AB
		vec2 = p2 - p1  # AC
		lambda1: float = -0.173
		lambda2: float = 0.857
		center_of_mass = lambda1 * vec1 + lambda2 * vec2  # TODO: not working..
		center_of_mass = (p0 + p1 + p2) / 3  # close enough to COM
		df['center_of_mass'] = center_of_mass
		return df

	def inertial_force(
			self,
			df: pd.DataFrame,
			**kwargs
	) -> pd.DataFrame:
		dt = kwargs.get('inertial_force_dt', 1e-4)  # sec
		wing_mass = kwargs.get('inertial_force_wing_mass', 0.003)  # kg (=3g)
		if 'center_of_mass' not in df.columns:
			df = self.center_of_mass(df, **kwargs)
		logger.info(f'Encoding inertial force using dt={dt}, wing_mass={wing_mass}')
		center_of_mass_z = np.vstack(df['center_of_mass'])[:, 2] / 1000  # meters
		center_of_mass_z_dot = np.gradient(center_of_mass_z, dt)
		center_of_mass_z_ddot = np.gradient(center_of_mass_z_dot, dt)
		force = - wing_mass * center_of_mass_z_ddot
		df['F_inertia'] = force
		return df

	def derivatives(
			self,
			df: pd.DataFrame,
			**kwargs
	) -> pd.DataFrame:
		cols = kwargs.get('deriv_cols', self._angle_cols)
		logger.info(f'Encoding derivative for cols = {cols}')
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
		cols = kwargs.get('hermite_cols', self._force_cols)
		orders = kwargs.get('orders', [2, 3, 4])
		logger.info(f'Encoding hermite with cols = {cols} and orders = {orders}')
		for n in orders:
			col_names = [f"hermite_{n}({col})" for col in cols]
			poly_h = special.hermite(n)
			df[col_names] = poly_h(df)
		return df

	def sin_cos(
			self,
			df: pd.DataFrame,
			**kwargs
	) -> pd.DataFrame:
		cols = kwargs.get('angle_cols', self._angle_cols)
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
		radius = kwargs.get('encode_torque_radius', 0.06)  # in meters (=6cm)
		logger.info(f'Encoding torque with radius = {radius}')
		f1, f2, f3, f4 = df['F1'], df['F2'], df['F3'], df['F4']
		df['torque_x'] = (radius / 2) * ((f1 + f2) - (f3 + f4))
		df['torque_y'] = (radius / 2) * ((f2 + f3) - (f1 + f4))
		return df

	def sum_cols(
			self,
			df: pd.DataFrame,
			**kwargs
	) -> pd.DataFrame:
		cols = kwargs.get('sum_cols', self._force_cols)
		logger.info(f'Encoding Sum of {cols}')
		df['+'.join(cols)] = df[cols].sum(axis=1)
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
