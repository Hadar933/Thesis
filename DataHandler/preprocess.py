import numpy as np
import pandas as pd
from loguru import logger


class Preprocess:
	def __init__(self, preprocessing_tasks: list[str] | None):
		self.preprocessing_tasks = preprocessing_tasks or []
		self.task_map = {
			method: getattr(self, method) for method in
			[
				func for func in dir(self) if
				callable(getattr(self, func))
				and not func.startswith("_")
				and func != 'run'
			]
		}

	def run(
			self,
			df: pd.DataFrame,
			**kwargs
	) -> pd.DataFrame:
		for task in self.preprocessing_tasks:
			func = self.task_map.get(task)
			if func:
				df = func(df, **kwargs)
			else:
				logger.warning(f"Could not find preprocess function {func}, moving on.")
		return df

	@staticmethod
	def resample(
			df: pd.DataFrame,
			**kwargs
	) -> pd.DataFrame:
		"""
		resamples the data with frequency 5000hz = 1/5000sec = 1/200microsec
		:param df:
		:return:
		"""
		resample_freq = kwargs.get('resample_freq', '200us')
		logger.info(f'Resampling with freq = {resample_freq}')
		df = df.resample(resample_freq).mean()
		return df

	@staticmethod
	def interpolate(
			df: pd.DataFrame,
			**kwargs
	) -> pd.DataFrame:
		limit = kwargs.get('interpolate_limit', 20)
		method = kwargs.get('interpolate_method', 'linear')
		order = kwargs.get('interpolate_order', None)
		logger.info(f"Interpolating with limit = {limit}, method = {method} and order = {order}")
		df = df.interpolate(method=method, order=order, limit=limit)
		return df

	@staticmethod
	def drop(
			df: pd.DataFrame,
			**kwargs
	) -> pd.DataFrame:
		cols_to_drop = kwargs.get('drop_cols', ['theta'])
		logger.info(f'Dropping columns = {cols_to_drop}')
		df = df.drop(columns=[cols_to_drop])
		return df


if __name__ == '__main__':
	_df = pd.DataFrame([
		(0.0, np.nan, -1.0, 1.0),
		(np.nan, 2.0, np.nan, np.nan),
		(2.0, 3.0, np.nan, 9.0),
		(np.nan, 4.0, -4.0, 16.0)
	], columns=list('abcd'))
