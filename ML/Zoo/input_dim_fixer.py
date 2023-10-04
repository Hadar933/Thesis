from typing import List

import torch
import torch.nn as nn


class InputDimFixer(nn.Module):
	def __init__(
			self,
			new_input_dim: int,
			trained_model_input_dim: int,
			trained_model: nn.Module,
			freeze: bool
	):
		"""
		adds a linear layer before a given model to match the dimension of data with different number of features
		:param new_input_dim: number of features in the new data
		:param trained_model_input_dim: number of features used in the original dataset
		:param trained_model: the model object
		:param freeze: if true, freezes the parameters of the trained model
		"""
		super(InputDimFixer, self).__init__()
		self.dim_fixer = nn.Linear(new_input_dim, trained_model_input_dim)
		self.trained_model = trained_model
		self.freeze = freeze
		for param in self.trained_model.parameters():
			param.requires_grad = not freeze

	def forward(
			self,
			x: torch.Tensor
	) -> torch.Tensor:
		x = self.dim_fixer(x)
		x = self.trained_model(x)
		return x
