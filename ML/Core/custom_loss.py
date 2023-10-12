import torch
import torch.nn as nn


class PairwiseVariationLossL1(nn.Module):
	def __init__(self):
		super(PairwiseVariationLossL1, self).__init__()
		self.last_prediction = None

	def forward(self, current_prediction):
		if self.last_prediction is None:
			self.last_prediction = current_prediction
			return 0

		# Check for size mismatch and adjust if necessary
		if self.last_prediction.size(0) != current_prediction.size(0):
			self.last_prediction = self.last_prediction[:current_prediction.size(0)]

		variation = torch.mean(torch.abs(current_prediction - self.last_prediction))

		self.last_prediction = current_prediction

		return variation


class PairwiseVariationLossMSE(nn.Module):
	def __init__(self):
		super(PairwiseVariationLossMSE, self).__init__()
		self.last_prediction = None

	def forward(self, current_prediction):
		if self.last_prediction is None:
			self.last_prediction = current_prediction
			return 0

		# Check for size mismatch and adjust if necessary
		if self.last_prediction.size(0) != current_prediction.size(0):
			self.last_prediction = self.last_prediction[:current_prediction.size(0)]

		variation = torch.mean((current_prediction - self.last_prediction) ** 2)

		self.last_prediction = current_prediction

		return variation


class LossFactory:
	@staticmethod
	def get_loss(loss_name):
		match loss_name.lower():
			case 'pairwise_variation_mae' | 'pairwise_variation_l1' | 'pv_l1' | 'pv_mae':
				return PairwiseVariationLossL1()
			case 'pairwise_variation_mse' | 'pairwise_variation_l2' | 'pv_l2' | 'pv_mse':
				return PairwiseVariationLossMSE()
			case _:  # tries loading default name from torch
				return getattr(torch.nn, loss_name)()
