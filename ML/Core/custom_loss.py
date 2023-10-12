import torch
import torch.nn as nn


class TotalVariationLossL1(nn.Module):
	def __init__(self):
		super(TotalVariationLossL1, self).__init__()

	def forward(self, signals):
		diff_signal = torch.sum(torch.abs(signals[:, :, 1:] - signals[:, :, :-1]))
		return diff_signal


class TotalVariationLossMSE(nn.Module):
	def __init__(self):
		super(TotalVariationLossMSE, self).__init__()

	def forward(self, signals):
		diff_signal = torch.sum((signals[:, :, 1:] - signals[:, :, :-1]) ** 2)
		return diff_signal


class LossFactory:
	@staticmethod
	def get_loss(loss_name):
		match loss_name.lower():
			case 'total_variation_mae' | 'total_variation_l1' | 'tv_l1' | 'tv_mae':
				return TotalVariationLossL1()
			case 'total_variation_mse' | 'total_variation_l2' | 'tv_l2' | 'tv_mse':
				return TotalVariationLossMSE()
			case _:  # tries loading default name from torch
				return getattr(torch.nn, loss_name)()
