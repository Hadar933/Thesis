import torch


class Loss:
	@staticmethod
	def total_variation_l1(signals):
		return torch.sum(torch.abs(signals[:, :, 1:] - signals[:, :, :-1]))

	@staticmethod
	def total_variation_mse(signals):
		return torch.sum((signals[:, :, 1:] - signals[:, :, :-1]) ** 2)

	@staticmethod
	def get_loss_function(loss_name):
		match loss_name.lower():
			case 'total_variation_mae' | 'total_variation_l1' | 'tv_l1' | 'tv_mae':
				return Loss.total_variation_l1
			case 'total_variation_mse' | 'total_variation_l2' | 'tv_l2' | 'tv_mse':
				return Loss.total_variation_mse
			case _:  # tries loading default name from torch
				getattr(torch.nn, loss_name)()
