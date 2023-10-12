import torch


class Loss:
	@staticmethod
	def total_variation_l1(signals):
		diff_signal = torch.sum(torch.abs(signals[:, :, 1:] - signals[:, :, :-1]))
		return diff_signal

	@staticmethod
	def total_variation_mse(signals):
		diff_signal = torch.sum((signals[:, :, 1:] - signals[:, :, :-1]) ** 2)
		return diff_signal

	@staticmethod
	def get_loss_function(name):
		match name:
			case 'total_variation_l1':
				return Loss.total_variation_l1
			case 'total_variation_mse':
				return Loss.total_variation_mse
			case _:
				raise ValueError(f"Loss function '{name}' not recognized")


if __name__ == '__main__':
	t = torch.randint(0, 10, (2, 2, 3))
	t1 = t[:, :, 1:]
	t2 = t[:, :, :-1]
	z = 1
