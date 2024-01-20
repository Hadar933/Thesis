import numpy as np
import torch
import torch.nn as nn


class SimpleGate(nn.Module):
	def __init__(
			self,
			input_size: int,
			n_frequencies: int
	):
		super(SimpleGate, self).__init__()
		self.input_size = input_size
		self.linear1 = nn.Linear(input_size, n_frequencies)
		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()
		self.weights = nn.Parameter(torch.ones(n_frequencies))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		a gated linear unit with learnable weights.
		:param x: a tensor with shape (B,H,input_size)
		:return: a positive value tensor with shape (B,H,history_size)
		"""
		x = self.linear1(x)
		x = x * self.sigmoid(self.weights * x)
		return x


class AdaptiveSpectrumLayer(nn.Module):
	def __init__(
			self,
			history_size: int,
			hidden_dim: int,
			sampling_rate: float,
			frequency_threshold: float | None = None,
			smoothing_kernel_size: int = 0
	):
		super(AdaptiveSpectrumLayer, self).__init__()
		self.history_size = history_size
		self.hidden_dim = hidden_dim
		self.n_fourier_features = 2  # magnitude, angle
		self.frequency_threshold = frequency_threshold if frequency_threshold else sampling_rate / 2  # Nyquist
		self.time_axis = 1
		self.sampling_rate = sampling_rate
		self.smoothing_kernel_size = smoothing_kernel_size
		self.n_frequencies = self.frequencies_to_use()

		self.softmax = nn.Softmax(dim=-1)
		self.gate_of_flattened_ffts = SimpleGate(
			input_size=self.n_frequencies * self.hidden_dim,
			n_frequencies=self.n_frequencies
		)
		self.frequency_projection_layers = nn.ModuleList([
			# for each frequency, we project features from fourier (like magnitude, angle,...)
			nn.Linear(self.n_fourier_features, hidden_dim) for _ in range(self.n_frequencies)
		])

	def frequencies_to_use(self) -> int:
		"""
		we trim the fftfreq values given the frequency threshold and return the number of frequencies to consider,
		:return: number of frequencies to consider, that represent frequencies from 0 to frequency_threshold
		"""
		frequencies = torch.fft.rfftfreq(self.history_size, 1 / self.sampling_rate)
		frequencies_mask = frequencies <= self.frequency_threshold
		n_frequencies = frequencies_mask.sum().item()
		return n_frequencies

	def __str__(self):
		raise NotImplementedError

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""

		:param x: a tensor with shape (B,H,F)
		:TODO: think of frequencies encoding, like positional encoding. maybe use the frequency as a context vector?
				or encode the freq with transformer-like positional encoding?
		:TODO: maybe encode the angle value using sine and cosine?
		:return:
		"""
		fft = torch.fft.rfft(x, dim=self.time_axis)[:, :self.n_frequencies, :]  # (B,self.n_frequencies,F)
		magnitude, angle = torch.abs(fft), torch.angle(fft)  # (B,H,F), (B,H,F)
		fourier_features = torch.stack([magnitude, angle], dim=-1)  # (B,H,F,2)
		projected_fourier_features = [
			self.frequency_projection_layers[i](fourier_features[:, [i], :, :])
			for i in range(self.n_frequencies)
		]  # [(B,1,F,hidden_dim)_{1}, ...,(B,1,F,hidden_dim)_{n_frequencies}]
		flattened_fourier_features = torch.cat(projected_fourier_features, dim=-1)  # (B,1,F,hidden_dim*n_frequencies)
		weights = self.softmax(self.gate_of_flattened_ffts(flattened_fourier_features))  # (B,F,n_frequencies)
		weighted_fft = fft * weights.squeeze().permute(0, 2, 1)  # (B,n_frequencies,F)
		weighted_fft = torch.nn.functional.pad(weighted_fft, (0, 0, 0, (self.history_size//2+1) - self.n_frequencies, 0, 0))
		reconstructed_x = torch.fft.irfft(weighted_fft, dim=self.time_axis)
		return reconstructed_x


if __name__ == '__main__':
	T = 0.1024
	fs = 5000
	t = torch.arange(0, T, 1 / fs)
	x = torch.stack(
		[
			torch.sin(2 * np.pi * 50 * t),
			torch.sin(2 * np.pi * 120 * t),
			torch.sin(2 * np.pi * 30 * t) + torch.sin(2 * np.pi * 9 * t),
		],
		dim=1
	).repeat(4, 1, 1)
	asl = AdaptiveSpectrumLayer(
		history_size=x.shape[1],
		hidden_dim=10,
		sampling_rate=fs,
		frequency_threshold=200
	)
	y = asl(x)
	print(y.shape)
