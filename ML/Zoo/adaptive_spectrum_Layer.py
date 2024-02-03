import numpy as np
import torch
import torch.nn as nn


class SimpleGate(nn.Module):
	def __init__(
			self,
			input_size: int,
			n_frequencies: int,
			dropout_rate: float
	):
		super(SimpleGate, self).__init__()
		self.input_size = input_size
		self.linear1 = nn.Linear(input_size, n_frequencies)
		self.dropout = nn.Dropout(p=dropout_rate)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		a gated linear unit with learnable weights.
		:param x: a tensor with shape (B,H,input_size)
		:return: a positive value tensor with shape (B,H,history_size)
		"""
		x = self.dropout(x)
		x = self.linear1(x)
		x = x * self.sigmoid(x)
		return x


class Complexifier(nn.Module):
	def __init__(
			self,
			input_dim: int
	):
		super(Complexifier, self).__init__()
		self.input_dim = input_dim
		self.magnitude_fc = nn.Linear(self.input_dim, 1)
		self.phase_fc = nn.Linear(self.input_dim, 1)
		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		takes a tensor with shape (B,1,F,input_dim) where input_dim is usually the hidden dim of the ASL and returns a
		complex tensor with the same shape
		:param x:
		:return:
		"""
		x = self.relu(x)
		magnitude = self.relu(self.magnitude_fc(x))
		phase = 2 * torch.pi * self.sigmoid(self.phase_fc(x))
		return torch.complex(real=magnitude * torch.cos(phase), imag=magnitude * torch.sin(phase))


class AdaptiveSpectrumLayer(nn.Module):
	def __init__(
			self,
			history_size: int,
			hidden_dim: int,
			sampling_rate: float,
			dropout_rate: float,
			complexify: bool,
			gate: bool,
			use_freqs: bool,
			multidim_fft: bool,
			frequency_threshold: float | None = None
	):
		super(AdaptiveSpectrumLayer, self).__init__()
		self.history_size = history_size
		self.hidden_dim = hidden_dim
		self.sampling_rate = sampling_rate
		self.dropout_rate = dropout_rate
		self.complexify = complexify
		self.gate = gate
		self.fftfreq = None
		self.use_freqs = use_freqs
		self.multidim_fft = multidim_fft
		self.frequency_threshold = frequency_threshold if frequency_threshold else self.sampling_rate / 2  # Nyquist
		if self.use_freqs:
			if self.multidim_fft:
				self.n_fourier_features = 7  # 2 * 3 + 1
			else:
				self.n_fourier_features = 4  # 3 + 1
		else:
			if self.multidim_fft:
				self.n_fourier_features = 6  # 2 * 3
			else:
				self.n_fourier_features = 3  # 3
		self.time_axis = 1
		self.n_frequencies = self._frequencies_to_use()

		self.softmax = nn.Softmax(dim=-1)
		self.sigmoid = nn.Sigmoid()
		if self.gate:
			self.gate_of_flattened_ffts = SimpleGate(
				input_size=self.n_frequencies * self.hidden_dim,
				n_frequencies=self.n_frequencies,
				dropout_rate=self.dropout_rate
			)
		else:
			self.gate_of_flattened_ffts = nn.Sequential(
				nn.Dropout(p=self.dropout_rate),
				nn.Linear(self.n_frequencies * self.hidden_dim, self.n_frequencies)
			)
		self.frequency_projection_layers = nn.ModuleList([
			# for each frequency, we project features from fourier (like magnitude, angle,...)
			nn.Linear(self.n_fourier_features, self.hidden_dim) for _ in range(self.n_frequencies)
		])
		if self.complexify:
			self.complexifier_layers = nn.ModuleList([
				# convert a hidden representation to a complex number
				Complexifier(self.hidden_dim) for _ in range(self.n_frequencies)
			])

	def _frequencies_to_use(self) -> int:
		"""
		we trim the fftfreq values given the frequency threshold and return the number of frequencies to consider,
		:return: number of frequencies to consider, that represent frequencies from 0 to frequency_threshold
		"""
		frequencies = torch.fft.rfftfreq(self.history_size, 1 / self.sampling_rate)
		frequencies_mask = frequencies <= self.frequency_threshold
		self.fftfreq = frequencies[frequencies_mask]
		n_frequencies = frequencies_mask.sum().item()
		return n_frequencies

	def forward(self, x: torch.Tensor, time_features: torch.Tensor | None = None) -> torch.Tensor:
		"""
		applies rfft on the input and learns weights for the frequencies. then applies irfft on the weighted fft.
		:param x: a tensor with shape (B,H,F)
		:param time_features: a representation of x itself, with shape (B,H,F)
		:return: a tensor with shape (B,H,F)
		"""
		fft = torch.fft.rfft(x, dim=self.time_axis, norm='ortho')[:, :self.n_frequencies, :]  # (B,n_freqs,F)
		magnitude, angle = torch.abs(fft), torch.angle(fft)  # (B,n_freqs,F), (B,n_freqs,F)
		fourier_features = torch.stack([magnitude, torch.sin(angle), torch.cos(angle)], dim=-1)  # (B,n_freqs,F,3)
		if self.multidim_fft:
			fftn = torch.fft.rfftn(x, dim=self.time_axis)[:, :self.n_frequencies, :]  # (B,n_freqs,F)
			magnitude_n, angle_n = torch.abs(fftn), torch.angle(fftn)  # (B,n_freqs,F), (B,n_freqs,F)
			fourier_features_n = torch.stack( # (B,n_freqs,F,3)
				[magnitude_n, torch.sin(angle_n), torch.cos(angle_n)],
				dim=-1
			)
			fourier_features = torch.cat([fourier_features, fourier_features_n], dim=-1)  # (B,n_freqs,F,6)
		if self.use_freqs:
			fourier_features = torch.cat([
				fourier_features,
				self.fftfreq.unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(fft.shape[0], 1, fft.shape[-1], 1).to(
					fft.device)
			], dim=-1)  # (B,n_freqs,F,7) or (B,n_freqs,F,4) based on multidim_fft
		projected_fourier_features = [
			self.frequency_projection_layers[i](fourier_features[:, [i], :, :])
			for i in range(self.n_frequencies)
		]  # [(B,1,F,hidden_dim)_{1}, ...,(B,1,F,hidden_dim)_{n_freqs}]
		flattened_fourier_features = torch.cat(projected_fourier_features, dim=-1)  # (B,1,F,hidden_dim*n_frequencies)

		weights = self.sigmoid(self.gate_of_flattened_ffts(flattened_fourier_features))  # (B,1,F,n_frequencies)
		if self.complexify:
			new_fft = torch.cat([
				self.complexifier_layers[i](projected_fourier_features[i]) for i in range(self.n_frequencies)
			], dim=1).squeeze(-1)
			# weighted_Fft is a convex sum of fft and new_Fft:
			w = weights.squeeze(1).permute(0, 2, 1)
			weighted_fft = w * new_fft + (1 - w) * fft  # (B,n_frequencies,F)
		else:
			weighted_fft = fft * weights.squeeze(1).permute(0, 2, 1)  # (B,n_frequencies,F)
		weighted_fft = torch.nn.functional.pad(
			input=weighted_fft,
			pad=(0, 0, 0, (self.history_size // 2 + 1) - self.n_frequencies, 0, 0)
		)

		reconstructed_x = torch.fft.irfft(weighted_fft, dim=self.time_axis, norm='ortho')
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
