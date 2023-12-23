import torch
import torch.nn as nn
from typing import Optional

"""
Adaptation from LTSFLinear git repo
"""


class Linear(nn.Module):
	"""
	Just one Linear layer
	"""

	def __init__(
			self,
			seq_len: int,
			pred_len: int,
			channels: int,
			individual: bool,
			output_size: Optional[int] = None
	):
		super(Linear, self).__init__()
		self.seq_len = seq_len
		self.pred_len = pred_len
		self.channels = channels
		self.individual = individual
		self.use_matching_layer = output_size is not None and output_size != channels
		if self.use_matching_layer:
			self.output_dim_matching_layer = nn.Linear(channels, output_size)
		if self.individual:
			self.Linear = nn.ModuleList()
			for i in range(self.channels):
				self.Linear.append(nn.Linear(self.seq_len, self.pred_len))
		else:
			self.Linear = nn.Linear(self.seq_len, self.pred_len)

	def __str__(self):
		return f"LTSFLinear[{self.seq_len},{self.pred_len},{self.channels},{self.individual}]"

	def forward(self, x):
		# x: [Batch, Input length, Channel]
		if self.individual:
			output = torch.zeros([x.size(0), self.pred_len, x.size(2)], dtype=x.dtype).to(x.device)
			for i in range(self.channels):
				output[:, :, i] = self.Linear[i](x[:, :, i])
			x = output
		else:
			x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
		if self.use_matching_layer:
			x = self.output_dim_matching_layer(x)
		return x  # [Batch, Output length, output_size]
