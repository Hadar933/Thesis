import torch
import torchinfo
from torch import nn


class GLU(nn.Module):
	def __init__(
			self,
			input_dim: int
	) -> None:
		super(GLU, self).__init__()
		self.input_dim = input_dim
		self.fc1 = nn.Linear(input_dim, input_dim)
		self.fc2 = nn.Linear(input_dim, input_dim)
		self.sigmoid = nn.Sigmoid()

	def forward(
			self,
			x: torch.Tensor
	) -> torch.Tensor:
		return self.fc2(x) * self.sigmoid(self.fc1(x))


class GRN(nn.Module):
	def __init__(
			self, input_dim: int, hidden_dim: int, output_dim: int,
			dropout: float = 0.05,
			context_dim: int = None
	) -> None:
		super(GRN, self).__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.context_dim = context_dim
		self.hidden_dim = hidden_dim
		self.dropout = dropout

		# =================================================
		# Input conditioning components (Eq.4 in the original paper)
		# =================================================
		# for using direct residual connection the dimension of the input must match the output dimension.
		# otherwise, we'll need to project the input for creating this residual connection
		self.project_residual: bool = self.input_dim != self.output_dim
		if self.project_residual:
			self.skip_layer = nn.Linear(self.input_dim, self.output_dim)

		# A linear layer for projecting the primary input (acts across time if necessary)
		self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)

		# In case we expect context input, an additional linear layer will project the context
		if self.context_dim is not None:
			self.context_projection = nn.Linear(self.context_dim, self.hidden_dim, bias=False)
		# non-linearity to be applied on the sum of the projections
		self.elu1 = nn.ELU()

		# ============================================================
		# Further projection components (Eq.3 in the original paper)
		# ============================================================
		# additional projection on top of the non-linearity
		self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)

		# ============================================================
		# Output gating components (Eq.2 in the original paper)
		# ============================================================
		self.dropout = nn.Dropout(self.dropout)
		self.gate = GLU(self.output_dim)
		self.layernorm = nn.LayerNorm(self.output_dim)

	def forward(self, x, context=None):

		# compute residual (for skipping) if necessary
		if self.project_residual:
			residual = self.skip_layer(x)
		else:
			residual = x
		# ===========================
		# Compute Eq.4
		# ===========================
		x = self.fc1(x)
		if context is not None:
			context = self.context_projection(context)
			x = x + context

		# compute eta_2 (according to paper)
		x = self.elu1(x)

		# ===========================
		# Compute Eq.3
		# ===========================
		# compute eta_1 (according to paper)
		x = self.fc2(x)

		# ===========================
		# Compute Eq.2
		# ===========================
		x = self.dropout(x)
		x = self.gate(x)
		# perform skipping using the residual
		x = x + residual
		# apply normalization layer
		x = self.layernorm(x)

		return x


if __name__ == '__main__':
	x = torch.rand(3, 4, 5)
	grn = GRN(input_dim=5, hidden_dim=5, output_dim=5)
	torchinfo.summary(grn, x.shape)
	y = grn(x)
	z = 1
