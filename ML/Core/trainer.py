from datetime import datetime
from ML.Utilities import utils
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import psutil
import torch
from tqdm import tqdm
import os
from ML.Utilities.normalizer import NormalizerFactory


class Trainer:
	"""
	The Trainer class takes in a torch model, alongside some training parameters, and performs testing,
	evaluating and prediction.
	"""

	def __init__(
			self,
			features_path: str,
			targets_path: str,
			train_percent: float,
			val_percent: float,
			feature_win: int,
			target_win: int,
			intersect: int,
			batch_size: int,
			model: torch.nn.Module,
			exp_name: str,
			optimizer: torch.optim.Optimizer,
			criterion: torch.nn.modules.loss._Loss,
			device: torch.cuda.device,
			patience: int,
			patience_tolerance: float,
			n_epochs: int,
			seed: int,
			features_norm_method: str,
			targets_norm_method: str,
			features_global_normalizer: bool,
			targets_global_normalizer: bool
	):
		torch.manual_seed(seed)  # TODO: doesnt seem to do anything

		self.feature_win = feature_win
		self.target_win = target_win
		self.intersect = intersect

		self.batch_size = batch_size

		self.val_percent = val_percent
		self.train_percent = train_percent

		self.exp_name = exp_name

		self.features_normalizer = NormalizerFactory.create(features_norm_method, features_global_normalizer)
		self.targets_normalizer = NormalizerFactory.create(targets_norm_method, targets_global_normalizer)

		self.features_path = features_path
		self.targets_path = targets_path
		self.features = torch.load(features_path).float()
		self.targets = torch.load(targets_path).float()

		self.patience_tolerance: float = patience_tolerance
		self.patience: int = patience
		self._stop_training: bool = False
		self._early_stopping: int = 0

		self.model: torch.nn.Module = model.to(device)

		self.criterion: torch.nn.modules.loss = criterion
		self.optimizer: torch.optim.Optimizer = optimizer
		self.device: torch.cuda.device = device
		self._best_val_loss: float = float('inf')

		self.n_epochs: int = n_epochs

		self._tb_writer: SummaryWriter = None

		self.model_dir: str = ""
		self._best_model_path: str = ""
		self._info_path = ""
		self._create_model_dir()

		self.train_loader, self.val_loader, self.all_test_loaders = self._set_loaders()

	def _create_model_dir(self):
		""" called when a trainer is initialized and creates a model dir with relevant information txt file(s) """
		init_timestamp = datetime.now().strftime(utils.TIME_FORMAT)
		self.model_dir = os.path.join(os.getcwd(), 'saved_models', f"{self.model}_{self.exp_name}_{init_timestamp}")
		self._info_path = os.path.join(self.model_dir, 'trainer_info.yaml')
		if not os.path.exists(self.model_dir):
			os.makedirs(self.model_dir)
		# TODO: yaml file looks bad
		utils.update_json(self._info_path, {'model': str(self.model), 'patience': self.patience,
											'patience_tolerance': self.patience_tolerance, 'loss': str(self.criterion),
											'optim': str(self.optimizer), 'epochs': self.n_epochs})

	def _set_loaders(self):
		""" sets the train/val/test loaders and updates the training statistics in the yaml (for normalization)  """
		data_dict = utils.train_val_test_split(self.features, self.targets, self.train_percent, self.val_percent,
											   self.feature_win, self.target_win, self.intersect, self.batch_size,
											   self.features_normalizer, self.targets_normalizer)
		return data_dict['train']['loader'], data_dict['val']['loader'], data_dict['test']['loader']

	def _train_one_epoch(self, epoch: int) -> float:
		""" performs a training process for one epoch """
		self.model.train(True)
		total_loss = 0.0
		tqdm_loader = tqdm(self.train_loader)
		for inputs, targets in tqdm_loader:
			self.optimizer.zero_grad()
			inputs, targets = inputs.to(self.device), targets.to(self.device)
			predictions = self.model(inputs)
			loss = self.criterion(predictions, targets)
			total_loss += loss
			loss.backward()
			self.optimizer.step()
			tqdm_loader.set_postfix({'Train Loss': f"{loss.item():.5f}",
									 'RAM_%': psutil.virtual_memory().percent,
									 'Epoch': epoch})
		total_loss /= len(self.train_loader)
		return total_loss

	def _evaluate(self, epoch: int) -> float:
		""" evaluates the model on the validation set """
		self.model.train(False)
		total_loss = 0.0
		with torch.no_grad():
			tqdm_loader = tqdm(self.val_loader)
			for inputs, targets in tqdm_loader:
				inputs, targets = inputs.to(self.device), targets.to(self.device)
				predictions = self.model(inputs)
				loss = self.criterion(predictions, targets)
				total_loss += loss
				tqdm_loader.set_postfix({'Val Loss': f"{loss.item():.5f}",
										 'RAM_%': psutil.virtual_memory().percent,
										 'Epoch': epoch})
			total_loss /= len(self.val_loader)
		return total_loss

	def _finish_epoch(self, val_loss, prev_val_loss, time, epoch):
		""" saves the best model and performs early stopping if needed """
		if val_loss < self._best_val_loss:  # if val is improved, we update the best model
			self._best_val_loss = val_loss
			if self._best_model_path:
				os.remove(self._best_model_path)
			self._best_model_path = os.path.join(self.model_dir, f"{self.model}_{self.exp_name}_{time}.pt")
			torch.save(self.model.state_dict(), self._best_model_path)
		elif torch.abs(val_loss - prev_val_loss) <= self.patience_tolerance:  # if val doesn't improve, counter += 1
			self._early_stopping += 1
		if self._early_stopping >= self.patience:  # if patience value is reached, the training process halts
			self._stop_training = True
			print(f"[Early Stopping] at epoch #{epoch}.")

	def fit(self) -> None:
		""" fits the model to the training data, with early stopping """
		fit_time = datetime.now().strftime(utils.TIME_FORMAT)
		self._tb_writer = SummaryWriter(os.path.join('tb_runs', f"{self.model}_{self.exp_name}_{fit_time}"))
		prev_val_loss = float('inf')
		for epoch in range(self.n_epochs):
			train_avg_loss = self._train_one_epoch(epoch)
			val_avg_loss = self._evaluate(epoch)
			self._tb_writer.add_scalar('Loss/Train', train_avg_loss, epoch)
			self._tb_writer.add_scalar('Loss/Val', val_avg_loss, epoch)
			self._finish_epoch(val_avg_loss, prev_val_loss, fit_time, epoch)
			if self._stop_training:
				break
			prev_val_loss = val_avg_loss
		utils.update_json(self._info_path, {"early_stopping": self._early_stopping,
											"best_val_loss": self._best_val_loss.item(),
											"best_model_path": self._best_model_path})

	def predict(self) -> pd.DataFrame:
		""" creates a prediction for every test loader in our test loaders list """
		self.model.train(False)
		all_preds = pd.DataFrame()
		with torch.no_grad():
			for j, test_loader in enumerate(self.all_test_loaders):
				curr_preds, curr_trues = [], []
				for inputs_i, true_i in tqdm(test_loader, desc=f"Predicting on test loader {j}"):
					inputs_i = inputs_i.to(self.device)
					pred_i = self.model(inputs_i).to('cpu')
					pred_i = self.targets_normalizer.inverse_transform(pred_i)
					curr_preds.append(pred_i.squeeze())
					curr_trues.append(true_i.squeeze())
				all_preds[f"pred_{j}"] = curr_preds
				all_preds[f"true_{j}"] = curr_trues
		return all_preds

	def load_trained_model(self, trained_model_path: str) -> None:
		""" loads into the trainer a trained model from memory"""
		self.model.load_state_dict(torch.load(trained_model_path, map_location=self.device))
		self.model.eval()
