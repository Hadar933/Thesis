import importlib.util
import inspect
import json
from datetime import datetime
from ML.Utilities import utils
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import psutil
import torch
from tqdm import tqdm
import os
from ML.Utilities.normalizer import NormalizerFactory
import torchinfo


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
			model_class_name: str,
			model_args: dict,
			exp_name: str,
			optimizer_name: str,
			criterion_name: str,
			patience: int,
			patience_tolerance: float,
			n_epochs: int,
			seed: int,
			features_norm_method: str,
			targets_norm_method: str,
			features_global_normalizer: bool,
			targets_global_normalizer: bool,
			flip_history: bool
	):
		torch.manual_seed(seed)  # TODO: doesnt seem to do anything

		self.init_args = {key: val for key, val in locals().copy().items() if key != 'self'}
		for key, val in self.init_args.items():
			setattr(self, key, val)

		self._stop_training: bool = False
		self._early_stopping: int = 0
		self._best_val_loss: float | torch.Tensor = float('inf')
		self._tb_writer: SummaryWriter | None = None

		self.model_dir: str = ""
		self.best_model_path: str = ""
		self.info_path = ""

		self.features_normalizer = NormalizerFactory.create(features_norm_method, features_global_normalizer)
		self.targets_normalizer = NormalizerFactory.create(targets_norm_method, targets_global_normalizer)
		self.features: torch.Tensor = torch.load(features_path).float()
		self.targets: torch.Tensor = torch.load(targets_path).float()
		if flip_history:  # fliplr is an alternative to [:,::-1,:], which is not yet supported as is in torch
			self.features = torch.fliplr(self.features)
			self.targets = torch.fliplr(self.targets)
		self.device: torch.cuda.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.model: torch.nn.Module = self._construct_model()
		self.criterion: torch.nn.modules.loss = getattr(torch.nn, self.criterion_name)()
		self.optimizer: torch.optim.Optimizer = getattr(torch.optim, optimizer_name)(self.model.parameters())

		self._create_model_dir()
		self.train_loader, self.val_loader, self.all_test_loaders = self._set_loaders()

	def _save_init_args(self, init_locals):
		local_vars = dict()
		for key, val in init_locals.items():
			if (
					key == 'self' or
					key.startswith('__') or
					inspect.ismodule(val) or
					inspect.isclass(val) or
					inspect.isfunction(val)
			):
				continue
			local_vars[key] = val
		return local_vars

	def _construct_model(self):
		file_path = os.path.join('Zoo', f'{self.model_class_name}.py')
		if not os.path.exists(file_path):
			raise FileNotFoundError(f"The model file '{file_path}' does not exist")

		spec = importlib.util.spec_from_file_location(self.model_class_name, file_path)
		module = importlib.util.module_from_spec(spec)
		spec.loader.exec_module(module)
		model_class = getattr(module, self.model_class_name)
		model = model_class(**self.model_args)
		torchinfo.summary(model, input_size=(self.batch_size, self.feature_win, self.features.shape[-1]))

		return model.to(self.device)

	def _create_model_dir(self):
		""" called when a trainer is initialized and creates a model dir with relevant information txt file(s) """
		init_timestamp = datetime.now().strftime(utils.TIME_FORMAT)
		self.model_dir = os.path.join(os.getcwd(), 'saved_models', f"{self.model}_{self.exp_name}_{init_timestamp}")
		self.info_path = os.path.join(self.model_dir, 'trainer_info.yaml')
		if not os.path.exists(self.model_dir):
			os.makedirs(self.model_dir)
		# trainer_info = {  # TODO: USE INTROSPECTION!
		# 	'features_path': self.features_path,
		# 	'targets_path': self.targets_path,
		# 	'train_percent': self.train_percent,
		# 	'val_percent': self.val_percent,
		# 	'feature_win': self.feature_win,
		# 	'target_win': self.target_win,
		# 	'intersect': self.intersect,
		# 	'batch_size': self.batch_size,
		# 	'model_name': self.model_class_name,
		# 	'model_args': self.model_args,
		# 	'exp_name': self.exp_name,
		# 	'optimizer_name': self.optimizer_name,
		# 	'criterion_name': self.criterion_name,
		# 	'patience': self.patience,
		# 	'patience_tolerance': self.patience_tolerance,
		# 	'n_epochs': self.n_epochs,
		# 	'seed': self.seed,
		# 	'features_norm_method': self.features_norm_method,
		# 	'targets_norm_method': self.targets_norm_method,
		# 	'features_global_normalizer': self.features_global_normalization,
		# 	'targets_global_normalizer': self.targets_global_normalizer,
		# 	'flip_history': self.flip_history
		# }
		utils.update_json(self.info_path, self.init_args)

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
			if self.best_model_path:
				os.remove(self.best_model_path)
			self.best_model_path = os.path.join(self.model_dir, f"{self.model}_{self.exp_name}_{time}.pt")
			torch.save(self.model.state_dict(), self.best_model_path)
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
		fit_info = {
			"early_stopping": self._early_stopping,
			"best_val_loss": self._best_val_loss.item(),
			"best_model_path": self.best_model_path
		}
		utils.update_json(self.info_path, fit_info)

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
		all_preds = utils.format_df_torch_entries(all_preds)
		return all_preds

	@classmethod
	def from_model_dirname(cls, model_dirname):
		model_pt_path = ""
		trainer_yaml_path = ""
		for file_name in os.listdir(model_dirname):
			if file_name.endswith('.pt'):
				model_pt_path = os.path.join(model_dirname, file_name)
			elif file_name.endswith('.yaml'):
				trainer_yaml_path = os.path.join(model_dirname, file_name)

		with open(trainer_yaml_path, 'r') as f:
			trainer_args = json.load(f)

		trainer = cls(**trainer_args)
		trainer.model = trainer.model.load_state_dict(torch.load(model_pt_path, map_location=trainer.device))
		trainer.model.eval()
		return trainer
