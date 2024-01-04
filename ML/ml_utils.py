import os.path
from itertools import product
from typing import Any, Optional

import pandas as pd
from loguru import logger

from ML.Core.trainer import Trainer


def generate_hyperparam_combinations(
        model_args: dict,
        global_args: dict,
        model_args_key: str,
        model_shared_pairs: Optional[dict[str, str]] = None,
        global_shared_pairs: Optional[dict[str, str]] = None
) -> list[dict[str, Any]]:
    """
    generates a cross product of hyperparameters to consider
    :param model_args: a dictionary with keys that represent specific model argument and a value that is a list
                         of relevant hyperparams to consider
    :param global_args: like model_args but unrelated to the model itself (not part of model init arguments)
    :param model_args_key: the key directing to the model arguments sub dictionary
    :param model_shared_pairs: a dictionary that maps model arg name(s) to other model arg name(s), that indicates if
                               the former should be set the same as the latter (and not part of the product)
    :param global_shared_pairs: same as model_shared_pairs, just for the global arguments.
    :return: a list of dictionary that represents runtime arguments, which is the cross product of all given args.
    """
    model_args_product = product(*(model_args[param] for param in model_args))
    model_param_names = model_args.keys()
    model_args_product_dicts = []
    for model_combination in model_args_product:
        combination_dict = dict(zip(model_param_names, model_combination))
        if model_shared_pairs is not None:
            for new_model_arg_key, existing_model_arg_key in model_shared_pairs.items():
                combination_dict[new_model_arg_key] = combination_dict[existing_model_arg_key]
        model_args_product_dicts.append(combination_dict)

    global_args_product = product(*(global_args[param] for param in global_args))
    global_param_names = global_args.keys()
    global_args_product_dicts = []
    for global_combination in global_args_product:
        combination_dict = dict(zip(global_param_names, global_combination))
        if global_shared_pairs is not None:
            for new_global_arg_key, existing_global_arg_key in model_shared_pairs.items():
                combination_dict[new_global_arg_key] = combination_dict[existing_global_arg_key]
        global_args_product_dicts.append(combination_dict)

    hyperparam_combinations = [
        {**{model_args_key: model_args_dict}, **global_args_dict} for global_args_dict in global_args_product_dicts for
        model_args_dict in model_args_product_dicts
    ]
    logger.info(f'Generated {len(hyperparam_combinations)} hyper-parameter combinations')
    return hyperparam_combinations


def predict_multiple_models(models_dirs: list[str], dataset_name: str, add_inputs: bool) ->pd.DataFrame:
    """
    given a list of pretrained model directory paths runs a prediction and aggregates the losses and predictions
    :param models_dirs: list of model directory paths
    :param dataset_name: either train val or test
    :param add_inputs: if True, stacks the input to the prediction datafarme (TODO)
    :return: the loss dataframe for each model and each dataset.
    """
    all_losses = pd.DataFrame()
    for model_dir in models_dirs:
        model_name = os.path.basename(model_dir)
        trainer = Trainer.from_model_dirname(model_dir)
        preds, losses = trainer.predict(dataset_name, add_inputs)
        all_losses = pd.concat([all_losses, losses.rename(index={losses.index[-1]: model_name})], axis=0)
    all_losses['mean'] = all_losses.mean(axis=1)
    return all_losses


if __name__ == '__main__':
    results = predict_multiple_models(
        models_dirs=[
            # TRANSFORMER:
            "/home/hadar/Thesis/ML/saved_models/LTSFVanillaTransformer[1, False, 4, 16, None, None, 0.05, 3, '3', 1, 16, 1, 'gelu', 2, 2, 3]_[ours,T=1]_2024-01-02_04-03-09",
            "/home/hadar/Thesis/ML/saved_models/LTSFVanillaTransformer[16, False, 4, 16, None, None, 0.05, 3, '3', 1, 16, 2, 'gelu', 2, 2, 3]_[ours,T=16]_2024-01-01_09-10-33",
            "/home/hadar/Thesis/ML/saved_models/LTSFVanillaTransformer[32, False, 4, 16, None, None, 0.05, 3, '3', 1, 16, 2, 'gelu', 2, 2, 3]_[ours,T=32]_2023-12-31_14-26-35",
            "/home/hadar/Thesis/ML/saved_models/LTSFVanillaTransformer[64, False, 4, 32, None, None, 0.05, 3, '3', 1, 32, 1, 'gelu', 2, 2, 3]_[ours,T=64]_2023-12-31_19-34-47",
            "/home/hadar/Thesis/ML/saved_models/LTSFVanillaTransformer[128, False, 4, 16, None, None, 0.05, 3, '3', 1, 16, 2, 'gelu', 2, 1, 3]_[ours,T=128]_2024-01-01_00-15-22"
        ],
        dataset_name='test',
        add_inputs=False
    )