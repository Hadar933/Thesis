from itertools import product
from typing import Any, Optional
from loguru import logger


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


if __name__ == '__main__':
    seq2seq_params = generate_hyperparam_combinations(
        global_args=dict(
            feature_lag=[30, 60, 90, 120],
            batch_size=[30, 60, 90, 120]
        ),
        model_args=dict(
            target_lag=[30, 60, 90, 120],
            enc_embedding_size=[30, 60, 90, 120],
            enc_hidden_size=[30, 60, 90, 120],
            enc_num_layers=[1],
            enc_bidirectional=[True]
        ),
        model_shared_pairs={
            'dec_hidden_size': 'enc_hidden_size',
            'dec_embedding_size': 'enc_embedding_size'
        }
    )
    for i, item in enumerate(seq2seq_params, start=1):
        print(f"{[i]} {item}")
