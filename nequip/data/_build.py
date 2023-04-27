import inspect
from importlib import import_module
from typing import Dict, List
from os import listdir
from os.path import isfile, join

from torch.utils.data import ConcatDataset
from nequip import data
from nequip.data.transforms import TypeMapper
from nequip.data import AtomicDataDict, register_fields
from nequip.utils import instantiate, get_w_prefix


def dataset_from_config(config, prefix: str = "dataset") -> ConcatDataset:
    """initialize database based on a config instance

    It needs dataset type name (case insensitive),
    and all the parameters needed in the constructor.

    Examples see tests/data/test_dataset.py TestFromConfig
    and tests/datasets/test_simplest.py

    Args:

    config (dict, nequip.utils.Config): dict/object that store all the parameters
    prefix (str): Optional. The prefix of all dataset parameters

    Return:

    dataset (torch.utils.data.ConcatDataset)
    """

    instances = []
    dataset_idx_offset = 0
    config_dataset_list: List[Dict] = config.get(f"{prefix}_list", [config])
    for dataset_idx, _config_dataset in enumerate(config_dataset_list):
        is_folder = False
        config_dataset_type = _config_dataset.get(prefix, None)
        if config_dataset_type is None:
            raise KeyError(f"Dataset with prefix `{prefix}` isn't present in this config!")
        
        if config_dataset_type.split("_")[0] == "folder":
            is_folder = True
            config_dataset_type = config_dataset_type.split("_")[1]
        
        if inspect.isclass(config_dataset_type):
            class_name = config_dataset_type
        else:
            try:
                module_name = ".".join(config_dataset_type.split(".")[:-1])
                class_name = ".".join(config_dataset_type.split(".")[-1:])
                class_name = getattr(import_module(module_name), class_name)
            except Exception:
                # ^ TODO: don't catch all Exception
                # default class defined in nequip.data or nequip.dataset
                dataset_name = config_dataset_type.lower()

                class_name = None
                for k, v in inspect.getmembers(data, inspect.isclass):
                    if k.endswith("Dataset"):
                        if k.lower() == dataset_name:
                            class_name = v
                        if k[:-7].lower() == dataset_name:
                            class_name = v
                    elif k.lower() == dataset_name:
                        class_name = v

        if class_name is None:
            raise NameError(f"dataset type {dataset_name} does not exists")

        f_name = _config_dataset.get(f"{prefix}_file_name")
        if is_folder:
            dataset_file_names = [join(f_name, f) for f in listdir(f_name) if isfile(join(f_name, f))]
        else:
            dataset_file_names = [f_name]
        
        _config: dict = config.as_dict()
        _config.update(_config_dataset)

        # if dataset r_max is not found, use the universal r_max
        eff_key = "extra_fixed_fields"
        prefixed_eff_key = f"{prefix}_{eff_key}"
        _config[prefixed_eff_key] = get_w_prefix(
            eff_key, {}, prefix=prefix, arg_dicts=_config
        )
        _config[prefixed_eff_key]["r_max"] = get_w_prefix(
            "r_max",
            prefix=prefix,
            arg_dicts=[_config[prefixed_eff_key], _config],
        )
        _config["using_bead_numbers"] = "bead_numbers" in _config_dataset.get("npz_fixed_field_keys", {})

        # Build a TypeMapper from the config
        type_mapper, _ = instantiate(TypeMapper, prefix=prefix, optional_args=_config)

        for dataset_file_name in dataset_file_names:
            _config[AtomicDataDict.DATASET_INDEX_KEY] = dataset_idx + dataset_idx_offset
            dataset_idx_offset += 1
            _config[f"{prefix}_file_name"] = dataset_file_name

            # Register fields:
            # This might reregister fields, but that's OK:
            instantiate(register_fields, all_args=_config)

            instance, _ = instantiate(
                class_name,
                prefix=prefix,
                positional_args={"type_mapper": type_mapper},
                optional_args=_config,
            )

            instances.append(instance)
    
    if config.get("train_on_delta", False):
        return ConcatDataset(instances), data.dataset.ReferenceConcatDataset(instances)
    return ConcatDataset(instances), None
