import os
from typing import Optional
import sys
import copy
import argparse
import logging
import textwrap
from pathlib import Path
import contextlib
from tqdm.auto import tqdm
from nequip.data import AtomicDataDict

import ase.io
import torch

from nequip.data import AtomicData, Collater, dataset_from_config, register_fields
from nequip.scripts.deploy import load_deployed_model, R_MAX_KEY
from nequip.scripts._logger import set_up_script_logger
from nequip.scripts.train import default_config, check_code_version
from nequip.utils._global_options import _set_global_options
from nequip.train import Trainer, Loss, Metrics
from nequip.utils import load_file, instantiate, Config


ORIGINAL_DATASET_INDEX_KEY: str = "original_dataset_index"
register_fields(graph_fields=[ORIGINAL_DATASET_INDEX_KEY])

def register_field(field_name: str):
    split = field_name.split(":")
    if len(split) == 1:
        return split
    assert len(split) == 2
    name, field = split
    if field == "node":
        register_fields(node_fields=[name])
    elif field == "edge":
        register_fields(edge_fields=[name])
    elif field == "graph":
        register_fields(graph_fields=[name])
    elif field == "long":
        register_fields(long_fields=[name])
    else:
        raise ValueError(f"{name} is not a valid field type. Permissible field types are [node, edge, graph, long]")
    return name


def main(args=None, running_as_script: bool = True):
    # in results dir, do: nequip-deploy build --train-dir . deployed.pth
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(
            """Compute the error of a model on a test set using various metrics.

            The model, metrics, dataset, etc. can specified individually, or a training session can be indicated with `--train-dir`.
            In order of priority, the global settings (dtype, TensorFloat32, etc.) are taken from:
              1. The model config (for a training session)
              2. The dataset config (for a deployed model)
              3. The defaults

            Prints only the final result in `name = num` format to stdout; all other information is logging.debuged to stderr.

            WARNING: Please note that results of CUDA models are rarely exactly reproducible, and that even CPU models can be nondeterministic.
            """
        )
    )
    parser.add_argument(
        "--train-dir",
        help="Path to a working directory from a training session.",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--model",
        help="A deployed or pickled NequIP model to load. If omitted, defaults to `best_model.pth` in `train_dir`.",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--dataset-config",
        help="A YAML config file specifying the dataset to load test data from. If omitted, `config.yaml` in `train_dir` will be used",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--metrics-config",
        help="A YAML config file specifying the metrics to compute. If omitted, `config.yaml` in `train_dir` will be used. If the config does not specify `metrics_components`, the default is to logging.debug MAEs and RMSEs for all fields given in the loss function. If the literal string `None`, no metrics will be computed.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--test-indexes",
        help="Path to a file containing the indexes in the dataset that make up the test set. If omitted, all data frames *not* used as training or validation data in the training session `train_dir` will be used.",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--stride",
        help="If dataset config is provided and test indexes are not provided, take all dataset idcs with this stride",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--batch-size",
        help="Batch size to use. Larger is usually faster on GPU. If you run out of memory, lower this.",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--repeat",
        help=(
            "Number of times to repeat evaluating the test dataset. "
            "This can help compensate for CUDA nondeterminism, or can be used to evaluate error on models whose inference passes are intentionally nondeterministic. "
            "Note that `--repeat`ed passes over the dataset will also be `--output`ed if an `--output` is specified."
        ),
        type=int,
        default=1,
    )
    parser.add_argument(
        "--use-deterministic-algorithms",
        help="Try to have PyTorch use deterministic algorithms. Will probably fail on GPU/CUDA.",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--device",
        help="Device to run the model on. If not provided, defaults to CUDA if available and CPU otherwise.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output",
        help="ExtXYZ (.xyz) file to write out the test set and model predictions to.",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-fields",
        help="Extra fields (names[:field] comma separated with no spaces) to write to the `--output`.\n"
             "Field options are: [node, edge, graph, long].\n"
             "If [:field] is omitted, the field with that name is assumed to be already registered by default.",
        type=str,
        default="",
    )
    parser.add_argument(
        "--log",
        help="log file to store all the metrics and screen logging.debug",
        type=Path,
        default=None,
    )
    # Something has to be provided
    # See https://stackoverflow.com/questions/22368458/how-to-make-argparse-logging.debug-usage-when-no-option-is-given-to-the-code
    if len(sys.argv) == 1:
        parser.print_help()
        parser.exit()
    # Parse the args
    args = parser.parse_args(args=args)

    # Do the defaults:
    dataset_is_from_training: bool = False
    if args.train_dir:
        if args.dataset_config is None:
            args.dataset_config = args.train_dir / "config.yaml"
            dataset_is_from_training = True
        if args.metrics_config is None:
            args.metrics_config = args.train_dir / "config.yaml"
        if args.model is None:
            args.model = args.train_dir / "best_model.pth"
        if args.test_indexes is None:
            # Find the remaining indexes that arent train or val
            trainer = torch.load(
                str(args.train_dir / "trainer.pth"), map_location="cpu"
            )
            train_idcs = []
            dataset_offset = 0
            for tr_idcs in trainer["train_idcs"]:
                train_idcs.extend([tr_idx + dataset_offset for tr_idx in tr_idcs.tolist()])
                dataset_offset += len(tr_idcs)
            train_idcs = set(train_idcs)
            val_idcs = []
            dataset_offset = 0
            for v_idcs in trainer["val_idcs"]:
                val_idcs.extend([v_idx + dataset_offset for v_idx in v_idcs.tolist()])
                dataset_offset += len(v_idcs)
            val_idcs = set(val_idcs)
        else:
            train_idcs = val_idcs = None
    # update
    if args.metrics_config == "None":
        args.metrics_config = None
    elif args.metrics_config is not None:
        args.metrics_config = Path(args.metrics_config)
    do_metrics = args.metrics_config is not None
    # validate
    if args.dataset_config is None:
        raise ValueError("--dataset-config or --train-dir must be provided")
    if args.metrics_config is None and args.output is None:
        raise ValueError(
            "Nothing to do! Must provide at least one of --metrics-config, --train-dir (to use training config for metrics), or --output"
        )
    if args.model is None:
        raise ValueError("--model or --train-dir must be provided")
    output_type: Optional[str] = None
    if args.output is not None:
        if args.output.suffix != ".xyz":
            raise ValueError("Only .xyz format for `--output` is supported.")
        args.output_fields = [register_field(e) for e in args.output_fields.split(",") if e != ""] + [
            ORIGINAL_DATASET_INDEX_KEY
        ]
        output_type = "xyz"
    else:
        assert args.output_fields == ""
        args.output_fields = []

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if running_as_script:
        set_up_script_logger(args.log)
    logger = logging.getLogger("nequip-evaluate")
    logger.setLevel(logging.INFO)

    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(
            "WARNING: please note that models running on CUDA are usually nondeterministc and that this manifests in the final test errors; for a _more_ deterministic result, please use `--device cpu`",
        )

    if args.use_deterministic_algorithms:
        logger.info(
            "Telling PyTorch to try to use deterministic algorithms... please note that this will likely error on CUDA/GPU"
        )
        torch.use_deterministic_algorithms(True)

    # Load model:
    logger.info("Loading model... ")
    loaded_deployed_model: bool = False
    model_r_max = None
    try:
        model, metadata = load_deployed_model(
            args.model,
            device=device,
            set_global_options=True,  # don't warn that setting
        )
        logger.info("loaded deployed model.")
        # the global settings for a deployed model are set by
        # set_global_options in the call to load_deployed_model
        # above
        model_r_max = float(metadata[R_MAX_KEY])
        loaded_deployed_model = True
    except ValueError:  # its not a deployed model
        loaded_deployed_model = False
    # we don't do this in the `except:` block to avoid "during handing of this exception another exception"
    # chains if there is an issue loading the training session model. This makes the error messages more
    # comprehensible:
    if not loaded_deployed_model:
        # Use the model config, regardless of dataset config
        global_config = args.model.parent / "config.yaml"
        global_config = Config.from_file(str(global_config), defaults=default_config)
        _set_global_options(global_config)
        check_code_version(global_config)
        del global_config

        # load a training session model
        model, model_config = Trainer.load_model_from_training_session(
            traindir=args.model.parent, model_name=args.model.name
        )
        model = model.to(device)
        logger.info("loaded model from training session")
        model_root = model_config["root"]
        model_r_max = model_config["r_max"]
    model.eval()

    # Load a config file
    logger.info(
        f"Loading {'original ' if dataset_is_from_training else ''}dataset...",
    )
    defaults = {"r_max": model_r_max}
    if not loaded_deployed_model:
        defaults.update({
            "root": model_root
        })
    dataset_config = Config.from_file(
        str(args.dataset_config), defaults=defaults
    )
    if dataset_config["r_max"] != model_r_max:
        raise RuntimeError(
            f"Dataset config has r_max={dataset_config['r_max']}, but model has r_max={model_r_max}!"
        )

    dataset_is_test: bool = False
    dataset_is_validation: bool = False
    try:
        # Try to get test dataset
        dataset, _ = dataset_from_config(dataset_config, prefix="test_dataset")
        dataset_is_test = True
    except KeyError:
        pass
    if not dataset_is_test:
        try:
            # Try to get validation dataset
            dataset, _ = dataset_from_config(dataset_config, prefix="validation_dataset")
            dataset_is_validation = True
        except KeyError:
            pass
    
    if not (dataset_is_test or dataset_is_validation):
        # Get shared train + validation dataset
        # prefix `dataset`
        dataset, _ = dataset_from_config(dataset_config)
    logger.info(
        f"Loaded {'test_' if dataset_is_test else 'validation_' if dataset_is_validation else ''}dataset specified in {args.dataset_config.name}.",
    )

    c = Collater.for_dataset(dataset, exclude_keys=[])

    # Determine the test set
    # this makes no sense if a dataset is given seperately
    if (
        args.test_indexes is None
        and dataset_is_from_training
        and train_idcs is not None
    ):
        if dataset_is_test:
            test_idcs = torch.arange(len(dataset))
            logger.info(
                f"Using all frames from original test dataset, yielding a test set size of {len(test_idcs)} frames.",
            )
        else:
            # we know the train and val, get the rest
            all_idcs = set(range(len(dataset)))
            # set operations
            if dataset_is_validation:
                test_idcs = list(all_idcs - val_idcs)
                logger.info(
                    f"Using origial validation dataset ({len(dataset)} frames) minus validation set frames ({len(val_idcs)} frames), yielding a test set size of {len(test_idcs)} frames.",
                )
            else:
                test_idcs = list(all_idcs - train_idcs - val_idcs)
                assert set(test_idcs).isdisjoint(train_idcs)
                logger.info(
                    f"Using origial training dataset ({len(dataset)} frames) minus training ({len(train_idcs)} frames) and validation frames ({len(val_idcs)} frames), yielding a test set size of {len(test_idcs)} frames.",
                )
            # No matter what it should be disjoint from validation:
            assert set(test_idcs).isdisjoint(val_idcs)
            if not do_metrics:
                logger.info(
                    "WARNING: using the automatic test set ^^^ but not computing metrics, is this really what you wanted to do?",
                )
    elif args.test_indexes is None:
        # Default to all frames
        test_idcs = torch.arange(len(dataset))
        logger.info(
            f"Using all frames from the specified test dataset with stride {args.stride}, yielding a test set size of {len(test_idcs)} frames.",
        )
    else:
        # load from file
        test_idcs = load_file(
            supported_formats=dict(
                torch=["pt", "pth"], yaml=["yaml", "yml"], json=["json"]
            ),
            filename=str(args.test_indexes),
        )
        logger.info(
            f"Using provided test set indexes, yielding a test set size of {len(test_idcs)} frames.",
        )
    test_idcs = torch.as_tensor(test_idcs, dtype=torch.long)[::args.stride]
    test_idcs = test_idcs.tile((args.repeat,))

    # Figure out what metrics we're actually computing
    if do_metrics:
        metrics_config = Config.from_file(str(args.metrics_config))
        metrics_components = metrics_config.get("metrics_components", None)
        # See trainer.py: init() and init_metrics()
        # Default to loss functions if no metrics specified:
        if metrics_components is None:
            loss, _ = instantiate(
                builder=Loss,
                prefix="loss",
                positional_args=dict(coeffs=metrics_config.loss_coeffs),
                all_args=metrics_config,
            )
            metrics_components = []
            for key, func in loss.funcs.items():
                params = {
                    "PerSpecies": type(func).__name__.startswith("PerSpecies"),
                }
                metrics_components.append((key, "mae", params))
                metrics_components.append((key, "rmse", params))

        metrics, _ = instantiate(
            builder=Metrics,
            prefix="metrics",
            positional_args=dict(components=metrics_components),
            all_args=metrics_config,
        )
        metrics.to(device=device)

    batch_i: int = 0
    batch_size: int = args.batch_size
    stop = False
    already_computed_nodes = None

    logger.info("Starting...")
    context_stack = contextlib.ExitStack()
    with contextlib.ExitStack() as context_stack:
        # "None" checks if in a TTY and disables if not
        prog = context_stack.enter_context(tqdm(total=len(test_idcs), disable=None))
        if do_metrics:
            display_bar = context_stack.enter_context(
                tqdm(
                    bar_format=""
                    if prog.disable  # prog.ncols doesn't exist if disabled
                    else ("{desc:." + str(prog.ncols) + "}"),
                    disable=None,
                )
            )

        if output_type is not None:
            output = []
            output_target = []
            dataset_idx_to_idx = dict()
            for idx, ds in enumerate(dataset.datasets):
                ds_filename = ".".join(os.path.split(ds.file_name)[-1].split(".")[:-1])
                path, out_filename = os.path.split(args.output)
                out_filename_split = out_filename.split(".")
                dataset_idx_to_idx[ds.dataset_idx] = idx
                output_filename = ".".join([f"ds_{ds.dataset_idx}__{ds_filename}__" + ".".join(out_filename_split[:-1])] + out_filename_split[-1:])
                output.append(context_stack.enter_context(open(os.path.join(path, output_filename), "w")))
                output_target_filename = ".".join([f"ds_{ds.dataset_idx}__{ds_filename}__" + ".".join(out_filename_split[:-1]) + "_target"] + out_filename_split[-1:])
                output_target.append(context_stack.enter_context(open(os.path.join(path, output_target_filename), "w")))
        else:
            output = None
            output_target = None

        while True:
            complete_out = None
            torch.cuda.empty_cache()
            while True:
                this_batch_test_indexes = test_idcs[
                    batch_i * batch_size : (batch_i + 1) * batch_size
                ]
                try:
                    datas = [dataset[int(idex)] for idex in this_batch_test_indexes]
                except ValueError: # Most probably an atom in pdb that is missing in model
                    batch_i += 1
                    prog.update(len(this_batch_test_indexes))
                    continue
                if len(datas) == 0:
                    stop = True
                    break

                out, batch, already_computed_nodes = evaluate(c, datas, device, model, already_computed_nodes)
                if complete_out is None:
                    complete_out = copy.deepcopy(batch)
                    if AtomicDataDict.PER_ATOM_ENERGY_KEY in out:
                        complete_out[AtomicDataDict.PER_ATOM_ENERGY_KEY] = torch.zeros(
                            (len(batch[AtomicDataDict.POSITIONS_KEY]), 1),
                            dtype=torch.get_default_dtype(),
                            device=out[AtomicDataDict.PER_ATOM_ENERGY_KEY].device
                        )
                    if AtomicDataDict.TOTAL_ENERGY_KEY in out:
                        complete_out[AtomicDataDict.TOTAL_ENERGY_KEY] = torch.zeros_like(
                            out[AtomicDataDict.TOTAL_ENERGY_KEY]
                        )
                if AtomicDataDict.PER_ATOM_ENERGY_KEY in complete_out:
                    original_nodes = out[AtomicDataDict.ORIG_EDGE_INDEX_KEY][0].unique()
                    nodes = out[AtomicDataDict.EDGE_INDEX_KEY][0].unique()
                    complete_out[AtomicDataDict.PER_ATOM_ENERGY_KEY][original_nodes] = out[AtomicDataDict.PER_ATOM_ENERGY_KEY][nodes].detach()
                
                if AtomicDataDict.FORCE_KEY in complete_out:
                    complete_out[AtomicDataDict.FORCE_KEY][original_nodes] = out[AtomicDataDict.FORCE_KEY][nodes].detach()

                if AtomicDataDict.TOTAL_ENERGY_KEY in complete_out:
                    complete_out[AtomicDataDict.TOTAL_ENERGY_KEY] += out[AtomicDataDict.TOTAL_ENERGY_KEY].detach()
                del out

                if already_computed_nodes is None:
                    break
            if stop:
                break

            with torch.no_grad():
                # Write output
                if output_type == "xyz":
                    # add test frame to the output:
                    complete_out[ORIGINAL_DATASET_INDEX_KEY] = torch.LongTensor(
                        this_batch_test_indexes
                    )
                    batch[ORIGINAL_DATASET_INDEX_KEY] = torch.LongTensor(
                        this_batch_test_indexes
                    )
                    # append to the file
                    for dataset_idx in torch.unique(complete_out['dataset_idx']).to('cpu').tolist():
                        idx = dataset_idx_to_idx[dataset_idx]
                        ase.io.write(
                            output[idx],
                            AtomicData.from_AtomicDataDict(complete_out)
                            .to(device="cpu")
                            .to_ase(
                                type_mapper=dataset.datasets[idx].type_mapper,
                                extra_fields=args.output_fields,
                                filter_idcs=(complete_out['dataset_idx'] == dataset_idx).to('cpu'),
                            ),
                            format="extxyz",
                            append=True,
                        )
                        ase.io.write(
                            output_target[idx],
                            AtomicData.from_AtomicDataDict(batch)
                            .to(device="cpu")
                            .to_ase(
                                type_mapper=dataset.datasets[idx].type_mapper,
                                filter_idcs=(complete_out['dataset_idx'] == dataset_idx).to('cpu'),
                            ),
                            format="extxyz",
                            append=True,
                        )

            # Accumulate metrics
            if do_metrics:
                try:
                    metrics(complete_out, batch)
                    display_bar.set_description_str(
                        " | ".join(
                            f"{k} = {v:4.4f}"
                            for k, v in metrics.flatten_metrics(
                                metrics.current_result(),
                                type_names=dataset.datasets[0].type_mapper.type_names,
                            )[0].items()
                        )
                    )
                except:
                    display_bar.set_description_str(
                        "No metrics available for this dataset. Ground truth may be missing."
                    )

            batch_i += 1
            prog.update(len(batch['ptr'] - 1))

        prog.close()
        if do_metrics:
            display_bar.close()

    if do_metrics:
        logger.info("\n--- Final result: ---")
        logger.critical(
            "\n".join(
                f"{k:>20s} = {v:< 20f}"
                for k, v in metrics.flatten_metrics(
                    metrics.current_result(),
                    type_names=dataset.datasets[0].type_mapper.type_names,
                )[0].items()
            )
        )

def evaluate(c, datas, device, model, already_computed_nodes=None):
    batch = c.collate(datas)
    batch = batch.to(device)
    batch_ = AtomicData.to_AtomicDataDict(batch)

    # if AtomicDataDict.PER_ATOM_ENERGY_KEY in batch_:
    #     not_nan_edge_filter = torch.isin(batch_[AtomicDataDict.EDGE_INDEX_KEY][0], torch.argwhere(~torch.isnan(batch_[AtomicDataDict.PER_ATOM_ENERGY_KEY].flatten())).flatten())
    #     batch_[AtomicDataDict.EDGE_INDEX_KEY] = batch_[AtomicDataDict.EDGE_INDEX_KEY][:, not_nan_edge_filter]
    #     if AtomicDataDict.EDGE_CELL_SHIFT_KEY in batch_:
    #         batch_[AtomicDataDict.EDGE_CELL_SHIFT_KEY] = batch_[AtomicDataDict.EDGE_CELL_SHIFT_KEY][not_nan_edge_filter]
    #     batch_[AtomicDataDict.ORIG_BATCH_KEY] = batch_[AtomicDataDict.BATCH_KEY].clone()
    #     batch_[AtomicDataDict.BATCH_KEY] = batch_[AtomicDataDict.BATCH_KEY][~torch.isnan(batch_[AtomicDataDict.PER_ATOM_ENERGY_KEY]).flatten()]

    # Limit maximum batch size to avoid CUDA Out of Memory
    x = batch_[AtomicDataDict.EDGE_INDEX_KEY]
    y = x.clone()
    z = batch_.get(AtomicDataDict.PER_ATOM_ENERGY_KEY).clone() if AtomicDataDict.PER_ATOM_ENERGY_KEY in batch_ else None
    if already_computed_nodes is not None:
        y = y[:, ~torch.isin(y[0], already_computed_nodes)]
    node_center_idcs = y[0].unique()
    max_atoms_correction = 0
    z_mask = None

    while True:
        batch_atom_idcs = node_center_idcs[torch.multinomial(
            node_center_idcs.float(),
            num_samples=min(len(node_center_idcs), 3000 - max_atoms_correction),
            replacement=False
        ).sort().values]
        batch_size_edge_filter = torch.isin(x[0], batch_atom_idcs)
        y = x[:, batch_size_edge_filter]
        if z is not None:
            z_mask = torch.ones_like(z, dtype=torch.bool)
            z_mask[y[0].unique()] = False
            z = batch_.get(AtomicDataDict.PER_ATOM_ENERGY_KEY).clone()
            z[z_mask] = torch.nan
        max_atoms_correction += 100
        if y.shape[1] <= 40000 and (z is None or (~torch.isnan(z)).sum() <= 1800):
            break
    x_ulen = len(x[0].unique())
    del x

    if max_atoms_correction > 0:
        batch_[AtomicDataDict.EDGE_INDEX_KEY] = y
        batch_[AtomicDataDict.EDGE_CELL_SHIFT_KEY] = batch_[AtomicDataDict.EDGE_CELL_SHIFT_KEY][batch_size_edge_filter]
        batch_[AtomicDataDict.BATCH_KEY] = batch_.get(AtomicDataDict.ORIG_BATCH_KEY, batch_[AtomicDataDict.BATCH_KEY])[y[0].unique()]
        if z is not None and z_mask is not None:
            batch_[AtomicDataDict.PER_ATOM_ENERGY_KEY] = z

    # Remove all atoms that do not appear in edges and update edge indices
    edge_index = batch_[AtomicDataDict.EDGE_INDEX_KEY]

    edge_index_unique = edge_index.unique()
    batch_[AtomicDataDict.POSITIONS_KEY] = batch_[AtomicDataDict.POSITIONS_KEY][edge_index_unique]
    if AtomicDataDict.ATOMIC_NUMBERS_KEY in batch_:
        batch_[AtomicDataDict.ATOMIC_NUMBERS_KEY] = batch_[AtomicDataDict.ATOMIC_NUMBERS_KEY][edge_index_unique]
    if AtomicDataDict.ATOM_TYPE_KEY in batch_:
        batch_[AtomicDataDict.ATOM_TYPE_KEY] = batch_[AtomicDataDict.ATOM_TYPE_KEY][edge_index_unique]
    if AtomicDataDict.PER_ATOM_ENERGY_KEY in batch_:
        batch_[AtomicDataDict.PER_ATOM_ENERGY_KEY] = batch_[AtomicDataDict.PER_ATOM_ENERGY_KEY][edge_index_unique]
    if AtomicDataDict.ORIG_BATCH_KEY in batch_:
        batch_[AtomicDataDict.ORIG_BATCH_KEY] = batch_[AtomicDataDict.ORIG_BATCH_KEY][edge_index_unique]

    last_idx = -1
    batch_[AtomicDataDict.ORIG_EDGE_INDEX_KEY] = edge_index.clone()
    updated_edge_index = edge_index.clone()
    for idx in edge_index_unique:
        if idx > last_idx + 1:
            updated_edge_index[edge_index >= idx] -= idx - last_idx - 1
        last_idx = idx
    batch_[AtomicDataDict.EDGE_INDEX_KEY] = updated_edge_index

    node_index_unique = edge_index[0].unique()
    del edge_index
    del edge_index_unique

    out = model(batch_)
    # from e3nn import o3
    # R = o3.Irreps('1o').D_from_angles(*[torch.tensor(x) for x in [0, 90, 0]]).to(batch_['pos'].device)
    # batch_['pos'] = torch.einsum("ij,zj->zi", R, batch_['pos'])
    # out2 = model(batch_)
    # R2 = o3.Irreps('1o').D_from_angles(*[torch.tensor(x) for x in [0, -90, 0]]).to(batch_['pos'].device)
    # o1 = out['forces']
    # o2 = torch.einsum("ij,zj->zi", R2, out2['forces'])

    if already_computed_nodes is None:
        if len(node_index_unique) < x_ulen:
            already_computed_nodes = node_index_unique
    elif len(already_computed_nodes) + len(node_index_unique) == x_ulen:
        already_computed_nodes = None
    else:
        assert len(already_computed_nodes) + len(node_index_unique) < x_ulen
        already_computed_nodes = torch.cat([already_computed_nodes, node_index_unique], dim=0)

    return out, AtomicData.to_AtomicDataDict(batch), already_computed_nodes

if __name__ == "__main__":
    main(running_as_script=True)