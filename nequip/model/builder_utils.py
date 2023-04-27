from functools import partial
from typing import Optional

import torch
from torch.utils.data import ConcatDataset

from nequip.utils import Config
from nequip.data import AtomicDataset, AtomicDataDict


def _add_avg_num_neighbors_helper(data, graph_is_undirected: bool = True):
    counts = torch.unique(
        data[AtomicDataDict.EDGE_INDEX_KEY][0],
        sorted=True,
        return_counts=True,
    )[1]
    # By default, assume that the graph is undirected, hence if a node does
    # not appear in the source index, it means that it has 0 neighbours.
    # If graph_is_undirected is False, ignore atoms with 0 neighbours because
    # the graph is directed and we simply are counting the number of incoming edges
    # in nodes that have at least one incoming edge
    
    # When graph is undirected, in case the cutoff is small and some nodes have no neighbors,
    # we need to pad `counts` up to the right length
    counts = torch.nn.functional.pad(
        counts, pad=(0, len(data[AtomicDataDict.POSITIONS_KEY]) - len(counts))
    )
    if not graph_is_undirected:
        counts[counts == 0] = counts[counts > 0].float().mean().int()
    return (counts, "node")


def add_avg_num_neighbors(
    config: Config,
    initialize: bool,
    dataset: Optional[ConcatDataset] = None,
) -> Optional[float]:
    # Compute avg_num_neighbors
    annkey: str = "avg_num_neighbors"
    var_nn_key: str = "var_num_neighbors"
    ann = config.get(annkey, "auto")
    var_nn = config.get(var_nn_key, None)
    if ann == "auto":
        if not initialize:
            raise ValueError("avg_num_neighbors = auto but initialize is False")
        if dataset is None:
            raise ValueError(
                "When avg_num_neighbors = auto, the dataset is required to build+initialize a model"
            )
        cnns, anns, var_nns = [], [], []
        for ds in dataset.datasets:
            cnn, ann, var_nn = ds.statistics(
                fields=[partial(_add_avg_num_neighbors_helper, graph_is_undirected=not config.get("directed_graph", False))],
                modes=["count_mean_std"],
                stride=config.get("dataset_statistics_stride", 1),
            )[0]
            cnns.append(cnn.sum().item())
            anns.append(ann.item())
            var_nns.append(var_nn.item())
        ann = 0.
        for c, a in zip(cnns, anns):
            ann += c * a
        ann /= sum(cnns)
        c1 = cnns[0]
        a1 = anns[0]
        var_nn = var_nns[0]
        for c2, a2, var2 in zip(cnns[1:], anns[1:], var_nns[1:]):
            var_nn = (c1 * var_nn + c2 * var2) / (c1 + c2) + (c1 * c2 * (a1 - a2)**2) / (c1 + c2)**2
            c1 = c2
            a1 = a2


    # make sure its valid
    if ann is not None:
        ann = float(ann)
    config[annkey] = ann
    if var_nn is not None:
        config[var_nn_key] = var_nn
    return ann, var_nn


def _add_avg_num_atoms_helper(data):
    counts = torch.unique(
        data[AtomicDataDict.BATCH_KEY],
        sorted=True,
        return_counts=True,
    )[1]
    return (counts, "graph")


def add_avg_num_atoms(
    config: Config,
    initialize: bool,
    dataset: Optional[AtomicDataset] = None,
) -> Optional[float]:
    # Compute avg_num_atoms
    anakey: str = "avg_num_atoms"
    var_na_key: str = "var_num_atoms"
    ana = config.get(anakey, "auto")
    var_na = config.get(var_na_key, None)
    if ana == "auto":
        if not initialize:
            raise ValueError("avg_num_atoms = auto but initialize is False")
        if dataset is None:
            raise ValueError(
                "When avg_num_atoms = auto, the dataset is required to build+initialize a model"
            )
        ana, var_na = dataset.statistics(
            fields=[_add_avg_num_atoms_helper],
            modes=["mean_std"],
            stride=config.get("dataset_statistics_stride", 1),
        )[0]
        ana = ana.item()
        var_na = var_na.item()

    # make sure its valid
    if ana is not None:
        ana = float(ana)
    config[anakey] = ana
    if var_na is not None:
        config[var_na_key] = var_na
    return ana, var_na
