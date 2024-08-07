from typing import List, Optional
import torch
import torch.nn.functional

from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode

from nequip.data import AtomicDataDict
from .._graph_mixin import GraphModuleMixin


@compile_mode("script")
class OneHotAtomEncoding(GraphModuleMixin, torch.nn.Module):
    """Copmute a one-hot floating point encoding of atoms' discrete atom types.

    Args:
        set_features: If ``True`` (default), ``node_features`` will be set in addition to ``node_attrs``.
    """

    num_types: int
    set_features: bool

    # node_input_features: List[str]
    # has_node_input_features: bool

    def __init__(
        self,
        num_types: int,
        set_features: bool = True,
        irreps_in=None,
        # node_input_features: List[str] = [],
    ):
        super().__init__()
        self.num_types = num_types
        self.set_features = set_features

        #self.node_input_features = node_input_features
        #self.has_node_input_features = len(self.node_input_features) > 0
        # Output irreps are num_types even (invariant) scalars
        irreps_out = {AtomicDataDict.NODE_ATTRS_KEY: Irreps([(self.num_types, (0, 1))])}
        #irreps_out = {AtomicDataDict.NODE_ATTRS_KEY: Irreps([(self.num_types + len(node_input_features), (0, 1))])}
        if self.set_features:
            irreps_out[AtomicDataDict.NODE_FEATURES_KEY] = irreps_out[
                AtomicDataDict.NODE_ATTRS_KEY
            ]
        self._init_irreps(irreps_in=irreps_in, irreps_out=irreps_out)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        type_numbers = data.get(AtomicDataDict.ATOM_TYPE_KEY, data["node_types"]).squeeze(-1)
        one_hot = torch.nn.functional.one_hot(
            type_numbers, num_classes=self.num_types
        ).to(device=type_numbers.device, dtype=data[AtomicDataDict.POSITIONS_KEY].dtype)

        # if self.has_node_input_features:
        #     for node_input_feature in self.node_input_features:
        #         one_hot = torch.cat(
        #             [
        #                 one_hot,
        #                 data[node_input_feature][:, None]
        #             ],
        #             dim=1,
        #         )

        data[AtomicDataDict.NODE_ATTRS_KEY] = one_hot
        if self.set_features:
            data[AtomicDataDict.NODE_FEATURES_KEY] = one_hot
        return data
