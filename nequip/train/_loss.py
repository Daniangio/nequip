import inspect
import logging

import torch.nn
from torch_runstats.scatter import scatter, scatter_mean

from nequip.data import AtomicDataDict
from nequip.utils import instantiate_from_cls_name

from e3nn.o3 import Irreps

class SimpleLoss:
    """wrapper to compute weighted loss function

    Args:

    func_name (str): any loss function defined in torch.nn that
        takes "reduction=none" as init argument, uses prediction tensor,
        and reference tensor for its call functions, and outputs a vector
        with the same shape as pred/ref
    params (str): arguments needed to initialize the function above

    Return:

    if mean is True, return a scalar; else return the error matrix of each entry
    """

    def __init__(self, func_name: str, params: dict = {}):
        self.ignore_nan = params.get("ignore_nan", False)
        self.ignore_zeroes = params.get("ignore_zeroes", False)
        self.normalize_irreps = params.get("normalize_irreps", None)
        if self.normalize_irreps is not None:
            self.normalize_irreps = Irreps(self.normalize_irreps)
        func, _ = instantiate_from_cls_name(
            torch.nn,
            class_name=func_name,
            prefix="",
            positional_args=dict(reduction="none"),
            optional_args=params,
            all_args={},
        )
        self.func_name = func_name
        self.func = func

    def __call__(
        self,
        pred: dict,
        ref: dict,
        key: str,
        mean: bool = True,
    ):
        # zero the nan entries
        ref_key = ref.get(key, torch.zeros_like(pred[key], device=pred[key].device))
        pred_key = torch.nan_to_num(pred[key], nan=0.0)
        if self.normalize_irreps is not None:
            cum_pos = 0
            with torch.no_grad():
                for irrep in self.normalize_irreps:
                    dim = irrep.ir.dim
                    tensor = pred_key[..., cum_pos:cum_pos + dim]
                    norm = tensor.norm(dim=-1, keepdim=True)
                    norm[norm == 0.] += 1.
                    ref_key[..., cum_pos:cum_pos + dim] *= norm
                    cum_pos += dim
        has_nan = self.ignore_nan and torch.isnan(ref_key.sum())
        not_zeroes = torch.ones_like(ref_key).mean(dim=-1).int() if not self.ignore_zeroes else (~torch.all(ref_key == 0., dim=-1)).int()
        if has_nan:
            not_nan = (ref[key] == ref[key]).int() * not_zeroes[..., None]
            loss = self.func(pred_key, torch.nan_to_num(ref_key, nan=0.0)) * not_nan
            if mean:
                return loss.sum() / not_nan.sum()
            else:
                return loss
        else:
            loss = self.func(pred[key], ref_key) * not_zeroes[..., None]
            if mean:
                return loss.mean(dim=-1).sum() / not_zeroes.sum()
            else:
                return loss


class DihedralLoss(SimpleLoss):

    def __call__(
        self,
        pred: dict,
        ref: dict,
        key: str,
        mean: bool = True,
    ):
        # zero the nan entries
        ref_key = ref.get(key, torch.zeros_like(pred[key], device=pred[key].device))
        pred_key = torch.nan_to_num(pred[key], nan=0.0)
        has_nan = self.ignore_nan and torch.isnan(ref_key.sum())
        ref_key = torch.nan_to_num(ref_key, nan=0.0)
        not_zeroes = torch.ones_like(ref_key).mean(dim=-1).int() if not self.ignore_zeroes else (~torch.all(ref_key == 0., dim=-1)).int()
        if has_nan:
            not_nan = (ref[key] == ref[key]).int() * not_zeroes[..., None]
            # loss = (self.func(torch.cos(pred_key), torch.cos(ref_key)) + self.func(torch.sin(pred_key), torch.sin(ref_key))) * not_nan
            loss = (2 + torch.cos(pred_key - ref_key - torch.pi) + torch.sin(pred_key - ref_key - torch.pi/2)) * not_nan
            if mean:
                return loss.sum() / not_nan.sum()
            else:
                return loss
        else:
            # loss = (self.func(torch.cos(pred_key), torch.cos(ref_key)) + self.func(torch.sin(pred_key), torch.sin(ref_key))) * not_zeroes[..., None]
            loss = (2 + torch.cos(pred_key - ref_key - torch.pi) + torch.sin(pred_key - ref_key - torch.pi/2)) * not_zeroes[..., None]
            if mean:
                return loss.mean(dim=-1).sum() / not_zeroes.sum()
            else:
                return loss


class SideChainLoss(SimpleLoss):

    def __call__(
        self,
        pred: dict,
        ref: dict,
        key: str,
        mean: bool = True,
    ):
        # zero the nan entries
        ref_key = ref.get(key, torch.zeros_like(pred[key], device=pred[key].device))
        pred_key = torch.nan_to_num(pred[key], nan=0.0)
        ref_key = torch.nan_to_num(ref_key, nan=0.0)
        not_zeroes = torch.ones_like(ref_key).mean(dim=-1).int() if not self.ignore_zeroes else (~torch.all(ref_key == 0., dim=-1)).int()
        
        loss_rmsd = self.func(pred_key, ref_key)
        loss_bond = 10 * torch.max(
            torch.zeros_like(pred_key[..., 0]),
            torch.abs(torch.norm(pred_key, dim=-1) - torch.norm(ref_key, dim=-1)) - (0.15)
        )

        lvl_idcs_mask = ref['lvl_idcs_mask']
        rel_vec_dist_vec_list_pred, rel_vec_dist_vec_list_ref = [], []
        for idcs_mask in lvl_idcs_mask[1:]:
            for i, mask in enumerate(idcs_mask):
                for comb in torch.combinations(torch.nonzero(mask).flatten(), r=2):
                    pred_comb = pred_key[i, comb]
                    ref_comb = ref_key[i, comb]
                    pred_dist_vec = torch.zeros((1, 3, 3), dtype=pred_comb.dtype, device=pred_comb.device)
                    ref_dist_vec = torch.zeros((1, 3, 3), dtype=ref_comb.dtype, device=ref_comb.device)
                    pred_dist_vec[:, 0::2] = pred_comb
                    ref_dist_vec[:, 0::2] = ref_comb
                    rel_vec_dist_vec_list_pred.append(pred_dist_vec)
                    rel_vec_dist_vec_list_ref.append(ref_dist_vec)
        rel_vec_dist_vec_pred = torch.stack(rel_vec_dist_vec_list_pred, dim=1)
        rel_vec_dist_vec_ref = torch.stack(rel_vec_dist_vec_list_ref, dim=1)
        
        angle_pred = get_angles(rel_vec_dist_vec_pred)[0]
        angle_ref = get_angles(rel_vec_dist_vec_ref)[0]
        angle_not_zeroes = ~torch.isnan(angle_ref)
        angle_pred = angle_pred[angle_not_zeroes]
        angle_ref = angle_ref[angle_not_zeroes]
        loss_angle = 10 * torch.max(
            torch.zeros_like(angle_pred),
            2 - (0.1) + \
            torch.cos(angle_pred - angle_ref - torch.pi) + \
            torch.sin(angle_pred - angle_ref - torch.pi/2)
        )

        if mean:
            # return loss_bond.sum() / not_zeroes.sum() + loss_angle.mean()
            # return loss_angle.mean()
            # return (loss_rmsd.mean(dim=-1).sum() + loss_bond.sum()) / not_zeroes.sum()
            return (loss_rmsd.mean(dim=-1).sum() + loss_bond.sum()) / not_zeroes.sum() + loss_angle.mean()
        else:
            return loss_rmsd


class BatchAverageLoss(SimpleLoss):
    def __call__(
        self,
        pred: dict,
        ref: dict,
        key: str,
        mean: bool = True,
    ):    
        ref_key = ref.get(key, torch.zeros_like(pred[key], device=pred[key].device))
        has_nan = self.ignore_nan and torch.isnan(ref_key.sum())
        N = torch.bincount(ref[AtomicDataDict.BATCH_KEY])
        N = N.reshape((-1, 1))
        if has_nan:
            not_nan = (ref[key] == ref[key]).int()
            loss = (
                self.func((pred[key][torch.nonzero(not_nan, as_tuple=True)]).sum(), (ref_key[torch.nonzero(not_nan, as_tuple=True)]).sum()) / N
            )
            if self.func_name == "MSELoss":
                loss = loss / N
            if mean:
                return loss.sum() / not_nan.sum()
            else:
                return loss
        else:
            loss = self.func(pred[key].sum(), ref_key.sum())
            loss = loss / N
            if self.func_name == "MSELoss":
                loss = loss / N
            if mean:
                return loss.mean()
            else:
                return loss


class PerAtomLoss(SimpleLoss):
    def __call__(
        self,
        pred: dict,
        ref: dict,
        key: str,
        mean: bool = True,
    ):
        # zero the nan entries
        ref_key = ref.get(key, torch.zeros_like(pred[key], device=pred[key].device))
        has_nan = self.ignore_nan and torch.isnan(ref_key.sum())
        N = torch.bincount(ref[AtomicDataDict.BATCH_KEY])
        N = N.reshape((-1, 1))
        if has_nan:
            not_nan = (ref[key] == ref[key]).int()
            loss = (
                self.func(pred[key], torch.nan_to_num(ref_key, nan=0.0)) * not_nan / N
            )
            if self.func_name == "MSELoss":
                loss = loss / N
            assert loss.shape == pred[key].shape  # [atom, dim]
            if mean:
                return loss.sum() / not_nan.sum()
            else:
                return loss
        else:
            loss = self.func(pred[key], ref_key)
            loss = loss / N
            if self.func_name == "MSELoss":
                loss = loss / N
            assert loss.shape == pred[key].shape  # [atom, dim]
            if mean:
                return loss.mean()
            else:
                return loss


class PerSpeciesLoss(SimpleLoss):
    """Compute loss for each species and average among the same species
    before summing them up.

    Args same as SimpleLoss
    """

    def __call__(
        self,
        pred: dict,
        ref: dict,
        key: str,
        mean: bool = True,
    ):
        if not mean:
            raise NotImplementedError("Cannot handle this yet")

        ref_key = ref.get(key, torch.zeros_like(pred[key], device=pred[key].device))
        has_nan = self.ignore_nan and torch.isnan(ref_key.sum())

        if has_nan:
            not_nan = (ref_key == ref_key).int()
            per_atom_loss = (
                self.func(pred[key], torch.nan_to_num(ref_key, nan=0.0)) * not_nan
            )
        else:
            per_atom_loss = self.func(pred[key], ref_key)

        reduce_dims = tuple(i + 1 for i in range(len(per_atom_loss.shape) - 1))

        spe_idx = pred[AtomicDataDict.ATOM_TYPE_KEY].squeeze(-1)
        if has_nan:
            if len(reduce_dims) > 0:
                per_atom_loss = per_atom_loss.sum(dim=reduce_dims)
            assert per_atom_loss.ndim == 1

            per_species_loss = scatter(per_atom_loss, spe_idx, dim=0)

            assert per_species_loss.ndim == 1  # [type]

            N = scatter(not_nan, spe_idx, dim=0)
            N = N.sum(reduce_dims)
            N = N.reciprocal()
            N_species = ((N == N).int()).sum()
            assert N.ndim == 1  # [type]

            per_species_loss = (per_species_loss * N).sum() / N_species

            return per_species_loss

        else:

            if len(reduce_dims) > 0:
                per_atom_loss = per_atom_loss.mean(dim=reduce_dims)
            assert per_atom_loss.ndim == 1

            # offset species index by 1 to use 0 for nan
            _, inverse_species_index = torch.unique(spe_idx, return_inverse=True)

            per_species_loss = scatter_mean(per_atom_loss, inverse_species_index, dim=0)
            assert per_species_loss.ndim == 1  # [type]

            return per_species_loss.mean()


def find_loss_function(name: str, params):
    """
    Search for loss functions in this module

    If the name starts with PerSpecies, return the PerSpeciesLoss instance
    """

    wrapper_list = dict(
        perspecies=PerSpeciesLoss,
        peratom=PerAtomLoss,
        batchaverage=BatchAverageLoss,
        dih=DihedralLoss,
        sc=SideChainLoss,
    )

    if isinstance(name, str):
        for key in wrapper_list:
            if name.lower().startswith(key):
                logging.debug(f"create loss instance {wrapper_list[key]}")
                return wrapper_list[key](name[len(key) :], params)
        return SimpleLoss(name, params)
    elif inspect.isclass(name):
        return SimpleLoss(name, params)
    elif callable(name):
        return name
    else:
        raise NotImplementedError(f"{name} Loss is not implemented")


def get_angles_from_vectors(b0: torch.Tensor, b1: torch.Tensor, return_cos: bool = False) -> torch.Tensor:
    b0n = torch.norm(b0, dim=2, keepdim=False)
    b1n = torch.norm(b1, dim=2, keepdim=False)
    angles = torch.sum(b0 * b1, axis=-1) / b0n / b1n
    clamped_cos = torch.clamp(angles, min=-1., max=1.)
    if return_cos:
        return clamped_cos
    return torch.arccos(clamped_cos)

def get_angles(dist_vectors: torch.Tensor) -> torch.Tensor:
    """ Compute angle values (in radiants) over specified angle_idcs for every frame in the batch

        :param dist_vectors: torch.Tensor | shape (batch, n_angles, 3, xyz)
        :return:             torch.Tensor | shape (batch, n_angles)
    """

    b0 = -1.0 * (dist_vectors[:, :, 1] - dist_vectors[:, :, 0])
    b1 = (dist_vectors[:, :, 2] - dist_vectors[:, :, 1])
    return get_angles_from_vectors(b0, b1)