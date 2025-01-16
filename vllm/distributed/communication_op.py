from typing import Any, Dict, Optional, Union

import torch
import torch.distributed

from .parallel_state import get_driver_group, get_pp_group, get_tp_group


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_tp_group().all_reduce(input_)


def tensor_model_parallel_all_gather(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    return get_tp_group().all_gather(input_, dim)


def tensor_model_parallel_gather(
    input_: torch.Tensor, dst: int = 0, dim: int = -1
) -> Optional[torch.Tensor]:
    """Gather the input tensor across model parallel group."""
    return get_tp_group().gather(input_, dst, dim)


def broadcast_tensor_dict(
    tensor_dict: Optional[Dict[Any, Union[torch.Tensor, Any]]] = None, src: int = 0
):
    if not torch.distributed.is_initialized():
        return tensor_dict
    return get_tp_group().broadcast_tensor_dict(tensor_dict, src)


def broadcast_pp_object(obj: Optional[Any], src: int = -1) -> Any:
    """Broadcast an object to the pp groups."""
    if not torch.distributed.is_initialized():
        return obj
    if src < 0:
        src = get_pp_group().rank_in_group
    return get_pp_group().broadcast_object(obj, src)


def broadcast_driver_object(obj: Optional[Any], src: int = -1) -> Any:
    """Broadcast an object to the pp groups."""
    if not torch.distributed.is_initialized():
        return obj
    if src < 0:
        src = get_driver_group().rank_in_group
    return get_driver_group().broadcast_object(obj, src)


def get_last_pp_group_rank() -> int:
    """Get the last rank in the pipeline parallel group."""
    return get_pp_group().last_rank_in_group
