import torch
import torch.distributed as dist


def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return -1


def is_main_process() -> bool:
    return get_rank() <= 0


def get_device() -> torch.device:
    rank = get_rank()
    if rank >= 0:
        return torch.device('cuda', rank)
    if dist.is_available():
        return torch.device('cuda')
    return torch.device('cpu')
