"""Distributed training and multiprocessing helpers."""
from datetime import timedelta
from functools import partial
from types import FunctionType
from typing import Optional, Collection, Any, Dict, List, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm

_pool: Optional[mp.Pool] = None


def get_rank() -> int:
    """
    Get the rank of current device.

    **Return**: Rank of current device if distributed training on GPU is running. Otherwise, return -1.
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return -1


def is_main_process() -> bool:
    """
    Check whether the current process is the main process.
    In the case of non-distributed training, this result is always `True`.

    **Return**: Boolean indicating whether the current process is the main process.
    """
    return get_rank() <= 0


def get_device() -> torch.device:
    """
    Get current device.

    **Return**: Current device.
    """
    rank = get_rank()
    if rank >= 0:
        return torch.device('cuda', rank)
    if dist.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def barrier():
    """
    Wait for other processes if distributed training is running.
    """
    if get_rank() >= 0:
        dist.barrier()


def init_process_group(distributed: bool = True, num_processes: int = 200):
    """
    Initialize process group and multiprocessing pool.

    **Args**:

    - `distributed` (`bool`): Whether to run distributed training if available. Default is `True`.
    - `num_processes` (`int`): Number of processes to run on the pool.
    """
    global _pool
    if torch.cuda.is_available() and distributed:
        dist.init_process_group(backend='nccl', init_method='env://', timeout=timedelta(days=1))
    _pool = mp.Pool(processes=num_processes)


def fast_filter(func: FunctionType, iterable: Collection[Any], **kwargs) -> List[Any]:
    """
    Filter an iterable by some criterion.

    **Args**:

    - `func` (`FunctionType`): The predicate function.
      The first argument of the function should be a single element in the iterable.
    - `iterable` (`Iterable[Any]`): The source iterable elements.
    - `kwargs`: Other arguments to the predicate other than the element.

    **Return**: The filtered iterables.
    """
    indicator = fast_map(func, iterable, func_kwargs=kwargs)
    filtered = [
        ele for ind, ele
        in zip(indicator, iterable) if ind
    ]
    return filtered


def fast_map(func: FunctionType, iterable: Collection[Any], verbose_descr: Optional[str] = None,
             filter_input: Optional[FunctionType] = None, filter_output: Optional[FunctionType] = None,
             func_kwargs: Optional[Dict[str, Any]] = None, input_kwargs: Optional[Dict[str, Any]] = None,
             output_kwargs: Optional[Dict[str, Any]] = None) \
        -> List[Any]:
    """
    Map or execute some process over all elements of an iterable by some function.

    **Args**:

    - `func` (`FunctionType`): The function to execute.
      The first argument of the function should be a single element in the iterable.
    - `iterable` (`Iterable[Any]`): The source iterable elements.
    - `verbose_descr` (`Optional[str]`): Verbose description of the execution process.
      Not verbosed if `None` given.
    - `filter_input` (`Optional[FunctionType]`): Boolean function that checks whether an element in the iterable
      is to be processed. If not provided, everything is retained. Returning `True` means remained.
    - `filter_output` (`Optional[FunctionType]`): Boolean function that checks whether a processed value need to be
      returned. If not provided, everything is retained. Returning `True` means remained.
    - `func_kwargs` (`Optional[Dict[str, Any]]`): Other arguments besides the content of iterable to the function
      of processing.
    - `input_kwargs` (`Optional[Dict[str, Any]]`): Other arguments besides the content of iterable to the input
      filter predicate.
    - `output_kwargs` (`Optional[Dict[str, Any]]`): Other arguments besides the content of iterable to the output
      filter predicate.

    **Return**: The returning iterables.
    """
    func_kwargs = {} if func_kwargs is None else func_kwargs
    input_kwargs = {} if input_kwargs is None else input_kwargs
    output_kwargs = {} if output_kwargs is None else output_kwargs
    func = partial(func, **func_kwargs)
    if filter_input is not None:
        iterable = fast_filter(filter_input, **input_kwargs)
    length = len(iterable)
    result = [None] * length

    if _pool is not None:
        iterable = _pool.imap(func, iterable)
    if verbose_descr is not None:
        iterable = tqdm(iterable, total=len(iterable))
        iterable.set_description(verbose_descr)

    if _pool is None:
        for i, ele in enumerate(iterable):
            result[i] = func(ele)
    else:
        for i, ele_res in enumerate(iterable):
            result[i] = ele_res

    if filter_output is not None:
        result = fast_filter(filter_output, result, func_kwargs=output_kwargs)
    return result


def fast_map_dict(func: FunctionType, dictionary: Dict, verbose_descr: Optional[str] = None,
                  key_mapper: Optional[FunctionType] = None, key_filter: Optional[FunctionType] = None,
                  input_filter: Optional[FunctionType] = None, output_value_filter: Optional[FunctionType] = None,
                  func_kwargs: Optional[Dict[str, Any]] = None, mapper_kwargs: Optional[Dict[str, Any]] = None,
                  key_filter_kwargs: Optional[Dict[str, Any]] = None,
                  input_filter_kwargs: Optional[Dict[str, Any]] = None,
                  output_value_filter_kwargs: Optional[Dict[str, Any]] = None) -> Dict:
    """
    Map or execute some process over all elements of a dictionary by some function.

    **Args**:

    - `func` (`FunctionType`): The function to execute.
      The first argument of the function should be a key in the dictionary.
      The second argument of the function should be the corresponding value of the key in the dictionary.
    - `dictionary` (`Dict`): The source dictionary.
    - `verbose_descr` (`Optional[str]`): Verbose description of the execution process.
      Not verbosed if `None` given.
    - `key_mapper` (`Optional[FunctionType]`): Map keys to other format in the output. Remain the same if not provided.
      The first argument should be a key.
    - `key_filter` (`Optional[FunctionType]`): Filter original keys to other format in the output.
      Similar as `input_filter` on keys for `fast_map`.
    - `input_filter` (`Optional[Function]`): Similar as `input_filter` but it takes first argument as the key and
      second argument as the value.
    - `output_value_filter` (`Optional[Function]`): Similar as `output_filter` on dictionary values.
    - `func_kwargs` (`Optional[Dict[str, Any]]`): Other arguments besides the key and value of dictionary to the
      function of processing.
    - `mapper_kwargs` (`Optional[Dict[str, Any]]`): Other arguments besides the keys to the key mapper.
    - `key_filter_kwargs` (`Optional[Dict[str, Any]]`): Other arguments besides the keys to the key filter.
    - `input_filter_kwargs` (`Optional[Dict[str, Any]]`): Other arguments besides the keys and values to the input
      filter.
    - `output_value_filter_kwargs` (`Optional[Dict[str, Any]]`): Other arguments besides the keys and values to the
      output value filter.

    **Return**: The returning iterables.
    """
    keys, values = dictionary.keys(), dictionary.values()

    if key_filter is not None:
        def kv_key_filter(kv, **kwargs):
            k, v = kv
            return key_filter(k, **kwargs)
        filtered = fast_filter(kv_key_filter, list(zip(keys, values)), **key_filter_kwargs)
        keys, values = _separate_kv(filtered)

    if input_filter is not None:
        def kv_input_filter(kv, **kwargs):
            k, v = kv
            return input_filter(k, v, **kwargs)
        filtered = fast_filter(kv_input_filter, list(zip(keys, values)), **input_filter_kwargs)
        keys, values = _separate_kv(filtered)

    def kv_func(kv, **kwargs):
        k, v = kv
        return func(k, v, **kwargs)

    iter_result = fast_map(kv_func, list(zip(keys, values)), verbose_descr, func_kwargs=func_kwargs)

    if output_value_filter is not None:
        def kv_output_filter(kv, **kwargs):
            k, v = kv
            return output_value_filter(v, **kwargs)
        filtered = fast_filter(kv_output_filter, list(zip(keys, iter_result)), **output_value_filter_kwargs)
        keys, iter_result = _separate_kv(filtered)

    result = {}
    if key_mapper is not None:
        keys = fast_map(key_mapper, list(keys), func_kwargs=mapper_kwargs)
    for i, key in enumerate(keys):
        result[key] = iter_result[i]
    return result


def _separate_kv(iterable: Collection[Tuple[Any, Any]]) -> Tuple[List[Any], List[Any]]:
    def extract_keys(kv):
        k, v = kv
        return k

    def extract_values(kv):
        k, v = kv
        return v
    keys = fast_map(extract_keys, iterable)
    values = fast_map(extract_values, iterable)
    return keys, values
