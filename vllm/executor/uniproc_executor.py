# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from concurrent.futures import Future

from vllm.distributed.parallel_state import get_pp_group
import vllm.envs as envs
from vllm.executor.executor_base import ExecutorBase
from vllm.logger import init_logger
from vllm.utils import (get_distributed_init_method, get_ip, get_open_port,
                        run_method)
from vllm.worker.worker_base import WorkerWrapperBase

logger = init_logger(__name__)


class UniProcExecutor(ExecutorBase):

    uses_ray: bool = False

    def _init_executor(self) -> None:
        """Initialize the worker and load the model.
        """
        self.driver_worker = WorkerWrapperBase(vllm_config=self.vllm_config,
                                               rpc_rank=0)
        distributed_init_method = get_distributed_init_method(
            get_ip(), get_open_port())
        local_rank = 0
        # set local rank as the device index if specified
        device_info = self.vllm_config.device_config.device.__str__().split(
            ":")
        if len(device_info) > 1:
            local_rank = int(device_info[1])
        rank = 0
        is_driver_worker = True
        kwargs = dict(
            vllm_config=self.vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )
        self.collective_rpc("init_worker", args=([kwargs], ))
        self.collective_rpc("init_device")
        self.collective_rpc("load_model")

    def collective_rpc(self,
                       method: Union[str, Callable],
                       timeout: Optional[float] = None,
                       args: Tuple = (),
                       kwargs: Optional[Dict] = None) -> List[Any]:
        if kwargs is None:
            kwargs = {}
        answer = run_method(self.driver_worker, method, args, kwargs)
        return [answer]

    def check_health(self) -> None:
        # UniProcExecutor will always be healthy as long as
        # it's running.
        return


UniProcExecutorAsync = UniProcExecutor


class ExecutorWithExternalLauncher(UniProcExecutor):
    """An executor that uses external launchers to launch engines,
    specially designed for torchrun-compatible launchers, for
    offline inference with tensor parallelism.

    see https://github.com/vllm-project/vllm/issues/11400 for
    the motivation, and examples/offline_inference/torchrun_example.py
    for the usage example.

    The key idea: although it is tensor-parallel inference, we only
    create one worker per executor, users will launch multiple
    engines with torchrun-compatible launchers, and all these engines
    work together to process the same prompts. When scheduling is
    deterministic, all the engines will generate the same outputs,
    and they don't need to synchronize the states with each other.
    """
    uses_ray: bool = False

    def _init_executor(self) -> None:
        """Initialize the worker and load the model.
        """
        assert self.vllm_config.scheduler_config.delay_factor == 0.0, \
            ("ExecutorWithExternalLauncher needs deterministic "
            "execution, so it"
            "does not support delay_factor in scheduling")
        if envs.VLLM_USE_V1:
            assert not envs.VLLM_ENABLE_V1_MULTIPROCESSING, \
            ("To get deterministic execution in V1, "
            "please set VLLM_ENABLE_V1_MULTIPROCESSING=0")
        self.driver_worker = WorkerWrapperBase(vllm_config=self.vllm_config,
                                               rpc_rank=0)
        # engines are launched in torchrun-compatible launchers
        # so we can use the env:// method.
        # required env vars:
        # - RANK
        # - LOCAL_RANK
        # - MASTER_ADDR
        # - MASTER_PORT
        distributed_init_method = "env://"
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        self.local_rank = local_rank
        is_driver_worker = True
        kwargs = dict(
            vllm_config=self.vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )
        self.collective_rpc("init_worker", args=([kwargs], ))
        self.collective_rpc("init_device")
        self.collective_rpc("load_model")

    def torch_future_to_concurrent_future(
        self, torch_future: torch.futures.Future,
        timeout: Optional[float] = None
    ) -> Future:
        """
        Convert a PyTorch distributed Future to a concurrent.futures.Future.
        
        Args:
            torch_future: The PyTorch Future to convert
            timeout: Optional timeout for the wait operation
            
        Returns:
            concurrent.futures.Future that will complete when the PyTorch Future completes
        """
        concurrent_future = Future()
        
        def _complete_torch_future(fut: torch.futures.Future):
            try:
                # Get the result from the PyTorch future
                result = fut.wait(timeout)
                if not concurrent_future.done():
                    concurrent_future.set_result(result)
            except Exception as e:
                if not concurrent_future.done():
                    concurrent_future.set_exception(e)
        
        # Register our callback to be called when the PyTorch future completes
        torch_future.add_done_callback(_complete_torch_future)
        
        return concurrent_future
    def collective_rpc(self,
                       method: Union[str, Callable],
                       timeout: Optional[float] = None,
                       args: Tuple = (),
                       kwargs: Optional[Dict] = None) -> List[Any]:
        if kwargs is None:
            kwargs = {}
        answer = run_method(self.driver_worker, method, args, kwargs)
        if method == "execute_model":
            pp_size = self.vllm_config.parallel_config.pipeline_parallel_size
            tp_size = self.vllm_config.parallel_config.tensor_parallel_size
            if pp_size > 1:
                # get the answer from the last PP rank
                if answer is None and self.local_rank // tp_size == 0: 
                    # we need to async receive last PP ranks
                    answer = get_pp_group().recv_object(pp_size-1, is_async=True)
                    answer = self.torch_future_to_concurrent_future(answer)
                if answer is not None and self.local_rank // tp_size == pp_size-1:
                    # we need to async send to the first PP rank
                    get_pp_group().send_object(answer, 0, is_async=True)
            
            return [answer]


        return [answer]
    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """
        Determine the number of available KV blocks.
        Add an additional all_reduce to get the min across all ranks.
        Note that even if we have the same `gpu_memory_utilization` and 
        `swap_space`, the available memory in every rank might still 
        differ because NCCL can take different amounts of memory in 
        different ranks. Therefore, it is necessary to test if all ranks 
        agree on the same KV cache configuration.
        """
        a, b = super().determine_num_available_blocks()
        from vllm.distributed.parallel_state import get_world_group
        cpu_group = get_world_group().cpu_group
        a_tensor = torch.tensor([a], device="cpu", dtype=torch.int64)
        b_tensor = torch.tensor([b], device="cpu", dtype=torch.int64)
        dist.all_reduce(a_tensor, group=cpu_group, op=dist.ReduceOp.MIN)
        dist.all_reduce(b_tensor, group=cpu_group, op=dist.ReduceOp.MIN)
        return a_tensor.item(), b_tensor.item()
