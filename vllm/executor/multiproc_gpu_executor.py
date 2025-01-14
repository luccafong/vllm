import asyncio
import json
import os
import socket
import time
from collections import defaultdict
from functools import partial
from typing import Any, List, Optional, Tuple

import torch

from torch.distributed.elastic.agent.server.api import _RoleInstanceInfo

from torch.distributed.elastic.rendezvous import RendezvousParameters

from torch.distributed.elastic.rendezvous.utils import parse_rendezvous_endpoint

from vllm.executor.distributed_gpu_executor import (  # yapf: disable
    DistributedGPUExecutor,
    DistributedGPUExecutorAsync,
)
from vllm.executor.gpu_executor import create_worker
from vllm.executor.multiproc_worker_utils import (
    ProcessWorkerWrapper,
    ResultHandler,
    set_multiprocessing_worker_envs,
    WorkerMonitor,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest
from vllm.utils import (
    _run_task_with_lock,
    cuda_device_count_stateless,
    get_distributed_init_method,
    get_open_port,
    make_async,
    update_environment_variables,
)
from vllm.worker.worker_base import RendezvousData


logger = init_logger(__name__)

import torch.distributed.elastic.rendezvous.registry as rdzv_registry


def _get_addr_and_port(
    rdzv_parameters: RendezvousParameters,
) -> Tuple[Optional[str], Optional[int]]:
    if rdzv_parameters.backend != "static":
        return (None, None)
    endpoint = rdzv_parameters.endpoint
    endpoint = endpoint.strip()
    if not endpoint:
        raise ValueError(
            "Endpoint is missing in endpoint. Try to add --master-addr and --master-port"
        )
    master_addr, master_port = parse_rendezvous_endpoint(endpoint, default_port=-1)
    if master_port == -1:
        raise ValueError(
            f"port is missing in endpoint: {endpoint}. Try to specify --master-port"
        )
    return (master_addr, master_port)


class MultiprocessingGPUExecutor(DistributedGPUExecutor):
    """Python multiprocessing-based multi-GPU executor"""

    uses_ray: bool = False
    use_rdzv: bool = True

    def _wait_for_host(self) -> None:
        if self.vllm_config.multi_host_rank > 0:

            def wait_for_leader() -> None:
                if self.vllm_config.pipeline_leader_connect_timeout_secs <= 0:
                    return

                logger.info(
                    f"Will block server until PP leader [{self.vllm_config.leader_host_addr}]:{self.vllm_config.leader_host_port} is up for up to {self.vllm_config.pipeline_leader_connect_timeout_secs} secs"
                )

                end = (
                    time.monotonic()
                    + self.vllm_config.pipeline_leader_connect_timeout_secs
                )
                while time.monotonic() < end:
                    try:
                        sock = socket.create_connection(
                            (
                                self.vllm_config.leader_host_addr,
                                self.vllm_config.leader_host_port,
                            ),
                            timeout=600,
                        )
                        sock.close()
                        return
                    except socket.error:
                        pass
                    time.sleep(5)

                raise Exception(
                    "PP leader is not reachable after timeout. Will exit predictor."
                )

            wait_for_leader()
            logger.info("Successfully opened socket to leader.")
        pass

    def _init_rdzv(
        self, run_id: str, nodes: int, local_world_size: int
    ) -> List[RendezvousData]:
        rdvz_is_host = self.vllm_config.multi_host_rank == 0
        rdzv_parameters = RendezvousParameters(
            backend="c10d",
            endpoint=f"{self.vllm_config.leader_host_addr}:{self.vllm_config.leader_host_port}",
            run_id=run_id,
            min_nodes=nodes,
            max_nodes=nodes,
            is_host=rdvz_is_host,
            rank=self.vllm_config.multi_host_rank,
            join_timeout=1000,
        )
        rdzv_handler = rdzv_registry.get_rendezvous_handler(rdzv_parameters)
        logger.info(f"Rendezvous handler: {type(rdzv_handler)}")
        rdzv_info = rdzv_handler.next_rendezvous()
        logger.info(f"Rendezvous created")

        store = rdzv_info.store
        group_rank = rdzv_info.rank
        group_world_size = rdzv_info.world_size

        master_addr, master_port = _get_addr_and_port(rdzv_parameters)
        master_addr = master_addr or rdzv_info.bootstrap_store_info.master_addr
        master_port = master_port or rdzv_info.bootstrap_store_info.master_port
        role = "default_role"
        if os.environ.get("TORCH_ELASTIC_WORKER_IDENTICAL", "0") == "1":
            global_world_size = group_world_size * local_world_size
            base_global_rank = group_rank * local_world_size
            base_role_rank = base_global_rank
            role_world_size = global_world_size
        else:
            ROLE_INFO_PREFIX = "torchelastic/role_info/"
            ASSIGNED_RANKS_PREFIX = "torchelastic/assigned_ranks/"

            agent_role_info = _RoleInstanceInfo(role, group_rank, local_world_size)
            store.set(f"{ROLE_INFO_PREFIX}{group_rank}", agent_role_info.serialize())
            # tcp store is collocated with rank 0 so we can use it to do extra compute to reduce overall # of operations.
            if group_rank == 0:
                role_infos_bytes = store.multi_get(
                    [f"torchelastic/role_info/{i}" for i in range(group_world_size)]
                )
                role_infos = [
                    _RoleInstanceInfo.deserialize(info_bytes)
                    for info_bytes in role_infos_bytes
                ]

                role_sizes = defaultdict(lambda: 0)
                global_size = 0
                for role_info in role_infos:
                    role_sizes[role_info.role] += role_info.local_world_size
                    global_size += role_info.local_world_size

                base_global_rank = 0
                role_ranks = defaultdict(lambda: 0)

                keys = []
                values = []
                for i, role_info in enumerate(role_infos):
                    keys.append(f"{ASSIGNED_RANKS_PREFIX}{i}")
                    values.append(
                        json.dumps(
                            [
                                base_global_rank,
                                global_size,
                                role_ranks[role_info.role],
                                role_sizes[role_info.role],
                            ]
                        )
                    )

                    base_global_rank += role_info.local_world_size
                    role_ranks[role_info.role] += role_info.local_world_size

                store.multi_set(keys, values)

            # get will block until the data is available in the store.
            (
                base_global_rank,
                global_world_size,
                base_role_rank,
                role_world_size,
            ) = json.loads(store.get(f"{ASSIGNED_RANKS_PREFIX}{group_rank}"))
        self.store = store
        rdzv_worker_data = []
        logger.info(f"Rendezvous store: {store}")
        for local_rank in range(local_world_size):
            rdzv_worker_data.append(
                RendezvousData(
                    local_rank=local_rank,
                    global_rank=base_global_rank + local_rank,
                    role_rank=base_role_rank + local_rank,
                    world_size=global_world_size,
                    role_world_size=role_world_size,
                    local_world_size=local_world_size,
                    max_restarts=3,
                    role=role,
                    group_world_size=group_world_size,
                    master_addr=master_addr,
                    master_port=master_port,
                    run_id=rdzv_handler.get_run_id(),
                    use_agent_store=rdzv_handler.use_agent_store,
                    group_rank=group_rank,
                )
            )
        logger.info(f"Rendezvous data: {rdzv_worker_data}")
        return rdzv_worker_data

        # TODO:
        # https://fburl.com/code/yxxmzybm
        # https://www.internalfb.com/code/fbsource/[c1bd8bfb4dfd]/fbcode/caffe2/torch/distributed/elastic/agent/server/api.py?lines=555
        # Assign Ranks
        # pass

    def _init_executor(self) -> None:

        world_size = self.parallel_config.world_size
        max_local_world_size = torch.cuda.device_count()
        self.local_world_size = min(max_local_world_size, world_size)
        self._check_executor_parameters()

        # Create the parallel GPU workers.
        nnodes = world_size // self.local_world_size
        tensor_parallel_size = self.parallel_config.tensor_parallel_size
        # Set multiprocessing envs that are common to V0 and V1
        set_multiprocessing_worker_envs(self.parallel_config)

        # Multiprocessing-based executor does not support multi-node setting.
        # Since it only works for single node, we can use the loopback address
        # 127.0.0.1 for communication.
        if nnodes > 1:
            distributed_init_method = get_distributed_init_method(
                self.vllm_config.leader_host_addr, get_open_port()
            )
        else:
            distributed_init_method = get_distributed_init_method(
                "127.0.0.1", get_open_port()
            )

        self.workers: List[ProcessWorkerWrapper] = []
        # This is the list of workers that are rank 0 of each TP group EXCEPT
        # global rank 0. These are the workers that will broadcast to the
        # rest of the workers.
        self.tp_driver_workers: List[ProcessWorkerWrapper] = []
        # This is the list of workers that are not drivers and not the first
        # worker in a TP group. These are the workers that will be
        # broadcasted to.
        self.non_driver_workers: List[ProcessWorkerWrapper] = []
        self.store: Optional[torch.distributed.Store] = None
        if self.use_rdzv and nnodes > 1:
            self._wait_for_host()
            self.rdzv_datas = self._init_rdzv(
                self.vllm_config.instance_id, nnodes, self.local_world_size
            )
        else:
            self.rdzv_datas = None
        if world_size == 1:
            self.worker_monitor = None
        else:
            result_handler = ResultHandler()
            if self.use_rdzv and self.vllm_config.multi_host_rank > 0:
                worker_start_rank = 0  # no driver worker if using rdzv and not is host
            else:
                worker_start_rank = 1
            for rank in range(worker_start_rank, self.local_world_size):
                rdzv_data = (
                    self.rdzv_datas[rank] if self.rdzv_datas is not None else None
                )
                worker = ProcessWorkerWrapper(
                    result_handler,
                    partial(
                        create_worker,
                        **self._get_worker_kwargs(
                            rank=(
                                rdzv_data.global_rank if rdzv_data is not None else rank
                            ),
                            local_rank=rank,
                            distributed_init_method=distributed_init_method,
                            rdzv_data=rdzv_data,
                            # store=self.store,
                        ),
                    ),
                )
                self.workers.append(worker)
                if (
                    rank % tensor_parallel_size == 0
                    and self.vllm_config.multi_host_rank == 0
                ):
                    self.tp_driver_workers.append(worker)
                else:
                    self.non_driver_workers.append(worker)

            self.worker_monitor = WorkerMonitor(self.workers, result_handler)
            result_handler.start()
            self.worker_monitor.start()

        # Set up signal handlers to shutdown the executor cleanly
        # sometimes gc does not work well
        if not self.use_rdzv or self.vllm_config.multi_host_rank == 0:
            self.driver_worker = self._create_worker(
                distributed_init_method=distributed_init_method,
                rdzv_data=self.rdzv_datas[0] if self.rdzv_datas else None,
                # store=self.store,
            )
        else:
            self.driver_worker = None
        self._run_workers("init_device")
        self._run_workers(
            "load_model",
            max_concurrent_workers=self.parallel_config.max_parallel_loading_workers,
        )

    def _check_executor_parameters(self):
        world_size = self.local_world_size
        tensor_parallel_size = self.parallel_config.tensor_parallel_size

        assert world_size % self.local_world_size == 0, (
            f"please ensure that world_size ({world_size}) "
            f"is divisible by max local gpu count ({self.local_world_size})"
        )
        # Set CUDA_VISIBLE_DEVICES for the driver, inherited by workers
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            update_environment_variables(
                {"CUDA_VISIBLE_DEVICES": (",".join(map(str, range(world_size))))}
            )

        cuda_device_count = cuda_device_count_stateless()
        # Use confusing message for more common TP-only case.
        assert tensor_parallel_size <= cuda_device_count, (
            f"please set tensor_parallel_size ({tensor_parallel_size}) "
            f"to less than max local gpu count ({cuda_device_count})"
        )

        assert world_size <= cuda_device_count, (
            f"please ensure that world_size ({world_size}) "
            f"is less than than max local gpu count ({cuda_device_count})"
        )

    def shutdown(self):
        if (worker_monitor := getattr(self, "worker_monitor", None)) is not None:
            worker_monitor.close()

    def _driver_execute_model(
        self, execute_model_req: Optional[ExecuteModelRequest]
    ) -> Optional[List[SamplerOutput]]:
        """Run execute_model in the driver worker.

        Passing None will cause the driver to stop the model execution
        loop running in each of the remote workers.
        """
        return self.driver_worker.execute_model(execute_model_req)

    def _run_workers(
        self,
        method: str,
        *args,
        async_run_tensor_parallel_workers_only: bool = False,
        max_concurrent_workers: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers.

        Args:
            async_run_tensor_parallel_workers_only: If True the method will be
                run only in the remote TP workers, not the driver worker.
                It will also be run asynchronously and return a list of futures
                rather than blocking on the results.
        """

        if max_concurrent_workers:
            raise NotImplementedError("max_concurrent_workers is not supported yet.")

        if async_run_tensor_parallel_workers_only:
            # Run only non-driver workers and just return futures.
            return [
                worker.execute_method(method, *args, **kwargs)
                for worker in self.non_driver_workers
            ]

        # Start all remote workers first.
        worker_outputs = [
            worker.execute_method(method, *args, **kwargs) for worker in self.workers
        ]
        if self.driver_worker is None:
            return [output.get() for output in worker_outputs]
        driver_worker_method = getattr(self.driver_worker, method)
        driver_worker_output = driver_worker_method(*args, **kwargs)

        # Get the results of the workers.
        return [driver_worker_output] + [output.get() for output in worker_outputs]

    def check_health(self) -> None:
        """Raises an error if engine is unhealthy."""
        if self.worker_monitor is not None and not self.worker_monitor.is_alive():
            raise RuntimeError("Worker processes are not running")

    def _wait_for_tasks_completion(self, parallel_worker_tasks: Any) -> None:
        """Wait for futures returned from _run_workers() with
        async_run_remote_workers_only to complete."""
        for result in parallel_worker_tasks:
            result.get()


class MultiprocessingGPUExecutorAsync(
    MultiprocessingGPUExecutor, DistributedGPUExecutorAsync
):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.driver_worker is not None:
            self.driver_exec_model = make_async(self.driver_worker.execute_model)
        else:
            self.driver_exec_model = None
        self.pp_locks: Optional[List[asyncio.Lock]] = None

    async def _driver_execute_model_async(
        self, execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> List[SamplerOutput]:
        if self.driver_exec_model is None:
            raise RuntimeError("non-driver worker does not accept model requests")
        if not self.tp_driver_workers:
            return await self.driver_exec_model(execute_model_req)

        if self.pp_locks is None:
            # This locks each pipeline parallel stage so multiple virtual
            # engines can't execute on the same stage at the same time
            # We create the locks here to avoid creating them in the constructor
            # which uses a different asyncio loop.
            self.pp_locks = [
                asyncio.Lock()
                for _ in range(self.parallel_config.pipeline_parallel_size)
            ]

        tasks = [
            asyncio.create_task(
                _run_task_with_lock(
                    self.driver_exec_model, self.pp_locks[0], execute_model_req
                )
            )
        ]
        for pp_rank, driver_worker in enumerate(self.tp_driver_workers, start=1):
            tasks.append(
                asyncio.create_task(
                    _run_task_with_lock(
                        driver_worker.execute_method_async,
                        self.pp_locks[pp_rank],
                        "execute_model",
                        execute_model_req,
                    )
                )
            )
        results = await asyncio.gather(*tasks)

        # Only the last PP stage has the final results.
        return results[-1]

    async def _start_worker_execution_loop(self):
        coros = [
            worker.execute_method_async("start_worker_execution_loop")
            for worker in self.non_driver_workers
        ]
        return await asyncio.gather(*coros)
