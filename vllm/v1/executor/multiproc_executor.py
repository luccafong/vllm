# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import multiprocessing
import os
import pickle
import signal
import threading
import time
import traceback
import weakref
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import auto, Enum
from functools import partial
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from threading import Thread
from typing import Any, Callable, cast, Optional, Union

import cloudpickle

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import destroy_distributed_environment, destroy_model_parallel
from vllm.distributed.device_communicators.shm_broadcast import Handle, MessageQueue
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.distributed.parallel_state import get_world_group
from vllm.executor.multiproc_worker_utils import set_multiprocessing_worker_envs
from vllm.logger import init_logger
from vllm.utils import (
    decorate_logs,
    get_distributed_init_method,
    get_loopback_ip,
    get_mp_context,
    get_open_port,
    set_process_title,
)
from vllm.v1.executor.abstract import Executor, FailureCallback
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.worker.worker_base import WorkerWrapperBase

logger = init_logger(__name__)


class MultiprocExecutor(Executor):

    supports_pp: bool = True

    def _init_executor(self) -> None:
        # Call self.shutdown at exit to clean up
        # and ensure workers will be terminated.
        self._finalizer = weakref.finalize(self, self.shutdown)
        self.is_failed = False
        self.shutdown_event = threading.Event()
        self.failure_callback: Optional[FailureCallback] = None
        self.io_thread_pool: Optional[ThreadPoolExecutor] = None

        self.world_size = self.parallel_config.world_size
        assert self.parallel_config.world_size % self.parallel_config.distributed_node_size == 0, (
            f"world_size ({self.parallel_config.world_size}) must be divisible by "
            f"distributed_node_size ({self.parallel_config.distributed_node_size}). ")
        self.local_world_size = self.parallel_config.world_size // self.parallel_config.distributed_node_size
        tensor_parallel_size = self.parallel_config.tensor_parallel_size
        pp_parallel_size = self.parallel_config.pipeline_parallel_size
        assert self.world_size == tensor_parallel_size * pp_parallel_size, (
            f"world_size ({self.world_size}) must be equal to the "
            f"tensor_parallel_size ({tensor_parallel_size}) x pipeline"
            f"_parallel_size ({pp_parallel_size}). ")

        # Set multiprocessing envs that are common to V0 and V1
        set_multiprocessing_worker_envs(self.parallel_config)

        # Multiprocessing-based executor does not support multi-node setting.
        # Since it only works for single node, we can use the loopback address
        # get_loopback_ip() for communication.
        distributed_init_method = get_distributed_init_method(
            get_loopback_ip(), get_open_port())

        # Initialize worker and set up message queues for SchedulerOutputs
        # and ModelRunnerOutputs
        max_chunk_bytes = envs.VLLM_MQ_MAX_CHUNK_BYTES_MB * 1024 * 1024

        if self.parallel_config.distributed_node_rank == 0:
            self.rpc_broadcast_mq = MessageQueue(
                self.world_size,self.local_world_size,
                max_chunk_bytes=max_chunk_bytes,
                connect_ip = self.parallel_config.distributed_master_ip,
            )
            scheduler_output_handle = self.rpc_broadcast_mq.export_handle()
        else:
            # it will be synced later
            self.rpc_broadcast_mq = None
            scheduler_output_handle = None

        # Create workers
        unready_workers: list[UnreadyWorkerProcHandle] = []
        success = False
        try:
            for local_rank in range(self.local_world_size):
                global_rank = self.local_world_size * self.parallel_config.distributed_node_rank + local_rank
                unready_workers.append(
                    WorkerProc.make_worker_process(
                        vllm_config=self.vllm_config,
                        local_rank=local_rank,
                        rank=global_rank,
                        # we read distributed_init_method from parallel_config
                        distributed_init_method=distributed_init_method,
                        input_shm_handle=scheduler_output_handle,
                    ))

            # Workers must be created before wait_for_ready to avoid
            # deadlock, since worker.init_device() does a device sync.
            self.workers = WorkerProc.wait_for_ready(unready_workers)
            logger.info("workers are ready")

            # Ensure message queues are ready. Will deadlock if re-ordered
            # Must be kept consistent with the WorkerProc.
            if self.rpc_broadcast_mq is not None:
                self.rpc_broadcast_mq.wait_until_ready()
            for w in self.workers:
                # local workers
                if w.worker_response_mq is not None:
                    w.worker_response_mq.wait_until_ready()
            # remote mqs
            for rpc_response_mq in self.workers[0].rpc_response_mqs:
                if rpc_response_mq is not None:
                    rpc_response_mq.wait_until_ready()
            self.start_worker_monitor()
            success = True
        finally:
            if not success:
                # Clean up the worker procs if there was a failure.
                # Close death_writers first to signal workers to exit
                for uw in unready_workers:
                    if uw.death_writer is not None:
                        uw.death_writer.close()
                self._ensure_worker_termination(
                    [uw.proc for uw in unready_workers])

        # For pipeline parallel, we use a thread pool for asynchronous
        # execute_model.
        if self.max_concurrent_batches > 1:
            # Note: must use only 1 IO thread to keep dequeue sequence
            # from the response queue
            # _async_aggregate_workers_output also assumes a single IO thread
            self.io_thread_pool = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="mp_exec_io")

        self.output_rank = self._get_output_rank()
        self.has_connector = self.vllm_config.kv_transfer_config is not None
        self.kv_output_aggregator = KVOutputAggregator(
            self.parallel_config.world_size)
        logger.info("MultiprocExecutor initialized.")

    def start_worker_monitor(self):
        workers = self.workers
        self_ref = weakref.ref(self)

        # Monitors worker process liveness. If any die unexpectedly,
        # logs an error, shuts down the executor and invokes the failure
        # callback to inform the engine.
        def monitor_workers():
            sentinels = [h.proc.sentinel for h in workers]
            died = multiprocessing.connection.wait(sentinels)
            _self = self_ref()
            if not _self or getattr(_self, 'shutting_down', False):
                return
            _self.is_failed = True
            proc_name = next(h.proc.name for h in workers
                             if h.proc.sentinel == died[0])
            logger.error(
                "Worker proc %s died unexpectedly, "
                "shutting down executor.", proc_name)
            _self.shutdown()
            callback = _self.failure_callback
            if callback is not None:
                _self.failure_callback = None
                callback()

        Thread(target=monitor_workers,
               daemon=True,
               name="MultiprocWorkerMonitor").start()

    def register_failure_callback(self, callback: FailureCallback):
        if self.is_failed:
            callback()
        else:
            self.failure_callback = callback

    def execute_model(
        self,
        scheduler_output,
    ) -> Union[ModelRunnerOutput, Future[ModelRunnerOutput]]:
        non_block = self.max_concurrent_batches > 1

        if not self.has_connector:
            # get output only from a single worker (output_rank)
            (output, ) = self.collective_rpc(
                "execute_model",
                args=(scheduler_output, ),
                unique_reply_rank=self.output_rank,
                non_block=non_block,
                timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS)
            return output

        # get output from all workers
        outputs = self.collective_rpc(
            "execute_model",
            args=(scheduler_output, ),
            non_block=non_block,
            timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS)

        # aggregate all workers output to a single output
        if non_block:
            return self.kv_output_aggregator.async_aggregate(
                outputs, self.output_rank)
        return self.kv_output_aggregator.aggregate(outputs, self.output_rank)

    def take_draft_token_ids(self) -> Optional[DraftTokenIds]:
        # OPTIMIZATION: Get output only from a single worker (output_rank)
        outputs = self.collective_rpc("take_draft_token_ids",
                                      unique_reply_rank=self.output_rank)
        return outputs[0]

    def collective_rpc(self,
                       method: Union[str, Callable],
                       timeout: Optional[float] = None,
                       args: tuple = (),
                       kwargs: Optional[dict] = None,
                       non_block: bool = False,
                       unique_reply_rank: Optional[int] = None) -> list[Any]:
        if self.is_failed:
            raise RuntimeError("Executor failed.")

        deadline = None if timeout is None else time.monotonic() + timeout
        kwargs = kwargs or {}

        # NOTE: If the args are heterogeneous, then we pack them into a list,
        # and unpack them in the method of every worker, because every worker
        # knows their own rank.
        try:
            if isinstance(method, str):
                send_method = method
            else:
                send_method = cloudpickle.dumps(
                    method, protocol=pickle.HIGHEST_PROTOCOL)
            self.rpc_broadcast_mq.enqueue(
                (send_method, args, kwargs, unique_reply_rank))
            message_queues = []
            for rank in range(self.world_size):
                if rank < self.local_world_size:
                    message_queues.append(self.workers[rank].worker_response_mq)
                else:
                    message_queues.append(self.workers[0].rpc_response_mqs[rank])
            if unique_reply_rank is not None:
                message_queues = (message_queues[unique_reply_rank],)
            responses = []

            def get_response(mq: MessageQueue,
                             dequeue_timeout: Optional[float] = None,
                             cancel_event: Optional[threading.Event] = None):
                status, result = mq.dequeue(
                    timeout=dequeue_timeout, cancel=cancel_event)

                if status != WorkerProc.ResponseStatus.SUCCESS:
                    raise RuntimeError(
                        f"Worker failed with error '{result}', please check the"
                        " stack trace above for the root cause")
                return result

            for mq in message_queues:
                dequeue_timeout = None if deadline is None else (
                    deadline - time.monotonic())

                if non_block:
                    result = self.io_thread_pool.submit(  # type: ignore
                        get_response, mq, dequeue_timeout, self.shutdown_event)
                else:
                    result = get_response(mq, dequeue_timeout)

                responses.append(result)

            return responses
        except TimeoutError as e:
            raise TimeoutError(f"RPC call to {method} timed out.") from e

    @staticmethod
    def _ensure_worker_termination(worker_procs: list[BaseProcess]):
        """Ensure that all worker processes are terminated. Assumes workers have
        received termination requests. Waits for processing, then sends
        termination and kill signals if needed."""

        def wait_for_termination(procs, timeout):
            if not time:
                # If we are in late stage shutdown, the interpreter may replace
                # `time` with `None`.
                return all(not proc.is_alive() for proc in procs)
            start_time = time.time()
            while time.time() - start_time < timeout:
                if all(not proc.is_alive() for proc in procs):
                    return True
                time.sleep(0.1)
            return False

        # Send SIGTERM if still running
        active_procs = [proc for proc in worker_procs if proc.is_alive()]
        for p in active_procs:
            p.terminate()
        if not wait_for_termination(active_procs, 4):
            # Send SIGKILL if still running
            active_procs = [p for p in active_procs if p.is_alive()]
            for p in active_procs:
                p.kill()

    def shutdown(self):
        """Properly shut down the executor and its workers"""
        if not getattr(self, 'shutting_down', False):
            self.shutting_down = True
            self.shutdown_event.set()

            if self.io_thread_pool is not None:
                self.io_thread_pool.shutdown(wait=False, cancel_futures=True)
                self.io_thread_pool = None

            if workers := getattr(self, 'workers', None):
                for w in workers:
                    # Close death_writer to signal child processes to exit
                    if w.death_writer is not None:
                        w.death_writer.close()
                        w.death_writer = None
                    w.worker_response_mq = None
                self._ensure_worker_termination([w.proc for w in workers])

        self.rpc_broadcast_mq = None

    def check_health(self) -> None:
        self.collective_rpc("check_health", timeout=10)
        return

    @property
    def max_concurrent_batches(self) -> int:
        if self.scheduler_config.async_scheduling:
            return 2
        return self.parallel_config.pipeline_parallel_size

    def _get_output_rank(self) -> int:
        # Only returns ModelRunnerOutput from TP rank=0 and PP rank=-1
        # (the first TP worker of the last PP stage).
        # Example:
        # Assuming TP=8, PP=4, then the world_size=32
        # 0-7, PP rank 0
        # 8-15, PP rank 1
        # 16-23, PP rank 2
        # 24-31, PP rank 3
        # so world_size - tp_size = 32 - 8 = 24 should be PP rank = -1 (i.e. 3)
        return self.world_size - self.parallel_config.tensor_parallel_size


@dataclass
class UnreadyWorkerProcHandle:
    """WorkerProcess handle before READY."""
    proc: BaseProcess
    rank: int
    ready_pipe: Connection
    death_writer: Optional[Connection] = None


@dataclass
class WorkerProcHandle:
    proc: BaseProcess
    rank: int
    worker_response_mq: Optional[MessageQueue]  # The worker process writes to this MQ
    rpc_response_mqs: list[Optional[MessageQueue]] = field(default_factory=list)
    # The worker process that will receive remote writes to this MQ
    death_writer: Optional[Connection] = None

    @classmethod
    def from_unready_handle(
            cls, unready_handle: UnreadyWorkerProcHandle,
            worker_response_mq: Optional[MessageQueue],
            rpc_response_mqs: list[Optional[MessageQueue]]) -> "WorkerProcHandle":
        return cls(
            proc=unready_handle.proc,
            rank=unready_handle.rank,
            worker_response_mq=worker_response_mq,
            rpc_response_mqs=rpc_response_mqs,
            death_writer=unready_handle.death_writer,
        )


class WorkerProc:
    """Wrapper that runs one Worker in a separate process."""

    READY_STR = "READY"

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle: Handle,
    ):
        self.rank = rank
        wrapper = WorkerWrapperBase(vllm_config=vllm_config, rpc_rank=local_rank, global_rank=rank)
        # TODO: move `init_worker` to executor level as a collective rpc call
        all_kwargs: list[dict] = [
            {} for _ in range(vllm_config.parallel_config.world_size)
        ]
        is_driver_worker = (
            rank % vllm_config.parallel_config.tensor_parallel_size == 0)
        all_kwargs[local_rank] = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "is_driver_worker": is_driver_worker,
        }
        wrapper.init_worker(all_kwargs)
        self.worker = wrapper

        node_size = vllm_config.parallel_config.distributed_node_size
        pp_size = vllm_config.parallel_config.pipeline_parallel_size
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        pp_str = f"PP{rank // tp_size}" if pp_size > 1 else ""
        tp_str = f"TP{rank % tp_size}" if tp_size > 1 else ""
        suffix = f"{pp_str}{'_' if pp_str and tp_str else ''}{tp_str}"
        process_name = "VllmWorker"
        if suffix:
            set_process_title(suffix, append=True)
            process_name = f"{process_name} {suffix}"
        decorate_logs(process_name)

        # Initializes a message queue for sending the model output
        # Initialize device and loads weights
        self.worker.init_device()
        self.rpc_response_handles = []

        # Initialize MessageQueue for receiving SchedulerOutput
        if node_size == 1:
            assert input_shm_handle is not None
            self.rpc_broadcast_mq = MessageQueue.create_from_handle(
                input_shm_handle, self.worker.rank)
            self.worker_response_mq = MessageQueue(1, 1)
        else:
            # multi node
            # generate mq broadcaster from world group for cross node communication
            self.rpc_broadcast_mq = get_world_group().create_mq_broadcaster(
                extra_writer_handler=input_shm_handle,
                # we will wait until ready later
                blocking = False
            )
            self.worker_response_mq, self.rpc_response_handles = get_world_group().create_single_reader_mq_broadcasters(
                reader_rank=0)
        self.worker.load_model()

    @staticmethod
    def make_worker_process(
            vllm_config: VllmConfig,
            local_rank: int,
            rank: int,
            distributed_init_method: str,
            input_shm_handle,  # Receive SchedulerOutput
    ) -> UnreadyWorkerProcHandle:
        context = get_mp_context()
        # (reader, writer)
        reader, writer = context.Pipe(duplex=False)

        # Create death pipe to detect parent process exit
        death_reader, death_writer = context.Pipe(duplex=False)

        process_kwargs = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "input_shm_handle": input_shm_handle,
            "ready_pipe": (reader, writer),
            "death_pipe": death_reader,
        }
        # Run EngineCore busy loop in background process.
        proc = context.Process(target=WorkerProc.worker_main,
                               kwargs=process_kwargs,
                               name=f"VllmWorker-{rank}",
                               daemon=True)

        proc.start()
        writer.close()
        # Keep death_writer open in parent - when parent exits,
        # death_reader in child will get EOFError
        return UnreadyWorkerProcHandle(proc, rank, reader, death_writer)

    @staticmethod
    def wait_for_ready(
        unready_proc_handles: list[UnreadyWorkerProcHandle]
    ) -> list[WorkerProcHandle]:

        e = Exception("WorkerProc initialization failed due to "
                      "an exception in a background process. "
                      "See stack trace for root cause.")

        pipes = {handle.ready_pipe: handle for handle in unready_proc_handles}
        ready_proc_handles: list[Optional[WorkerProcHandle]] = (
            [None] * len(unready_proc_handles))
        while pipes:
            ready = multiprocessing.connection.wait(pipes.keys())
            for pipe in ready:
                assert isinstance(pipe, Connection)
                try:
                    # Wait until the WorkerProc is ready.
                    unready_proc_handle = pipes.pop(pipe)
                    response: dict[str, Any] = pipe.recv()
                    if response["status"] != "READY":
                        raise e

                    worker_response_mq = None
                    # Extract the message queue handle.
                    if len(response["handle"].local_reader_ranks) > 0:
                        worker_response_mq = MessageQueue.create_from_handle(
                            response["handle"], 0)
                    # ensure this is remote handle
                    remote_response_mqs = [MessageQueue.create_from_handle(
                        response, -1) if response.remote_subscribe_addr is not None else None for response in response["rpc_response_handles"]]
                    idx = unready_proc_handle.rank % len(ready_proc_handles)
                    ready_proc_handles[idx] = (
                        WorkerProcHandle.from_unready_handle(
                            unready_proc_handle, worker_response_mq, remote_response_mqs))
                except EOFError:
                    e.__suppress_context__ = True
                    raise e from None

                finally:
                    # Close connection.
                    pipe.close()

        return cast(list[WorkerProcHandle], ready_proc_handles)

    def shutdown(self):
        self.rpc_broadcast_mq = None
        self.worker_response_mq = None
        destroy_model_parallel()
        destroy_distributed_environment()

    @staticmethod
    def worker_main(*args, **kwargs):
        """ Worker initialization and execution loops.
        This runs a background process """

        # Signal handler used for graceful termination.
        # SystemExit exception is only raised once to allow this and worker
        # processes to terminate without error
        shutdown_requested = False

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit()

        # Either SIGTERM or SIGINT will terminate the worker
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        worker = None
        # tuple[Connection, Connection]
        reader, ready_writer = kwargs.pop("ready_pipe")
        death_pipe = kwargs.pop("death_pipe", None)

        # Start death monitoring thread if death_pipe is provided
        if death_pipe is not None:

            def monitor_parent_death():
                try:
                    # This will block until parent process exits (pipe closes)
                    death_pipe.recv()
                except EOFError:
                    # Parent process has exited, terminate this worker
                    logger.info("Parent process exited, terminating worker")
                    # Send signal to self to trigger clean shutdown
                    os.kill(os.getpid(), signal.SIGTERM)
                except Exception as e:
                    logger.warning("Death monitoring error: %s", e)

            death_monitor = Thread(target=monitor_parent_death,
                                   daemon=True,
                                   name="WorkerDeathMonitor")
            death_monitor.start()

        try:
            reader.close()
            worker = WorkerProc(*args, **kwargs)

            # Send READY once we know everything is loaded
            ready_writer.send({
                "status":
                WorkerProc.READY_STR,
                "handle":
                worker.worker_response_mq.export_handle(),
                "rpc_response_handles": worker.rpc_response_handles,
            })

            # Ensure message queues are ready. Will deadlock if re-ordered.
            # Must be kept consistent with the Executor
            worker.rpc_broadcast_mq.wait_until_ready()
            worker.worker_response_mq.wait_until_ready()
            ready_writer.close()
            ready_writer = None

            worker.worker_busy_loop()

        except Exception:
            # NOTE: if an Exception arises in busy_loop, we send
            # a FAILURE message over the MQ RPC to notify the Executor,
            # which triggers system shutdown.
            # TODO(rob): handle case where the MQ itself breaks.

            if ready_writer is not None:
                logger.exception("WorkerProc failed to start.")
            else:
                logger.exception("WorkerProc failed.")

            # The parent sends a SIGTERM to all worker processes if
            # any worker dies. Set this value so we don't re-throw
            # SystemExit() to avoid zmq exceptions in __del__.
            shutdown_requested = True

        finally:
            if ready_writer is not None:
                ready_writer.close()
            if death_pipe is not None:
                death_pipe.close()
            # Clean up once worker exits busy loop
            if worker is not None:
                worker.shutdown()

    class ResponseStatus(Enum):
        SUCCESS = auto()
        FAILURE = auto()

    def worker_busy_loop(self):
        """Main busy loop for Multiprocessing Workers"""
        while True:
            method, args, kwargs, output_rank = self.rpc_broadcast_mq.dequeue()
            try:
                if isinstance(method, str):
                    func = getattr(self.worker, method)
                elif isinstance(method, bytes):
                    func = partial(cloudpickle.loads(method), self.worker)
                output = func(*args, **kwargs)
            except Exception as e:
                # Notes have been introduced in python 3.11
                if hasattr(e, "add_note"):
                    e.add_note(traceback.format_exc())
                logger.exception("WorkerProc hit an exception.")
                # exception might not be serializable, so we convert it to
                # string, only for logging purpose.
                if output_rank is None or self.rank == output_rank:
                    self.worker_response_mq.enqueue(
                        (WorkerProc.ResponseStatus.FAILURE, str(e)))
                continue

            if output_rank is None or self.rank == output_rank:
                self.worker_response_mq.enqueue(
                    (WorkerProc.ResponseStatus.SUCCESS, output))
