"""Private CPU scheduling probe; never connects to a configured/live provider.

Run with the CPU-contract environment after generating services/generated stubs.
The encode gate is synthetic: this establishes ordering, not production latency.
"""

import asyncio
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[3]
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
os.nice(10)
resource.setrlimit(resource.RLIMIT_AS, (16 * 1024**3, 16 * 1024**3))
sys.path[:0] = [str(ROOT / "services/tests"), str(ROOT / "services"),
               str(ROOT / "services/generated")]

import grpc
import torch
from test_training_rpc_validation import loaded_service
from hades.training import training_pb2 as pb, training_pb2_grpc as rpc
from training.session import SESSION_HEADER, SessionTrainingServicer


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


async def run_case(cancel):
    torch.manual_seed(7)
    backend = loaded_service()
    before = {k: v.clone() for k, v in backend.model.state_dict().items()}
    entered, released, finished = threading.Event(), threading.Event(), threading.Event()
    record = {"cancel_requested": cancel, "loop_callback_ran_before_step": False}
    errors = []
    loop = asyncio.get_running_loop()
    original_encode, original_step = backend._encode, backend.optimizer.step
    callback_ran = False

    def callback():
        nonlocal callback_ran
        callback_ran = True

    def encode():
        # Queue a callback before real encoding; it cannot run until this valid
        # TrainStep yields. A separate client thread controls the ordering gate.
        loop.call_soon(callback)
        entered.set()
        require(released.wait(5), "client failed to release encode gate")
        return original_encode()

    def step(*args, **kwargs):
        record["loop_callback_ran_before_step"] = callback_ran
        result = original_step(*args, **kwargs)
        record["optimizer_step_finished_ns"] = time.monotonic_ns()
        finished.set()
        return result

    backend._encode, backend.optimizer.step = encode, step
    service = SessionTrainingServicer(lambda: backend)
    with tempfile.TemporaryDirectory(prefix="hades-cancel-") as directory:
        endpoint = f"unix:{directory}/rpc.sock"
        server = grpc.aio.server()
        rpc.add_TrainingServiceServicer_to_server(service, server)
        require(server.add_insecure_port(endpoint), "private bind failed")
        await server.start()

        def client():
            try:
                with grpc.insecure_channel(endpoint) as channel:
                    stub = rpc.TrainingServiceStub(channel)
                    token = stub.AcquireSession(pb.AcquireSessionRequest(), timeout=5).token
                    pending = stub.TrainStep.future(
                        pb.TrainStepRequest(train_edge_indices=[0], neg_src=[3], neg_dst=[0]),
                        metadata=[(SESSION_HEADER, token)], timeout=5)
                    require(entered.wait(5), "TrainStep did not enter encoding")
                    if cancel:
                        record["client_cancel_accepted"] = pending.cancel()
                        record["client_cancel_ns"] = time.monotonic_ns()
                        require(record["client_cancel_accepted"], "call already completed")
                    released.set()
                    try:
                        pending.result(timeout=5)
                        record["client_outcome"] = "success"
                    except grpc.FutureCancelledError:
                        record["client_outcome"] = "cancelled"
                    require(finished.wait(5), "optimizer did not finish")
            except BaseException as exc:
                errors.append(repr(exc))
            finally:
                released.set()

        thread = threading.Thread(target=client, daemon=True)
        thread.start()
        try:
            deadline = loop.time() + 15
            while thread.is_alive() and loop.time() < deadline:
                await asyncio.sleep(0.01)
            require(not thread.is_alive(), "private client did not exit")
            require(not errors, str(errors))
            record["weights_changed"] = any(
                not torch.equal(value, backend.model.state_dict()[key])
                for key, value in before.items())
            record["callback_eventually_ran"] = callback_ran
            require(record["weights_changed"], "fixture did not update model")
            require(not record["loop_callback_ran_before_step"], "scheduling behavior changed")
            require(callback_ran, "event-loop callback never ran")
            if cancel:
                require(record["client_outcome"] == "cancelled", "client cancellation not observed")
                require(record["client_cancel_ns"] < record["optimizer_step_finished_ns"],
                        "cancellation did not precede optimizer completion")
        finally:
            released.set()
            await server.stop(0)
            await service.close()
            await asyncio.to_thread(thread.join, 1)
            require(not thread.is_alive(), "private client survived cleanup")
    record["private_state_removed"] = not Path(directory).exists()
    return record


async def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    paths = [Path(__file__), ROOT / "services/training/server.py",
             ROOT / "services/training/session.py",
             ROOT / "services/tests/test_training_rpc_validation.py"]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    cases = [await run_case(False), await run_case(True)]
    print(json.dumps({"status": "reproduced", "source_sha256": hashes,
                      "torch_version": torch.__version__, "grpc_version": grpc.__version__,
                      "cases": cases, "synthetic_ordering_gate": True}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
