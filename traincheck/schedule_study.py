"""Matched training-loop exposures and declared optimizer schedule controls."""

from __future__ import annotations

import gzip
import hashlib
import random
import struct
import time
from dataclasses import replace
from pathlib import Path
from urllib.request import urlopen

import torch
from torch import nn

from .heldout_datasets import fetch
from .heldout_workload import Workload, _batches, _evaluate, _model, load_workload

FASHION_SOURCE = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/b2617bb6d3ffa2e429640350f613e3291e10b141/data/fashion/"
FASHION_FILES = {
    "train-images-idx3-ubyte.gz": "3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84",
    "train-labels-idx1-ubyte.gz": "a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845",
    "t10k-images-idx3-ubyte.gz": "346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073",
    "t10k-labels-idx1-ubyte.gz": "67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5",
}
HEALTHY = ("plain", "accumulate2", "declared_skip4")


def _fashion_data(directory: Path, prefix: str) -> tuple[torch.Tensor, torch.Tensor]:
    images = ["images-idx3-ubyte.gz", "labels-idx1-ubyte.gz"]
    decoded = []
    for suffix in images:
        name = f"{prefix}-{suffix}"
        path = directory / name
        if path.is_file():
            data = path.read_bytes()
        else:
            with urlopen(FASHION_SOURCE + name, timeout=90) as response:
                data = response.read(32 * 1024**2)
        if hashlib.sha256(data).hexdigest() != FASHION_FILES[name]:
            raise ValueError(f"Fashion-MNIST checksum mismatch for {name}")
        if not path.is_file():
            path.write_bytes(data)
        decoded.append(gzip.decompress(data))
    pixels, labels = decoded
    magic, count, height, width = struct.unpack(">IIII", pixels[:16])
    label_magic, label_count = struct.unpack(">II", labels[:8])
    if (magic, height, width, label_magic, label_count) != (2051, 28, 28, 2049, count):
        raise ValueError("Unexpected Fashion-MNIST headers")
    return (
        torch.frombuffer(bytearray(pixels[16:]), dtype=torch.uint8).reshape(count, 1, 28, 28),
        torch.frombuffer(bytearray(labels[8:]), dtype=torch.uint8).long(),
    )


def load_schedule_workload(name: str, directory: Path, evaluate: bool) -> Workload:
    directory.mkdir(parents=True, exist_ok=True)
    if name != "fashion":
        fetch(directory)
        workload = load_workload(name, directory)
        if not evaluate:
            workload = replace(workload, test_x=workload.test_x[:0], test_y=workload.test_y[:0])
        return workload
    x, y = _fashion_data(directory, "train")

    def select(features: torch.Tensor, labels: torch.Tensor, start: int, stop: int):
        indices = torch.cat([(labels == c).nonzero().flatten()[start:stop] for c in range(10)])
        return features[indices].float().div_(255), labels[indices]

    train, train_y = select(x, y, 0, 600)
    val, val_y = select(x, y, 600, 620)
    if evaluate:
        test, test_y = select(*_fashion_data(directory, "t10k"), 0, 100)
    else:
        test, test_y = val[:0], val_y[:0]
    return Workload(name, train, train_y, val, val_y, test, test_y, (), 10, 32)


def expected_call(schedule: str, index: int) -> bool:
    if schedule == "accumulate2":
        return index % 2 == 1
    if schedule == "declared_skip4":
        return index % 4 != 0
    if schedule == "plain":
        return True
    raise ValueError(f"Unknown schedule: {schedule}")


def omitted(pattern: str, denominator: int, index: int, draws: random.Random) -> bool:
    if pattern == "none":
        return False
    if pattern == "periodic":
        return index % denominator == 0
    if pattern == "bernoulli":
        return draws.randrange(denominator) == 0
    raise ValueError(f"Unknown omission pattern: {pattern}")


def measure_schedule(
    workload: Workload, seed: int, cell: dict, budget: dict, *, include_heldout_loss: bool = False,
) -> dict:
    started = time.perf_counter()
    cpu_started = time.process_time()
    schedule, pattern, denominator = cell["schedule"], cell["pattern"], cell["denominator"]
    gradient_fault = cell.get("gradient_fault")
    if gradient_fault not in (None, "missing_zero_grad", "unscaled_accumulation"):
        raise ValueError(f"Unknown gradient fault: {gradient_fault}")
    if gradient_fault == "unscaled_accumulation" and schedule != "accumulate2":
        raise ValueError("Unscaled accumulation requires the accumulation schedule")
    accumulate = schedule == "accumulate2"
    average = accumulate and gradient_fault != "unscaled_accumulation"
    clear_gradients = gradient_fault != "missing_zero_grad"
    draws = random.Random(seed ^ 0xC117)
    torch.manual_seed(seed)
    if workload.name == "fashion":
        model = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.AvgPool2d(4),
                              nn.Flatten(), nn.Linear(8 * 7 * 7, 10))
    else:
        model = _model(workload)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.15)
    batches = _batches(workload, seed)
    initial, _ = _evaluate(model, workload.validation_x, workload.validation_y)
    actual = 0

    def count_call(_optimizer, _args, _kwargs):
        nonlocal actual
        actual += 1

    optimizer.register_step_post_hook(count_call)
    steps = planned = mismatches = injected = 0
    loop_started = time.perf_counter()
    deadline = loop_started + budget["seconds"] if "seconds" in budget else float("inf")
    maximum = budget.get("steps", 10000)
    while steps < maximum and time.perf_counter() < deadline:
        indices = next(batches)
        if clear_gradients and (not accumulate or steps % 2 == 0):
            optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(workload.train_x[indices]), workload.train_y[indices])
        (loss / 2 if average else loss).backward()
        # The detector receives the declared schedule and the observed call count for this iteration.
        required = expected_call(schedule, steps)
        planned += required
        skip = required and omitted(pattern, denominator, steps, draws)
        injected += skip
        old_actual = actual
        if required and not skip:
            optimizer.step()
        mismatches += actual - old_actual != required
        steps += 1
    elapsed = time.perf_counter() - loop_started
    final, _ = _evaluate(model, workload.validation_x, workload.validation_y)
    diagnosis = time.perf_counter() - started
    cpu = time.process_time() - cpu_started
    heldout_loss = heldout = None
    if workload.test_y.numel():
        heldout_loss, heldout = _evaluate(model, workload.test_x, workload.test_y)
    result = {
        "steps": steps, "planned_calls": planned, "actual_calls": actual, "mismatch_slots": mismatches,
        "injected_omissions": injected, "loss_score": final / initial, "heldout_accuracy": heldout,
        "loop_wall_seconds": elapsed, "diagnose_wall_seconds": diagnosis, "diagnose_cpu_seconds": cpu,
    }
    if include_heldout_loss:
        result["heldout_loss"] = heldout_loss
    return result
