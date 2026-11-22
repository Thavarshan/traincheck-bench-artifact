"""CPU workloads with training, diagnostic validation, and semantic test partitions."""

from __future__ import annotations

import csv
import hashlib
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from torch import Tensor, nn

from .experiment import FAULTS
from .heldout_datasets import DATA_FILES, read_verified


LOSS_DECREASE_RATIO = 0.98
FIRST_BATCH_LOSS_THRESHOLD = 0.25
PREPROCESSING_ATOL = 1e-6
PREPROCESSING_SHIFT = 0.75


@dataclass(frozen=True)
class Workload:
    name: str
    train_x: Tensor
    train_y: Tensor
    validation_x: Tensor
    validation_y: Tensor
    test_x: Tensor
    test_y: Tensor
    hidden_layers: tuple[int, ...]
    classes: int
    batch_size: int
    data_files: tuple[str, ...] = ()
    split_name: str = "fixture"


def _read_table(data_dir: Path, filename: str, expected_rows: int,
                n_features: int, label_first: bool) -> tuple[Tensor, Tensor]:
    rows = list(csv.reader(read_verified(data_dir, filename).decode("ascii").splitlines()))
    if len(rows) != expected_rows or any(len(row) != n_features + 1 for row in rows):
        raise ValueError(f"Unexpected shape for {filename}")
    features = [row[1:] for row in rows] if label_first else [row[:-1] for row in rows]
    labels = [int(row[0]) - 1 for row in rows] if label_first else [int(row[-1]) for row in rows]
    return torch.tensor([[float(value) for value in row] for row in features], dtype=torch.float32), torch.tensor(labels)


def _stratified_indices(labels: Tensor, *, include_test: bool) -> tuple[list[int], list[int], list[int]]:
    """Rank indices within each class by a fixed SHA-256 key, then allocate fifths."""
    members: dict[int, list[int]] = {}
    for row, label in enumerate(labels.tolist()):
        members.setdefault(label, []).append(row)
    train, validation, test = [], [], []
    for _, indices in sorted(members.items()):
        ordered = sorted(indices, key=lambda row: hashlib.sha256(
            f"traincheck-uci-split-v1:{row}".encode("ascii")).digest())
        fifth = (len(ordered) + 2) // 5
        test_count = fifth if include_test else 0
        test.extend(ordered[:test_count])
        validation.extend(ordered[test_count:test_count + fifth])
        train.extend(ordered[test_count + fifth:])
    return sorted(train), sorted(validation), sorted(test)


def load_workload(name: str, data_dir: Path) -> Workload:
    """Load pinned bytes and fit feature scaling exclusively on training rows."""
    if name == "digits":
        data_files = ("optdigits.tra", "optdigits.tes")
        original_x, original_y = _read_table(data_dir, data_files[0], 3823, 64, False)
        test_x, test_y = _read_table(data_dir, data_files[1], 1797, 64, False)
        train, validation, _ = _stratified_indices(original_y, include_test=False)
        hidden_layers, classes, batch_size = (32, 16), 10, 32
        split_name = "official-writer-disjoint-test; stratified-training-fifth-for-validation-v1"
    elif name == "wine":
        data_files = ("wine.data",)
        original_x, original_y = _read_table(data_dir, data_files[0], 178, 13, True)
        train, validation, test = _stratified_indices(original_y, include_test=True)
        test_x, test_y = original_x[test], original_y[test]
        hidden_layers, classes, batch_size = (16,), 3, 16
        split_name = "stratified-training-three-fifths-validation-fifth-test-fifth-v1"
    else:
        raise ValueError(f"Unknown workload {name!r}; use digits or wine")

    train_x = original_x[train]
    mean = train_x.mean(dim=0)
    scale = train_x.std(dim=0, unbiased=False)
    scale = torch.where(scale > 0, scale, 1.0)
    return Workload(
        name=name,
        train_x=(train_x - mean) / scale,
        train_y=original_y[train],
        validation_x=(original_x[validation] - mean) / scale,
        validation_y=original_y[validation],
        test_x=(test_x - mean) / scale,
        test_y=test_y,
        hidden_layers=hidden_layers,
        classes=classes,
        batch_size=batch_size,
        data_files=data_files,
        split_name=split_name,
    )


def _batches(workload: Workload, seed: int) -> Iterator[Tensor]:
    # Keep model initialization and dropout draws independent of minibatch order.
    generator = torch.Generator().manual_seed(seed ^ 0x5EEDB47C)
    while True:
        indices = torch.randperm(len(workload.train_y), generator=generator)
        yield from indices.split(workload.batch_size)


def _evaluate(model: nn.Module, features: Tensor, labels: Tensor) -> tuple[float, float]:
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            logits = model(features)
            loss = float(nn.functional.cross_entropy(logits, labels))
            accuracy = float((logits.argmax(dim=1) == labels).to(torch.float64).mean())
            return loss, accuracy
    finally:
        model.train(was_training)


def _model(workload: Workload) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = workload.train_x.shape[1]
    for hidden in workload.hidden_layers:
        layers.extend((nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(0.2)))
        width = hidden
    layers.append(nn.Linear(width, workload.classes))
    return nn.Sequential(*layers)


def measure(workload: Workload, seed: int, fault: str, budget_seconds: float, max_steps: int) -> dict:
    """Measure candidate checks, a call-schedule contract, and separate held-out quality."""
    if fault not in FAULTS:
        raise ValueError(f"Unknown fault {fault!r}")
    if not math.isfinite(budget_seconds) or budget_seconds <= 0:
        raise ValueError("Budget must be finite and positive")
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")

    diagnose_started = time.perf_counter()
    diagnose_cpu_started = time.process_time()
    torch.set_num_threads(1)
    serving_shift = PREPROCESSING_SHIFT if fault == "preprocessing_mismatch" else 0.0
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = _model(workload)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.15)
        parameters = tuple(model.parameters())
        before = tuple(parameter.detach().clone() for parameter in parameters)
        validation_initial, _ = _evaluate(model, workload.validation_x, workload.validation_y)
        batches = _batches(workload, seed)
        first_batch = next(batches)
        first_x, first_y = workload.train_x[first_batch], workload.train_y[first_batch]
        first_initial, _ = _evaluate(model, first_x, first_y)
        model.train(fault != "train_eval_mode")

        steps = 0
        optimizer_calls = 0
        gradients_ok = True
        training_mode_ok = True
        finite_training_loss = True

        def record_completed_step(_optimizer: torch.optim.Optimizer, _args: tuple, _kwargs: dict) -> None:
            nonlocal optimizer_calls
            optimizer_calls += 1

        optimizer.register_step_post_hook(record_completed_step)
        started = time.perf_counter()
        cpu_started = time.process_time()
        deadline = started + budget_seconds
        while steps < max_steps and time.perf_counter() < deadline:
            batch = first_batch if steps == 0 else next(batches)
            x, y = workload.train_x[batch], workload.train_y[batch]
            training_mode_ok = model.training and training_mode_ok
            optimizer.zero_grad(set_to_none=True)
            targets = (y + 1) % workload.classes if fault == "wrong_labels" else y
            loss = nn.functional.cross_entropy(model(x), targets)
            if fault == "detached_gradients":
                loss = loss.detach().requires_grad_(True)
            finite_training_loss = bool(torch.isfinite(loss.detach())) and finite_training_loss
            loss.backward()
            grads = [parameter.grad for parameter in parameters if parameter.grad is not None]
            step_gradients_ok = (bool(grads) and all(bool(torch.isfinite(grad).all()) for grad in grads)
                                 and any(bool(torch.count_nonzero(grad)) for grad in grads))
            gradients_ok = step_gradients_ok and gradients_ok
            skip = fault == "skip_step" or (fault == "intermittent_skip_step" and steps % 4 == 0)
            if not skip:
                optimizer.step()
            steps += 1
        elapsed = time.perf_counter() - started
        elapsed_cpu = time.process_time() - cpu_started

        validation_final, validation_accuracy = _evaluate(model, workload.validation_x, workload.validation_y)
        first_final, _ = _evaluate(model, first_x, first_y)
        serving_probe = first_x + serving_shift
        preprocessing_delta = float((first_x - serving_probe).abs().max())
        updated = any(not torch.equal(old, current.detach()) for old, current in zip(before, parameters))
        detections = {
            "finite_loss": steps > 0 and finite_training_loss and math.isfinite(validation_final),
            "nonzero_gradients": steps > 0 and gradients_ok,
            "weight_update": updated,
            "early_loss_decrease": steps > 0 and validation_final < validation_initial * LOSS_DECREASE_RATIO,
            "tiny_batch_overfit": steps > 0 and first_final < FIRST_BATCH_LOSS_THRESHOLD,
            "training_mode": steps > 0 and training_mode_ok,
            "preprocessing_contract": preprocessing_delta <= PREPROCESSING_ATOL,
            "expected_updates": steps > 0 and optimizer_calls == steps,
        }
        diagnostic_elapsed = time.perf_counter() - diagnose_started
        diagnostic_cpu = time.process_time() - diagnose_cpu_started

        # Evaluate the actual serving input transform as the primary semantic endpoint.
        # These labels and outputs are excluded from every detector and its timer.
        semantic_started = time.perf_counter()
        clean_test_loss, clean_test_accuracy = _evaluate(model, workload.test_x, workload.test_y)
        if serving_shift:
            test_loss, test_accuracy = _evaluate(model, workload.test_x + serving_shift, workload.test_y)
        else:
            test_loss, test_accuracy = clean_test_loss, clean_test_accuracy
        semantic_elapsed = time.perf_counter() - semantic_started

    return {
        "steps": steps,
        "optimizer_step_calls": optimizer_calls,
        "elapsed_seconds": elapsed,
        "elapsed_cpu_seconds": elapsed_cpu,
        "diagnose_wall_seconds": diagnostic_elapsed,
        "diagnose_cpu_seconds": diagnostic_cpu,
        "semantic_eval_wall_seconds": semantic_elapsed,
        "budget_overshoot_seconds": max(0.0, elapsed - budget_seconds),
        "over_budget": elapsed > budget_seconds,
        "termination_reason": "max_steps" if steps == max_steps else "budget_reached",
        "initial_loss": validation_initial,
        "final_loss": validation_final,
        "initial_batch_loss": first_initial,
        "final_batch_loss": first_final,
        "validation_loss": validation_final,
        "validation_accuracy": validation_accuracy,
        "heldout_loss": test_loss,
        "heldout_accuracy": test_accuracy,
        "heldout_clean_loss": clean_test_loss,
        "heldout_clean_accuracy": clean_test_accuracy,
        "heldout_n": len(workload.test_y),
        "reference_batch_size": len(first_y),
        "preprocessing_max_abs_delta": preprocessing_delta,
        "detections": detections,
        "configuration": {
            "dataset_sha256": {filename: DATA_FILES[filename][1] for filename in workload.data_files},
            "split": workload.split_name,
            "n_train": len(workload.train_y),
            "n_validation": len(workload.validation_y),
            "n_test": len(workload.test_y),
            "batch_size": workload.batch_size,
            "layers": [workload.train_x.shape[1], *workload.hidden_layers, workload.classes],
            "dropout": 0.2,
            "optimizer": "sgd",
            "learning_rate": 0.15,
            "torch_threads": 1,
            "loss_decrease_ratio": LOSS_DECREASE_RATIO,
            "first_batch_loss_threshold": FIRST_BATCH_LOSS_THRESHOLD,
            "preprocessing_atol": PREPROCESSING_ATOL,
            "preprocessing_shift": PREPROCESSING_SHIFT,
            "max_steps": max_steps,
        },
    }
