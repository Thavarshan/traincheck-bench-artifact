from __future__ import annotations

import os
import platform
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import torch
from torch import nn

FAULTS = ("healthy", "skip_step", "detached_gradients",
          "wrong_labels", "train_eval_mode", "preprocessing_mismatch",
          "intermittent_skip_step")


@dataclass
class Outcome:
    seed: int
    fault: str
    budget_seconds: float
    elapsed_seconds: float
    diagnose_wall_seconds: float
    budget_overshoot_seconds: float
    steps: int
    optimizer_step_calls: int
    termination_reason: str
    over_budget: bool
    initial_loss: float
    final_loss: float
    serving_loss: float
    preprocessing_max_abs_delta: float
    gradient_nonzero: bool
    weights_updated: bool
    loss_decreased: bool
    overfit_small_batch: bool
    detected: bool
    detections: dict[str, bool]
    torch_version: str
    device: str = "cpu"
    status: str = "ok"
    runtime_scope: str = "training_loop_only"
    python_version: str = platform.python_version()
    source_revision: str = "unavailable"
    working_tree_dirty: bool | None = None
    host_platform: str = platform.platform()
    cpu_count: int | None = os.cpu_count()
    configuration: dict[str, object] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def make_data(seed: int, n: int = 48, features: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, features, generator=generator)
    weights = torch.tensor([1., -1., .7, -.7, .4, -.4, .2, -.2])
    y = ((x @ weights) > 0).long()
    return x, y


@lru_cache(maxsize=1)
def _git_metadata() -> tuple[str, bool | None]:
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
    )
    if revision.returncode:
        return "unavailable", None
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return revision.stdout.strip(), bool(status.stdout.strip()) if status.returncode == 0 else None


def _configuration(max_steps: int) -> dict[str, object]:
    return {
        "task": "synthetic_binary_classification",
        "data_generator": "torch.randn with explicit seed",
        "examples": 48,
        "features": 8,
        "label_weights": [1.0, -1.0, 0.7, -0.7, 0.4, -0.4, 0.2, -0.2],
        "training_batch_size": 16,
        "model": [8, 16, 2],
        "activation": "ReLU",
        "dropout": 0.2,
        "optimizer": "SGD",
        "learning_rate": 0.15,
        "loss": "cross_entropy",
        "max_steps": max_steps,
        "step_counter_semantics": "training loop iterations; skipped optimizer calls still count",
        "optimizer_step_calls_semantics": "SGD.step invocations, not guaranteed parameter updates",
        "intermittent_skip_step_rule": "skip iteration indices 0,4,8,...; call optimizer on the other 3/4",
        "torch_num_threads": 1,
        "reference_loss_scope": "fixed first 16 raw examples; original labels; dropout disabled",
        "serving_preprocessing_mutation": "additive shift of +0.75 to serving features only",
        "preprocessing_contract_tolerance": 1e-6,
        "loss_decrease_ratio": 0.98,
        "tiny_batch_loss_threshold": 0.25,
        "composite_detection_rule": "any_candidate_check_false_unvalidated",
        "device": "cpu",
        "timer_scope": "elapsed_seconds: optimization loop; diagnose_wall_seconds: whole diagnose function, excluding interpreter startup and output serialization",
    }


def _reference_loss(model: nn.Module, features: torch.Tensor, targets: torch.Tensor) -> float:
    """Evaluate a fixed raw-input/reference-label batch without dropout noise.

    Restore the caller's mode so a diagnostic evaluation cannot change the
    training-mode mutation or subsequent training behavior.
    """
    previous_mode = model.training
    try:
        model.eval()
        with torch.inference_mode():
            return float(nn.functional.cross_entropy(model(features), targets))
    finally:
        model.train(previous_mode)


def _preprocessing_contract(
    training_features: torch.Tensor, serving_features: torch.Tensor
) -> tuple[bool, float]:
    """Compare observed feature tensors on the same probe, not fault labels."""
    max_delta = float((training_features - serving_features).abs().max())
    return bool(torch.allclose(training_features, serving_features, rtol=0.0, atol=1e-6)), max_delta


def diagnose(seed: int = 17, fault: str = "healthy", budget_seconds: float = 2.0, max_steps: int = 120) -> Outcome:
    if fault not in FAULTS:
        raise ValueError(f"unknown fault {fault!r}; choose from {FAULTS}")
    if budget_seconds <= 0:
        raise ValueError("budget_seconds must be positive")
    if max_steps < 0:
        raise ValueError("max_steps cannot be negative")

    diagnose_started = time.perf_counter()
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    x, y = make_data(seed)
    reference_x, reference_y = x[:16], y[:16]
    # A controlled serving-pipeline mutation: actual inference features differ
    # from training features for the same original input. The detector below
    # sees the resulting tensors only, never the injected condition label.
    serving_x = reference_x + \
        0.75 if fault == "preprocessing_mismatch" else reference_x.clone()

    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(),
                          nn.Dropout(.2), nn.Linear(16, 2))
    optimizer = torch.optim.SGD(model.parameters(), lr=.15)
    initial = _reference_loss(model, reference_x, reference_y)
    before = [p.detach().clone() for p in model.parameters()]
    model.train()
    if fault == "train_eval_mode":
        model.eval()

    started = time.perf_counter()
    deadline, steps = started + budget_seconds, 0
    optimizer_step_calls = 0
    all_gradients_valid = True
    all_training_modes_valid = True
    all_training_losses_finite = True
    while steps < max_steps and time.perf_counter() < deadline:
        all_training_modes_valid = all_training_modes_valid and model.training
        optimizer.zero_grad(set_to_none=True)
        logits = model(reference_x)
        targets = 1 - reference_y if fault == "wrong_labels" else reference_y
        loss = nn.functional.cross_entropy(logits, targets)
        if fault == "detached_gradients":
            loss = loss.detach().requires_grad_(True)
        all_training_losses_finite = all_training_losses_finite and bool(
            torch.isfinite(loss.detach()).all())
        loss.backward()
        present_grads = [p.grad for p in model.parameters()
                         if p.grad is not None]
        step_gradients_valid = (
            bool(present_grads)
            and all(bool(torch.isfinite(grad).all()) for grad in present_grads)
            and any(bool(grad.abs().sum() > 0) for grad in present_grads)
        )
        all_gradients_valid = all_gradients_valid and step_gradients_valid
        # The intermittent mutation drops every fourth attempted update.
        # Count calls separately: detached gradients still invoke SGD.step()
        # without necessarily changing a parameter.
        skip_update = fault == "skip_step" or (
            fault == "intermittent_skip_step" and steps % 4 == 0
        )
        if not skip_update:
            optimizer.step()
            optimizer_step_calls += 1
        steps += 1
    elapsed = time.perf_counter() - started
    budget_overshoot = max(0.0, elapsed - budget_seconds)

    # Both reported losses use identical examples, labels and evaluation mode.
    # These post-training evaluations are outside the training-loop timer.
    final = _reference_loss(model, reference_x, reference_y)
    serving_loss = _reference_loss(model, serving_x, reference_y)
    preprocessing_ok, preprocessing_delta = _preprocessing_contract(
        reference_x, serving_x)
    updated = any(not torch.equal(old, new.detach())
                  for old, new in zip(before, model.parameters()))
    gradient_nonzero = steps > 0 and all_gradients_valid
    training_mode_ok = steps > 0 and all_training_modes_valid
    finite_loss_ok = steps > 0 and all_training_losses_finite and bool(
        torch.isfinite(torch.tensor(final)))
    loss_decreased = steps > 0 and final < initial
    overfit_small_batch = steps > 0 and final < .25

    detections = {
        "finite_loss": finite_loss_ok,
        "nonzero_gradients": gradient_nonzero,
        "weight_update": updated,
        "early_loss_decrease": steps > 0 and final < initial * .98,
        "tiny_batch_overfit": overfit_small_batch,
        "training_mode": training_mode_ok,
        "preprocessing_contract": preprocessing_ok,
    }
    # Provisional, uncalibrated composite. Do not interpret as validated recall.
    detected = any(not value for value in detections.values())
    revision, dirty = _git_metadata()
    diagnose_wall_seconds = time.perf_counter() - diagnose_started
    return Outcome(
        seed=seed,
        fault=fault,
        budget_seconds=budget_seconds,
        elapsed_seconds=elapsed,
        diagnose_wall_seconds=diagnose_wall_seconds,
        budget_overshoot_seconds=budget_overshoot,
        steps=steps,
        optimizer_step_calls=optimizer_step_calls,
        termination_reason="max_steps" if steps >= max_steps else "budget_reached",
        over_budget=elapsed > budget_seconds,
        initial_loss=initial,
        final_loss=final,
        serving_loss=serving_loss,
        preprocessing_max_abs_delta=preprocessing_delta,
        gradient_nonzero=gradient_nonzero,
        weights_updated=updated,
        loss_decreased=loss_decreased,
        overfit_small_batch=overfit_small_batch,
        detected=detected,
        detections=detections,
        torch_version=torch.__version__,
        python_version=platform.python_version(),
        source_revision=revision,
        working_tree_dirty=dirty,
        host_platform=platform.platform(),
        cpu_count=os.cpu_count(),
        configuration=_configuration(max_steps),
    )
