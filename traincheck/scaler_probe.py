"""Observe CPU GradScaler and StepLR on successful and skipped updates."""

from __future__ import annotations

import warnings

import torch
from torch import nn

CASES = ("finite", "overflow", "omission", "ungated_scheduler")


def probe(seed: int, case: str) -> list[dict]:
    if case not in CASES:
        raise ValueError(f"Unknown scaler case: {case}")
    generator = torch.Generator().manual_seed(seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        scaler = torch.amp.GradScaler("cpu", init_scale=32.0, growth_interval=100)
        calls = 0

        def track(_optimizer, _args, _kwargs):
            nonlocal calls
            calls += 1

        optimizer.register_step_post_hook(track)
        trace = []
        for batch in range(8):
            optimizer.zero_grad(set_to_none=True)
            x = torch.randn(8, 4, generator=generator)
            y = torch.randint(2, (8,), generator=generator)
            with torch.autocast(device_type="cpu", dtype=torch.float16):
                loss = nn.functional.cross_entropy(model(x), y)
            scaler.scale(loss).backward()
            if batch == 0 and case in ("overflow", "ungated_scheduler"):
                next(model.parameters()).grad.flatten()[0] = float("inf")
            scaler.unscale_(optimizer)
            finite = all(bool(torch.isfinite(p.grad).all()) for p in model.parameters())
            before_calls = calls
            before_epoch = scheduler.last_epoch
            scale = scaler.get_scale()
            lr = optimizer.param_groups[0]["lr"]
            if batch != 0 or case != "omission":
                scaler.step(optimizer)
            scaler.update()
            applied = calls - before_calls
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                if applied or case == "ungated_scheduler":
                    scheduler.step()
            advanced = scheduler.last_epoch - before_epoch
            trace.append(dict(batch=batch, finite_gradients=finite, optimizer_calls=applied,
                              scheduler_calls=advanced, scale_before=scale, scale_after=scaler.get_scale(),
                              lr_before=lr, lr_after=optimizer.param_groups[0]["lr"],
                              naive_alarm=applied != 1, optimizer_mismatch=applied != int(finite),
                              scheduler_mismatch=advanced != applied,
                              scheduler_warnings=[str(item.message) for item in caught]))
    return trace
