import sys
from pathlib import Path

import pytest
import torch

from traincheck.heldout_workload import Workload
from traincheck.schedule_study import measure_schedule

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs/scripts"))
from analyze_schedule_study import calibration, decisions


def test_calibration_uses_worst_healthy_schedule_and_strict_cutoff():
    assert calibration([list(range(40)), list(range(20, 60))], 0.05) == 58
    row = dict(status="ok", loss_score=58, steps=8, actual_calls=6, mismatch_slots=0)
    assert decisions(row, 58) == dict(fixed_loss=True, calibrated=False, naive=True, declared=False)


def test_declared_and_faulty_skips_can_execute_identical_training():
    torch.set_num_threads(1)
    x = torch.rand(16, 4, generator=torch.Generator().manual_seed(42))
    y = torch.arange(16) % 2
    workload = Workload("fixture", x[:8], y[:8], x[8:12], y[8:12], x[12:], y[12:], (4,), 2, 4)
    healthy = measure_schedule(workload, 42, dict(schedule="declared_skip4", pattern="none", denominator=1), {"steps": 8})
    faulty = measure_schedule(workload, 42, dict(schedule="plain", pattern="periodic", denominator=4), {"steps": 8})
    assert healthy["actual_calls"] == faulty["actual_calls"] == 6
    assert healthy["loss_score"] == faulty["loss_score"]
    assert healthy["heldout_accuracy"] == faulty["heldout_accuracy"]
    assert healthy["mismatch_slots"] == 0
    assert faulty["mismatch_slots"] == 2
    accumulation = measure_schedule(workload, 42, dict(schedule="accumulate2", pattern="none", denominator=1), {"steps": 7})
    assert accumulation["planned_calls"] == accumulation["actual_calls"] == 3
    assert accumulation["mismatch_slots"] == 0


@pytest.mark.parametrize(("schedule", "fault", "calls"), [
    ("plain", "missing_zero_grad", 8),
    ("accumulate2", "unscaled_accumulation", 4),
])
def test_gradient_mutations_change_loss_but_preserve_call_conformance(schedule, fault, calls):
    torch.set_num_threads(1)
    x = torch.rand(16, 4, generator=torch.Generator().manual_seed(42))
    y = torch.arange(16) % 2
    workload = Workload("fixture", x[:8], y[:8], x[8:12], y[8:12], x[12:], y[12:], (4,), 2, 4)
    healthy_cell = dict(schedule=schedule, pattern="none", denominator=1)
    healthy = measure_schedule(workload, 42, healthy_cell, {"steps": 8})
    faulty = measure_schedule(workload, 42, {**healthy_cell, "gradient_fault": fault},
                              {"steps": 8}, include_heldout_loss=True)
    assert healthy["actual_calls"] == faulty["actual_calls"] == calls
    assert healthy["mismatch_slots"] == faulty["mismatch_slots"] == 0
    assert healthy["loss_score"] != faulty["loss_score"]
    assert "heldout_loss" not in healthy
    assert 0 < faulty["heldout_loss"] < float("inf")
