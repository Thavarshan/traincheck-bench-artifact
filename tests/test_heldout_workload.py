from dataclasses import replace

import pytest
import torch

from traincheck.experiment import FAULTS
from traincheck.heldout_datasets import read_verified
from traincheck.heldout_workload import Workload, _stratified_indices, measure


@pytest.fixture
def workload() -> Workload:
    base_x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    base_y = torch.tensor([0, 1, 2])
    return Workload(
        name="test-workload",
        train_x=base_x.repeat(10, 1),
        train_y=base_y.repeat(10),
        validation_x=base_x * 0.9,
        validation_y=base_y,
        test_x=base_x * 1.1,
        test_y=base_y,
        hidden_layers=(4,),
        classes=3,
        batch_size=6,
    )


def run(workload: Workload, fault: str) -> dict:
    return measure(workload, seed=47, fault=fault, budget_seconds=5.0, max_steps=8)


def test_frozen_data_validation_rejects_modified_file(tmp_path):
    (tmp_path / "wine.data").write_text("corrupted")
    with pytest.raises(ValueError, match="Dataset checksum mismatch"):
        read_verified(tmp_path, "wine.data")


def test_stratified_partitions_are_disjoint_and_reproducible():
    labels = torch.tensor([1, 0] * 15)
    train, validation, test = _stratified_indices(labels, include_test=True)
    assert (train, validation, test) == _stratified_indices(labels, include_test=True)
    assert sorted(train + validation + test) == list(range(30))
    assert [int(labels[indices].sum()) for indices in (train, validation, test)] == [9, 3, 3]


@pytest.mark.parametrize("fault", FAULTS)
def test_each_fault_runs_with_valid_heldout_and_timing_fields(workload, fault):
    record = run(workload, fault)
    assert record["steps"] == 8
    assert record["termination_reason"] == "max_steps"
    assert record["diagnose_wall_seconds"] >= record["elapsed_seconds"] > 0
    assert record["semantic_eval_wall_seconds"] > 0
    assert 0 <= record["heldout_accuracy"] <= 1
    assert record["heldout_n"] == 3
    assert all(type(passed) is bool for passed in record["detections"].values())


def test_schedule_contract_tracks_completed_calls_and_not_gradients(workload):
    healthy = run(workload, "healthy")
    partial = run(workload, "intermittent_skip_step")
    full = run(workload, "skip_step")
    detached = run(workload, "detached_gradients")
    assert [record["optimizer_step_calls"] for record in (healthy, partial, full, detached)] == [8, 6, 0, 8]
    assert [record["detections"]["expected_updates"] for record in (healthy, partial, full, detached)] == [
        True, False, False, True,
    ]
    assert full["initial_loss"] == full["final_loss"]
    assert not detached["detections"]["weight_update"]
    assert not detached["detections"]["nonzero_gradients"]


def test_test_labels_do_not_change_detectors(workload):
    original = run(workload, "healthy")
    changed = run(replace(workload, test_y=(workload.test_y + 1) % 3), "healthy")
    assert original["detections"] == changed["detections"]
    assert original["final_loss"] == changed["final_loss"]
    assert original["heldout_loss"] != changed["heldout_loss"]


def test_preprocessing_shift_changes_serving_inputs_without_retraining(workload):
    healthy = run(workload, "healthy")
    shifted = run(workload, "preprocessing_mismatch")
    assert shifted["preprocessing_max_abs_delta"] == pytest.approx(0.75)
    assert not shifted["detections"]["preprocessing_contract"]
    assert shifted["heldout_clean_loss"] == healthy["heldout_loss"]
    assert shifted["heldout_loss"] != shifted["heldout_clean_loss"]


def test_mode_contract_samples_mode_during_training(workload):
    assert not run(workload, "train_eval_mode")["detections"]["training_mode"]
