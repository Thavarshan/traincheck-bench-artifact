"""Regression tests for raw experimental evidence integrity."""

import hashlib
import json

from docs.scripts.validate_study_records import validate


def bundle():
    rows = [{
        "phase": "exploratory", "source_revision": "abc",
        "seed": 301, "fault": fault, "budget_seconds": 0.25,
        "status": "ok", "steps": 3, "working_tree_dirty": False,
        "configuration": {"max_steps": 20000},
        "elapsed_seconds": 0.01, "diagnose_wall_seconds": 0.02,
        "detections": {"finite_loss": True},
    } for fault in ("healthy", "skip_step")]
    raw = "".join(json.dumps(row) + "\n" for row in rows).encode()
    manifest = {
        "records_sha256": hashlib.sha256(raw).hexdigest(),
        "source_revision": "abc", "phase": "exploratory",
        "requested_seeds": [301],
        "requested_faults": ["healthy", "skip_step"],
        "requested_budgets_seconds": [0.25],
        "requested_max_steps": 20000,
        "record_count": 2, "success_count": 2,
        "failure_count": 0, "malformed_line_count": 0,
    }
    return raw, manifest


def test_consistent_bundle():
    raw, manifest = bundle()
    assert validate(raw, manifest) == []


def test_hash_mismatch():
    raw, manifest = bundle()
    manifest["records_sha256"] = "0" * 64
    assert any("SHA-256" in issue for issue in validate(raw, manifest))


def test_duplicate_and_missing_condition():
    raw, manifest = bundle()
    rows = [json.loads(line) for line in raw.splitlines()]
    rows[1]["fault"] = "healthy"
    changed = "".join(json.dumps(row) + "\n" for row in rows).encode()
    manifest["records_sha256"] = hashlib.sha256(changed).hexdigest()
    assert any("design combinations mismatch" in issue for issue in validate(
        changed, manifest))


def test_wrong_step_ceiling():
    raw, manifest = bundle()
    manifest["requested_max_steps"] = 120
    assert any("max_steps" in issue for issue in validate(raw, manifest))


def test_failed_run_is_kept_as_valid_record():
    raw, manifest = bundle()
    rows = [json.loads(line) for line in raw.splitlines()]
    rows[1]["status"] = "failed"
    rows[1]["diagnose_wall_seconds"] = None
    rows[1]["detections"] = {}
    changed = "".join(json.dumps(row) + "\n" for row in rows).encode()
    manifest["records_sha256"] = hashlib.sha256(changed).hexdigest()
    manifest["success_count"] = 1
    manifest["failure_count"] = 1
    assert validate(changed, manifest) == []


def test_new_optimizer_call_and_overshoot_fields_are_consistent():
    raw, manifest = bundle()
    rows = [json.loads(line) for line in raw.splitlines()]
    for row in rows:
        row["configuration"]["step_counter_semantics"] = "training_loop_iterations"
        row["optimizer_step_calls"] = 2
        row["budget_overshoot_seconds"] = 0.0
    encoded = "".join(json.dumps(row) + "\n" for row in rows).encode()
    manifest["records_sha256"] = hashlib.sha256(encoded).hexdigest()
    assert validate(encoded, manifest) == []
    rows[0]["optimizer_step_calls"] = 4
    broken = "".join(json.dumps(row) + "\n" for row in rows).encode()
    manifest["records_sha256"] = hashlib.sha256(broken).hexdigest()
    assert any("optimizer-call" in issue for issue in validate(broken, manifest))


def test_reject_inconsistent_reported_overshoot():
    raw, manifest = bundle()
    rows = [json.loads(line) for line in raw.splitlines()]
    for row in rows:
        row["configuration"]["step_counter_semantics"] = "training_loop_iterations"
        row["optimizer_step_calls"] = 3
        row["budget_overshoot_seconds"] = 0.0
    rows[1]["budget_overshoot_seconds"] = 0.2
    broken = "".join(json.dumps(row) + "\n" for row in rows).encode()
    manifest["records_sha256"] = hashlib.sha256(broken).hexdigest()
    assert any("budget-overshoot" in issue for issue in validate(broken, manifest))
