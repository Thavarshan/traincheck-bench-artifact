import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "docs/scripts/analyze_public_workload.py"
SPEC = importlib.util.spec_from_file_location("analyze_public_workload", SCRIPT)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def test_mcnemar_uses_discordant_complete_pairs():
    comparison = analysis.mcnemar(
        [False, False, False, True, None],
        [True, True, True, True, False],
    )
    assert comparison == {
        "pairs": 4,
        "left_only": 0,
        "right_only": 3,
        "left_minus_right": -0.75,
        "exact_p": 0.25,
    }


def test_multiple_comparison_adjustment_and_zero_event_bound():
    tests = [{"exact_p": value} for value in (0.04, 0.01, 0.03, 1)]
    analysis.apply_holm(tests)
    assert [test["holm_p"] for test in tests] == pytest.approx([0.09, 0.04, 0.09, 1])
    assert analysis.wilson(0, 80)[1] < 0.05
    assert analysis.wilson(0, 0) == (None, None)


def test_policy_applies_pass_booleans_and_preserves_failed_attempts():
    checks = dict.fromkeys(analysis.INCUMBENT, True)
    checks["expected_updates"] = False
    row = {"status": "ok", "detections": checks}
    assert analysis.decision(row, "seven") is False
    assert analysis.decision(row, "seven_plus_updates") is True
    assert analysis.decision({"status": "failed"}, "seven") is None


def test_raw_bundle_rejects_changed_records(tmp_path):
    design = tmp_path / "design.json"
    design.write_text(json.dumps({
        "status": "development", "faults": ["healthy"], "budgets_seconds": [0.025],
        "development_seed_range": [1, 1], "order_seed_offset": 123, "max_steps": 100,
    }))
    records = tmp_path / "records.jsonl"
    records.write_text(json.dumps({
        "workload": "digits", "phase": "development", "source_revision": "test",
        "seed": 1, "fault": "healthy", "budget_seconds": 0.025, "position": 0,
        "status": "failed",
    }) + "\n")
    (tmp_path / "manifest.json").write_text(json.dumps({
        "design": json.loads(design.read_bytes()), "design_sha256": analysis.digest(design),
        "records_sha256": analysis.digest(records), "workload": "digits", "phase": "development",
        "source_revision": "test", "record_count": 1, "failed_runs": 1,
    }))
    _, rows = analysis.validate(tmp_path, design, "digits")
    assert rows[(1, "healthy", 0.025)]["status"] == "failed"
    records.write_text(records.read_text() + "{}\n")
    with pytest.raises(ValueError, match="Manifest provenance"):
        analysis.validate(tmp_path, design, "digits")
