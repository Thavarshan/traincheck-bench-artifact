import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch

from traincheck.scaler_probe import probe

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs/scripts"))
import analyze_schedule_study as analysis
import run_schedule_study as runner
from analyze_public_workload import apply_holm, mcnemar
from matched_calibration import HOST_FIELDS, verify_host


@pytest.mark.parametrize("raw", ["{", "[]", "{}", '{"seed": 1}'])
def test_rejects_malformed_records(tmp_path, raw):
    path = tmp_path / "records.jsonl"
    path.write_text(raw)
    with pytest.raises(ValueError, match="records.jsonl:1"):
        analysis.load_records(path)


def test_rejects_digest_mismatch_and_missing_combinations(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    design = {"workloads": ["digits"], "development_seeds": [1, 1], "budgets": {"wall25": {"seconds": .025}}}
    config = tmp_path / "config.json"
    config.write_text(json.dumps(design))
    workload = tmp_path / "schedule-digits"
    workload.mkdir()
    records = workload / "records.jsonl"
    records.write_text("")
    manifest = dict(design=design, config_sha256=analysis.digest(config), phase="development",
                    workload="digits", source_revision="fixture", record_count=0, failures=0,
                    records_sha256="invalid")
    path = workload / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="invalid manifest"):
        analysis.analyze(tmp_path, tmp_path / "output", config, "development")
    manifest["records_sha256"] = analysis.digest(records)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unexpected run grid"):
        analysis.analyze(tmp_path, tmp_path / "output", config, "development")


def test_runner_preserves_failed_attempts(tmp_path, monkeypatch):
    from types import SimpleNamespace

    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "development_seeds": [1, 1], "budgets": {"wall25": {"seconds": .025}},
    }))
    monkeypatch.setattr(runner.torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(runner.torch, "set_num_interop_threads", lambda _: None)
    monkeypatch.setattr(runner, "load_schedule_workload",
                        lambda *_: SimpleNamespace(train_y=[1], validation_y=[1], test_y=[]))
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *_, **__: "fixture")

    def fail(*_):
        raise RuntimeError("intentional fixture failure")

    monkeypatch.setattr(runner, "measure_schedule", fail)
    out = tmp_path / "workload"
    runner.collect("digits", False, tmp_path, out, config)
    manifest = json.loads((out / "manifest.json").read_bytes())
    rows = analysis.load_records(out / "records.jsonl")
    assert manifest["failures"] == manifest["record_count"] == len(rows) == 3
    assert manifest["records_sha256"] == hashlib.sha256((out / "records.jsonl").read_bytes()).hexdigest()
    assert all(row["status"] == "failed" and "intentional fixture failure" in row["error"] for row in rows)
    assert all(value is None for value in analysis.decisions(rows[0], .98).values())


@pytest.mark.parametrize("groups,target", [([], .05), ([[]], .05), ([[.7, float("nan")]], .5),
                                         ([[.7]], .05), ([[.7]], 0)])
def test_rejects_invalid_calibration_groups(groups, target):
    with pytest.raises(ValueError):
        analysis.calibration(groups, target)


def test_miss_fields_are_only_for_realized_faults():
    scores = [(0.14, False, False), (0.07, False, True), (0.12, True, True)]
    faulty = analysis.quality_summary(scores, healthy=False)
    assert (faulty["miss_loss_2pp"], faulty["miss_loss_5pp"], faulty["miss_loss_10pp"]) == (1, 1, 0)
    assert faulty["quality_pairs_with_omission"] == 2
    healthy = analysis.quality_summary([(0.14, False, False)], healthy=True)
    assert healthy["median_legitimate_accuracy_drop"] == 0.14
    assert healthy["miss_loss_5pp"] is None
    assert healthy["median_accuracy_drop"] is None


def test_known_discordances_and_holm_values():
    result = mcnemar([True]*8 + [False]*2 + [None], [False]*8 + [True]*2 + [True])
    assert result["pairs"] == 10
    assert result["left_only"] == 8
    assert result["right_only"] == 2
    assert result["exact_p"] == .109375
    tests = [{"exact_p": p} for p in (.04, .01, .03)]
    apply_holm(tests)
    assert [row["holm_p"] for row in tests] == pytest.approx([.06, .03, .06])


def test_matched_host_verifies_boot_and_cpu_affinity():
    before = {key: "fixture" for key in HOST_FIELDS}
    before["cpu_affinity"] = [2]
    verify_host(before, before.copy())
    with pytest.raises(ValueError, match="runner_boot_sha256"):
        verify_host(before, before | {"runner_boot_sha256": "different"})
    with pytest.raises(ValueError, match="cpu_affinity"):
        verify_host(before, before | {"cpu_affinity": [3]})


def test_real_scaler_overflow_and_scheduler_cadence():
    torch.set_num_threads(1)
    overflow, omission, ungated = (probe(12, name)[0] for name in ("overflow", "omission", "ungated_scheduler"))
    assert not overflow["finite_gradients"] and overflow["scale_after"] < overflow["scale_before"]
    assert overflow["optimizer_calls"] == overflow["scheduler_calls"] == 0
    assert overflow["naive_alarm"] and not overflow["optimizer_mismatch"]
    assert omission["finite_gradients"] and omission["optimizer_mismatch"]
    assert ungated["optimizer_calls"] == 0 and ungated["scheduler_calls"] == 1
    assert ungated["scheduler_mismatch"] and ungated["scheduler_warnings"]
