from traincheck.experiment import FAULTS, diagnose


def test_healthy_run_has_gradient_and_updates():
    result = diagnose(seed=7, budget_seconds=0.2, max_steps=20)
    assert result.device == "cpu"
    assert result.gradient_nonzero
    assert result.weights_updated
    assert result.steps > 0


def test_skip_step_does_not_update_weights():
    result = diagnose(seed=7, fault="skip_step",
                      budget_seconds=0.2, max_steps=20)
    assert not result.weights_updated
    assert result.detected


def test_detached_gradient_is_detected():
    result = diagnose(seed=7, fault="detached_gradients",
                      budget_seconds=0.2, max_steps=20)
    assert not result.gradient_nonzero
    assert not result.weights_updated


def test_all_faults_are_runnable():
    for fault in FAULTS:
        assert diagnose(seed=2, fault=fault,
                        budget_seconds=0.05, max_steps=2).steps == 2


def test_preprocessing_fault_changes_observed_inputs_and_serving_loss():
    healthy = diagnose(seed=7, budget_seconds=0.2, max_steps=20)
    faulty = diagnose(seed=7, fault="preprocessing_mismatch",
                      budget_seconds=0.2, max_steps=20)
    assert healthy.detections["preprocessing_contract"]
    assert healthy.preprocessing_max_abs_delta == 0.0
    assert not faulty.detections["preprocessing_contract"]
    assert abs(faulty.preprocessing_max_abs_delta - 0.75) < 1e-6
    assert faulty.serving_loss != faulty.final_loss


def test_fixed_batch_reference_loss_uses_same_inputs_and_mode():
    untrained = diagnose(seed=7, budget_seconds=0.2, max_steps=0)
    assert untrained.initial_loss == untrained.final_loss


def test_training_mode_is_measured_during_training():
    healthy = diagnose(seed=7, budget_seconds=0.2, max_steps=5)
    faulty = diagnose(seed=7, fault="train_eval_mode",
                      budget_seconds=0.2, max_steps=5)
    assert healthy.detections["training_mode"]
    assert not faulty.detections["training_mode"]


def test_diagnose_wall_time_is_distinct_from_loop_time():
    result = diagnose(seed=7, budget_seconds=0.05, max_steps=20)
    assert result.diagnose_wall_seconds >= result.elapsed_seconds >= 0
    assert result.configuration["max_steps"] == 20


def test_exploratory_study_passes_through_explicit_step_ceiling(tmp_path):
    import json
    from traincheck.cli import main

    output = tmp_path / "exploratory.jsonl"
    assert main([
        "study", "--phase", "exploratory", "--seeds", "301",
        "--faults", "healthy", "--budgets", "0.05", "--max-steps", "3",
        "--output", str(output),
    ]) == 0
    record = json.loads(output.read_text().strip())
    assert record["configuration"]["max_steps"] == 3
    assert record["steps"] == 3
    assert record["diagnose_wall_seconds"] >= record["elapsed_seconds"]


def test_optimizer_step_calls_are_not_loop_iterations():
    healthy = diagnose(seed=7, budget_seconds=0.25, max_steps=40)
    missing = diagnose(seed=7, fault="skip_step",
                       budget_seconds=0.25, max_steps=40)
    intermittent = diagnose(
        seed=7, fault="intermittent_skip_step", budget_seconds=0.25, max_steps=40)
    detached = diagnose(seed=7, fault="detached_gradients",
                        budget_seconds=0.25, max_steps=40)
    assert all(run.steps == 40 for run in (
        healthy, missing, intermittent, detached))
    assert healthy.optimizer_step_calls == 40
    assert missing.optimizer_step_calls == 0
    assert intermittent.optimizer_step_calls == 30
    assert detached.optimizer_step_calls == 40
    assert not detached.weights_updated
    assert intermittent.weights_updated
    assert intermittent.gradient_nonzero


def test_budget_overshoot_is_recorded_as_continuous_nonnegative_amount():
    run = diagnose(seed=7, budget_seconds=0.025, max_steps=1000)
    assert run.budget_overshoot_seconds >= 0.0
    assert abs(run.budget_overshoot_seconds - max(
        0.0, run.elapsed_seconds - run.budget_seconds
    )) < 1e-10
