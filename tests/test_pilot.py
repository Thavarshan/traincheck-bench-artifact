"""Tests for the budget-exposure summary, not detector effectiveness."""

from docs.scripts.summarize_pilot import summarize


def row(budget, steps, cap=False):
    return {
        "phase": "exploratory", "fault": "healthy", "seed": 101,
        "budget_seconds": budget, "status": "ok", "steps": steps,
        "elapsed_seconds": budget, "diagnose_wall_seconds": budget + 0.02,
        "termination_reason": "max_steps" if cap else "budget_reached",
    }


def test_pilot_reports_matched_step_exposure():
    report = summarize([row(0.025, 7), row(0.125, 25)])
    assert "Seed 101: 7 to 25 steps" in report
    assert "not confirmatory evidence" in report


def test_pilot_warns_when_budgets_do_not_bind():
    report = summarize([row(0.025, 120, True), row(0.125, 120, True)])
    assert "no paired seed gained steps" in report


def test_three_budget_levels_report_adjacent_pairs():
    report = summarize([row(0.025, 7), row(0.125, 25), row(0.25, 48)])
    assert "0.025 s to 0.125 s" in report
    assert "0.125 s to 0.25 s" in report
    assert "Seed 101: 25 to 48 steps" in report
