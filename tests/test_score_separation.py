import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs/scripts"))
from analyze_score_separation import paired_auc, rank_auc, schedule_cutoffs, schedule_operating_point


@pytest.mark.parametrize(("pairs", "expected"), [
    ([(0, 2), (1, 3)], 1.0),
    ([(3, 0), (2, 1)], 0.0),
    ([(0, 1), (2, 2)], 0.625),
])
def test_auc_counts_cross_seed_pairs_and_half_ties(pairs, expected):
    assert rank_auc(pairs) == expected


def test_bootstrap_keeps_identical_scores_together():
    assert paired_auc([(0.4, 0.4), (0.7, 0.7), (0.8, 0.8)]) == (0.5, 0.5, 0.5)


def test_calibration_keeps_schedules_separate():
    groups = [list(range(offset, offset + 40)) for offset in (10, 0, 30)]
    assert schedule_cutoffs(groups, 0.05) == {"plain": 48, "accumulate2": 38, "declared_skip4": 68}


def test_operating_point_keeps_unrealized_assignments_out_of_sensitivity():
    cell = {
        1: {"loss_score": 0.8, "injected_omissions": 0},
        2: {"loss_score": 0.6, "injected_omissions": 1},
        3: {"loss_score": 0.7, "injected_omissions": 1},
    }
    result = schedule_operating_point(cell, 0.6)
    assert (result["n"], result["alarms"]) == (3, 2)
    assert (result["realized"], result["realized_alarms"]) == (2, 1)
