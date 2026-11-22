import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs/scripts"))
from analyze_cutoff_resplits import refit


def test_refit_fits_on_healthy_blocks_and_excludes_matching_fault_seeds():
    healthy = {
        "plain": {1: 1.0, 2: 2.0, 3: 3.0, 4: 9.0, 5: 0.0},
        "accumulate2": {1: 2.0, 2: 3.0, 3: 4.0, 4: 8.0, 5: 0.0},
        "declared_skip4": {1: 0.0, 2: 4.0, 3: 5.0, 4: 7.0, 5: 0.0},
    }
    fault = {1: 10.0, 3: 10.0, 5: 1.0}
    result = refit(healthy, fault, {1, 2, 3}, 0.25)
    assert result["plain"] == {"cutoff": 3.0, "alarms": 1, "n": 2}
    assert result["accumulate2"] == {"cutoff": 4.0, "alarms": 1, "n": 2}
    assert result["declared_skip4"] == {"cutoff": 5.0, "alarms": 1, "n": 2}
    assert result["periodic4"] == {"cutoff": 3.0, "alarms": 0, "n": 1}
