# TrainCheck Bench implementation

CPU benchmark for detecting training faults, including missing optimizer updates, missing gradient resets, and unscaled gradient accumulation. This repository contains the implementation, test suite, machine-readable study designs, and archived measurements.

## Install and inspect

Use Python 3.11 and the pinned CPU dependencies.

```sh
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements/ci-cpu.txt
python -m pip install --no-deps -e .
python -m pytest -q
```

## Reanalyze the archived measurements

The following commands use existing records and require no model training.

```sh
python docs/scripts/analyze_score_separation.py --output build/score-separation
python docs/scripts/analyze_cutoff_resplits.py --output build/score-separation/resplits
```

| Evidence | Archive |
| --- | --- |
| Original workloads | [Public-workload runs](results/public-study/) |
| Optimizer schedule and omissions | [Schedule-study runs](results/schedule-study/) |
| Matched Digits calibration and GradScaler | [Matched-runner records](results/matched-calibration/) |
| Equal-update and timing controls | [Control records](results/update-control/) |
| Gradient-handling faults and Wine replay | [Gradient-fault runs](results/gradient-faults/) |
| Alternative cutoffs and uncertainty | [Retrospective results](results/retrospective-score-separation/) |

Archived manifests contain original source revisions and workflow IDs. These refer to the historical study and serve as provenance for the retained records. The public checkout uses its own history. New experimental collections must freeze the executable inputs and calibration for that checkout. The public workflows include manually dispatched research jobs and an automated verification workflow.

The datasets are obtained by the experiment code from their respective publishers. The archived results contain run metrics and analyses.
