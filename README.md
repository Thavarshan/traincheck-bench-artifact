# TrainCheck Bench Artifact

A CPU-only PyTorch benchmark for studying silent training faults in continuous integration. It includes the implementation, test suite, machine-readable study designs, and archived run records for Digits, Wine, and Fashion-MNIST.

[Download v0.1.0](https://github.com/Thavarshan/traincheck-bench-artifact/releases/tag/v0.1.0) · [Implementation checks](https://github.com/Thavarshan/traincheck-bench-artifact/actions/workflows/checks.yml) · [Cite this artifact](CITATION.cff)

## What the benchmark tests

The experiments compare optimizer call conformance and short-horizon loss checks under legitimate schedules and injected faults. The schedules include plain updates, gradient accumulation, and declared skips. Faults include missing optimizer updates, missing gradient resets, and unscaled accumulation. The archives cover both elapsed-time and fixed-minibatch budgets, healthy and faulty runs, and the conditions under which the checks alarmed.

The original runs, cutoff refits, and tests answer different questions. Use the archived manifests to identify the phase and revision behind a result. The tests verify implementation behavior, and the retrospective scripts reanalyze previously collected measurements.

## Install and inspect

Use Python 3.11 and the pinned CPU dependencies.

```sh
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements/ci-cpu.txt
python -m pip install --no-deps -e .
python -m pytest -q
```

Run `python -m traincheck --help` for the benchmark CLI. The standalone research scripts live under [docs/scripts](docs/scripts/), and their machine-readable designs live under [docs/research](docs/research/).

## Reanalyze the archived measurements

These commands read the existing records and require no model training.

```sh
python docs/scripts/analyze_score_separation.py --output build/score-separation
python docs/scripts/analyze_cutoff_resplits.py --output build/score-separation/resplits
```

| Evidence | Archive |
| --- | --- |
| Original workload evaluations | [Public-workload runs](results/public-study/) |
| Optimizer schedule and omission comparisons | [Schedule-study runs](results/schedule-study/) |
| Same-runner Digits calibration and GradScaler | [Matched-runner records](results/matched-calibration/) |
| Equal-update accuracy and harness timing | [Control records](results/update-control/) |
| Gradient-handling faults and Wine cross-entropy replay | [Gradient-fault runs](results/gradient-faults/) |
| Alternative cutoffs, ROC-AUC, and resplit sensitivity | [Retrospective results](results/retrospective-score-separation/) |

Raw records use JSONL. Study folders pair the records with manifests or freeze files that identify the configuration, source revision, runner, and available SHA-256 checksums. The experiment code fetches the public datasets from their respective sources.

## Releases and provenance

The [v0.1.0 release](https://github.com/Thavarshan/traincheck-bench-artifact/releases/tag/v0.1.0) attaches a results ZIP, provenance file, and SHA-256 checksums. Its tagged source archive also includes the runnable implementation. [SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json) records how this public checkout was extracted.

Archived manifests retain the original study's source revisions and hosted run IDs. The public repository has its own Git history. A new collection must freeze its executable inputs and any calibrated thresholds for that checkout. Dedicated collection workflows require manual dispatch. Routine GitHub Actions runs the test suite and reanalyzes the existing records.

## Contributing

Use [issues](https://github.com/Thavarshan/traincheck-bench-artifact/issues) for implementation bugs and questions about reproducing an archived result. Include the version or commit and, for archive questions, the run ID and result path. See [CONTRIBUTING.md](CONTRIBUTING.md) for the distinction between software fixes and new measurement records.
