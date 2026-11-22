# Contributing to TrainCheck Bench

For a software change, identify the entry point it affects and run the test suite under the pinned CPU environment described in the [README](README.md). GitHub Actions also checks that the existing archives can be reanalyzed.

For a result discrepancy, include the release or commit, study and run ID, record or manifest path, and the command that produced the discrepancy. Include the actual and expected outputs.

Keep archived measurements and checksums with their original provenance. Record new collections separately with the experiment configuration, seed, runner, source revision, dependency versions, and outcome. Keep exploratory and frozen evaluations identifiable in the records.
