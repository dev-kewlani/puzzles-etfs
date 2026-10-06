"""etf66: walk forward research on a universe of 66 ETFs.

The package loads the data snapshot, builds labels and features, trains models in a walk forward, turns the
predictions into bins, simulates portfolios against benchmarks and logs every trial.
Data on or after the fence date (2021-01-01) is the holdout. Only the `holdout` command reads it, after the owner
writes HOLDOUT_UNLOCK.yaml. Read docs/DESIGN.md first.
"""
__version__ = "0.1.0"
