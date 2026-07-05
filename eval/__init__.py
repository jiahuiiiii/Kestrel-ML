"""Eval + replay harness (ml_plan.md §6).

Fixtures are historical catalyst events with hand-labeled ground truth;
replay.py feeds them through the real pipeline and scores it. This package is
also the symposium demo source: the same fixtures drive Brandon's `--replay`
scheduler flag.
"""
