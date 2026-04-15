"""
Privacy-preserving ML vs standard ML — Bank Marketing (tabular, binary y).

CLI entry:  python3 claude/main.py
Web UI:     python -m claude.app   # see claude/app.py
"""
from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import experiment_runner as er


def main() -> None:
    er.run_experiment(verbose=True)


if __name__ == "__main__":
    main()
