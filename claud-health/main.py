"""
Healthcare PPML / DP — treatments dataset (binary favorable outcome).

CLI:  python3 claud-health/main.py
Web:  cd claud-health && python3 app.py   (default port 5050)
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
