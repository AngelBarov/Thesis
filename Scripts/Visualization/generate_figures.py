"""Regenerate every results mock-up into the Visualization folder.

Run from anywhere:

    python "Scripts/Visualization/generate_figures.py"
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import game_figures
import hmm_figures
import profile_figures


def main() -> None:
    game_figures.main()
    hmm_figures.main()
    profile_figures.main()


if __name__ == "__main__":
    main()
