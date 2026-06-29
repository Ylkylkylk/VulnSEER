"""Compatibility entrypoint for the VulnSEER main pipeline.

The maintained implementation lives in run_multi_chain_pipeline.py.  This file
is kept only so older commands that referenced the one-file runner continue to
work without carrying a second copy of the framework logic.
"""

from run_multi_chain_pipeline import main


if __name__ == "__main__":
    main()
