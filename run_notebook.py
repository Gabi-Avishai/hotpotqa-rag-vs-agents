"""Execute a notebook cell by cell, saving after every cell.

``jupyter nbconvert --execute`` writes nothing until the whole notebook has run,
which loses every result when a late cell fails and gives no sign of progress on
the phases that take minutes to encode a corpus. This runner keeps one kernel
open, writes the notebook back to disk after each cell, and prints a line per
cell so a long run can be watched.

There is deliberately no way to start part-way through: saved outputs do not
restore Python state, so a notebook is always executed from its first cell.
Expensive intermediate results (indexes, embeddings, runs) are cached to disk by
the notebooks themselves, which is what makes a re-run cheap.

Usage:
    python run_notebook.py 00_eda.ipynb [--timeout 3600]
"""

from __future__ import annotations

import argparse
import os
import time

# Library progress bars and advisory warnings are not results; keep them out of the
# saved notebooks. The kernel inherits these from this process.
QUIET_ENV = {
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "HF_HUB_VERBOSITY": "error",
    "TRANSFORMERS_VERBOSITY": "error",
    "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
    "TQDM_DISABLE": "1",
    "PYTHONWARNINGS": "ignore:IProgress not found",   # tqdm's notebook-widget notice only
}

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


def run(path: str, timeout: int = 3600) -> int:
    os.environ.update(QUIET_ENV)
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3", allow_errors=False)
    failures = 0
    with client.setup_kernel():
        for index, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            started = time.time()
            try:
                client.execute_cell(cell, index)
                status = "ok"
            except CellExecutionError as err:
                status = f"FAILED: {str(err).splitlines()[-1][:120]}"
                failures += 1
            nbformat.write(nb, path)
            print(f"[{index:3d}] {time.time() - started:6.1f}s  {status}", flush=True)
            if failures:
                break
    return failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook")
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    raise SystemExit(run(args.notebook, args.timeout))
