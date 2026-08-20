"""Run BANKSY spatial-domain detection in a separate interpreter.

BANKSY needs its own environment, so it runs as a subprocess. The original
implementation hardcoded ``/home/vlad/xenium_mundus/...`` and interpolated raw
paths into single-quoted Python string literals - a SyntaxError (or injection)
waiting to happen, with no default timeout.

This version:

* takes the interpreter and source directory from :class:`PipelineConfig`
  (env-driven), so it runs on any machine;
* passes every path through ``sys.argv`` - never interpolated into source - so a
  path containing quotes/newlines cannot break or inject into the child script;
* detects success via the child's **exit code** (and the presence of the results
  file), not by scanning stdout for a magic substring;
* always applies a timeout when one is configured.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ..config import PipelineConfig
from ..logging_setup import get_logger

# The child reads all of its inputs from argv; nothing is interpolated.
_CHILD_SCRIPT = """
import sys
sys.path.insert(0, sys.argv[1])
from pathlib import Path
from run_banksy_all import run_banksy_on_sample

run_banksy_on_sample(Path(sys.argv[2]), top_genes=int(sys.argv[3]))
"""


class BanksyRunner:
    """Invoke BANKSY on a sample's ``processed.zarr`` via a child interpreter."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.logger = get_logger(__name__)

    def run(self, base_dir: Path, zarr_path: Path) -> bool:
        """Run BANKSY; return True only on a clean, verified success."""
        if not self.config.banksy_available:
            self.logger.warning(
                "BANKSY interpreter/dir not configured or missing "
                "(set XENIUM_BANKSY_PYTHON / XENIUM_BANKSY_DIR); skipping BANKSY"
            )
            return False
        if not zarr_path.exists():
            self.logger.error("Zarr not found at %s; cannot run BANKSY", zarr_path)
            return False

        banksy_python = self.config.banksy_python
        banksy_dir = self.config.banksy_dir
        self.logger.info("Launching BANKSY subprocess (may take several minutes)")
        args = [
            str(banksy_python),
            "-c",
            _CHILD_SCRIPT,
            str(banksy_dir),
            str(base_dir),
            str(self.config.banksy_top_genes),
        ]
        try:
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                cwd=str(banksy_dir),
                timeout=self.config.banksy_timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            self.logger.error(
                "BANKSY timed out after %s seconds", self.config.banksy_timeout_seconds
            )
            return False

        if result.returncode != 0:
            self.logger.error("BANKSY subprocess failed (exit %d)", result.returncode)
            self.logger.error("STDOUT: %s", result.stdout)
            self.logger.error("STDERR: %s", result.stderr)
            return False

        results_file = base_dir / "banksy_results.json"
        if not results_file.exists():
            self.logger.error("BANKSY exited 0 but produced no %s", results_file.name)
            self.logger.error("STDOUT: %s", result.stdout)
            self.logger.error("STDERR: %s", result.stderr)
            return False

        try:
            n_domains = json.loads(results_file.read_text()).get("n_domains", "unknown")
            self.logger.info("BANKSY found %s spatial domains", n_domains)
        except (OSError, json.JSONDecodeError) as exc:
            self.logger.warning("Could not parse %s: %s", results_file.name, exc)
        return True
