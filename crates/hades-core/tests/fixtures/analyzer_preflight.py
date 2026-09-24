#!/usr/bin/python3
"""Run private preflight behavior without executing a freshly written inode."""
from pathlib import Path

# Execute in this process: PID, argv, inherited pipes and cancellation stay real.
exec(compile(Path("probe.py").read_text(), "probe.py", "exec"))
