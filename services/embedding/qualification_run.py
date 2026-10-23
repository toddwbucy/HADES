"""Standard-library evidence ownership shared by the opt-in qualification CLIs."""

from datetime import datetime, timezone
import json
import os
import signal
import subprocess
import sys


def error_record(error):
    """Keep bounded diagnostics without replacing the exception that caused failure."""
    return {"type": type(error).__name__, "message": str(error)[:2048]}


def attach_cleanup_errors(primary, errors):
    """Carry secondary cleanup failures to the evidence writer on the original exception."""
    previous = getattr(primary, "qualification_cleanup_errors", [])
    primary.qualification_cleanup_errors = [*previous, *errors]


def validate_environment(parser, args):
    """Reject online loading, ambiguous devices and invalid test binaries before outputs."""
    if any(os.environ.get(name) != "1" for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")):
        parser.error("set HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 before starting")
    if args.device != "cpu" and not (args.device.startswith("cuda:") and args.device[5:].isdigit()):
        parser.error("device must be cpu or cuda:<ordinal>")
    arangod = getattr(args, "arangod", None)
    if arangod and not arangod.is_file():
        parser.error("arangod must name an existing binary")


def require_device(device):
    """Check actual CUDA availability only after evidence ownership has begun."""
    import torch
    if device.startswith("cuda:") and (
        not torch.cuda.is_available() or int(device[5:]) >= torch.cuda.device_count()
    ):
        raise RuntimeError("requested CUDA device is unavailable; CPU fallback is refused")


def install_interrupt_handler():
    """Let SIGTERM unwind through the owned processes, model and evidence writer."""
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt("qualification interrupted")
    signal.signal(signal.SIGTERM, interrupted)


class QualificationRun:
    """Own a new evidence directory and provider through success, failure and interruption."""

    def __init__(self, output, repo, *, protocol=None):
        self.output = output.absolute()
        self.repo = repo
        self.provider = None
        self.interrupted = False
        self.report = {"passed": False, "stage": "preflight"}
        if protocol is not None:
            self.report["protocol"] = protocol

    def __enter__(self):
        self.previous_umask = os.umask(0o077)
        try:
            self.output.mkdir(mode=0o700, parents=True, exist_ok=False)
        except BaseException:
            os.umask(self.previous_umask)
            raise
        self.report["started_at"] = datetime.now(timezone.utc).isoformat()
        return self

    def record_implementation(self):
        """Probe the exact checkout under the same failure/report owner as model loading."""
        self.report["implementation_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=self.repo, text=True).strip()
        self.report["implementation_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"], cwd=self.repo, text=True).strip())

    def own(self, provider):
        """Register cleanup immediately after construction, before any provider checks."""
        self.provider = provider
        return provider

    def __exit__(self, _kind, error, _traceback):
        report = self.report
        if error is not None:
            report["passed"] = False
            self.interrupted = isinstance(error, KeyboardInterrupt)
            report["error"] = error_record(error)
            cleanup_errors = getattr(error, "qualification_cleanup_errors", [])
            if cleanup_errors:
                report["cleanup_errors"] = list(cleanup_errors)
            print(f"Qualification failed during {report['stage']}: {error}", file=sys.stderr)
        try:
            if self.provider is not None:
                try:
                    self.provider.unload()
                except BaseException as cleanup:
                    self.interrupted |= isinstance(cleanup, KeyboardInterrupt)
                    if report["passed"]:
                        report["stage"] = "cleanup"
                    report["passed"] = False
                    report["cleanup_error"] = error_record(cleanup)
                    report.setdefault("cleanup_errors", []).append(dict(error_record(cleanup), component="provider"))
            report["finished_at"] = datetime.now(timezone.utc).isoformat()
            (self.output / "report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            print(f"Qualification evidence: {self.output}", flush=True)
        finally:
            os.umask(self.previous_umask)
        return isinstance(error, (Exception, KeyboardInterrupt))

    @property
    def exit_code(self):
        return 130 if self.interrupted else (0 if self.report["passed"] else 1)
