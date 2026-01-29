"""Destructive operation protection for HADES CLI.

This module provides safeguards for destructive operations like purge, delete,
and drop. Protection has two layers:

1. Environment variable gate (primary): HADES_DESTRUCTIVE_OPS must be set to
   "enabled" for destructive operations to work at all. This is the universal
   protection that works with any tool.

2. Interactive confirmation (secondary): When enabled, destructive operations
   require typing confirmation on stdin. This is designed specifically for
   Claude Code and similar AI agents that cannot provide interactive input.

   SECURITY NOTE: The interactive stdin confirmation is designed to prevent
   autonomous execution by Claude Code, which cannot provide interactive input.
   This is NOT a universal protection against all AI agents - other tools may
   handle stdin differently. The environment variable gate is the primary
   protection layer.
"""

from __future__ import annotations

import os
import sys

from core.cli.output import CLIResponse, ErrorCode, error_response

# Environment variable to enable destructive operations
ENV_VAR = "HADES_DESTRUCTIVE_OPS"
ENABLED_VALUE = "enabled"


def is_destructive_ops_enabled() -> bool:
    """Check if destructive operations are enabled via environment variable."""
    return os.environ.get(ENV_VAR, "").lower() == ENABLED_VALUE


def is_interactive() -> bool:
    """Check if stdin is a TTY (interactive terminal)."""
    return sys.stdin.isatty()


def require_confirmation(prompt: str) -> bool:
    """Require user to type exact confirmation text on stdin.

    This is designed to prevent autonomous execution by AI agents like Claude Code
    that cannot provide interactive input. The command will hang waiting for stdin,
    then timeout - forcing the user to run it manually.

    Note: This protection is specific to Claude Code and similar tools. Other AI
    agents may handle stdin differently and bypass this check.

    Args:
        prompt: The exact text the user must type to confirm

    Returns:
        True if user typed the correct confirmation, False otherwise
    """
    if not is_interactive():
        # Not a TTY - print message and return False
        # This will cause the command to fail, requiring manual execution
        print(
            "Interactive confirmation required. Run this command in a terminal.",
            file=sys.stderr,
        )
        return False

    try:
        print(f"Type '{prompt}' to confirm: ", end="", file=sys.stderr, flush=True)
        user_input = input().strip()
        return user_input == prompt
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.", file=sys.stderr)
        return False


def check_destructive_allowed(
    command: str,
    operation_desc: str,
    confirm_text: str,
    start_time: float,
    force: bool = False,
) -> CLIResponse | None:
    """Check if a destructive operation is allowed.

    Args:
        command: CLI command name for error response
        operation_desc: Human-readable description of the operation
        confirm_text: Text user must type to confirm (e.g., "PURGE 2409.04701")
        start_time: Start time for duration calculation
        force: If True, skip interactive confirmation (still requires env var)

    Returns:
        None if operation is allowed, CLIResponse with error if not allowed
    """
    # Check environment variable gate (primary protection)
    if not is_destructive_ops_enabled():
        return error_response(
            command=command,
            code=ErrorCode.DESTRUCTIVE_OP_DISABLED,
            message=(
                f"Destructive operations are disabled. "
                f"Set {ENV_VAR}={ENABLED_VALUE} to enable."
            ),
            details={
                "operation": operation_desc,
                "hint": f"export {ENV_VAR}={ENABLED_VALUE}",
            },
            start_time=start_time,
        )

    # Check interactive confirmation (secondary protection for Claude Code)
    # Skip if --force flag is used (for scripting)
    if not force:
        if not require_confirmation(confirm_text):
            return error_response(
                command=command,
                code=ErrorCode.DESTRUCTIVE_OP_DISABLED,
                message="Confirmation required. Operation aborted.",
                details={
                    "operation": operation_desc,
                    "expected_confirmation": confirm_text,
                    "note": (
                        "Interactive confirmation is designed to require human involvement "
                        "when using Claude Code. Other AI tools may behave differently."
                    ),
                },
                start_time=start_time,
            )

    # All checks passed
    return None
