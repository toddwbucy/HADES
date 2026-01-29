"""CLI command decorators for HADES.

Provides reusable decorators to reduce boilerplate in CLI commands.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import Any, TypeVar

import typer

from core.cli.output import (
    CLIResponse,
    ErrorCode,
    error_response,
    print_response,
)

F = TypeVar("F", bound=Callable[..., Any])


def cli_command(
    command_name: str,
    error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
) -> Callable[[F], F]:
    """Decorator that wraps CLI commands with standard error handling.

    Provides:
    - Automatic timing via start_time injection
    - Standard try/except pattern
    - JSON error response formatting
    - Proper exit codes

    Args:
        command_name: The command identifier for error responses (e.g., "arxiv.search")
        error_code: Default error code for uncaught exceptions

    Usage:
        @arxiv_app.command("search")
        @cli_command("arxiv.search", ErrorCode.SEARCH_FAILED)
        def arxiv_search(query: str, ..., start_time: float) -> CLIResponse:
            # Just implement business logic, return CLIResponse
            from core.cli.commands.arxiv import search_arxiv
            return search_arxiv(query, max_results, categories, start_time)

    The decorated function should:
    - Accept start_time as its last parameter (injected automatically)
    - Return a CLIResponse object
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> None:
            start_time = time.time()

            try:
                # Inject start_time into kwargs
                response = func(*args, start_time=start_time, **kwargs)

                # Handle the response
                if isinstance(response, CLIResponse):
                    print_response(response)
                    if not response.success:
                        raise typer.Exit(1) from None
                else:
                    # If function returned something else, just exit normally
                    pass

            except typer.Exit:
                raise
            except Exception as e:
                response = error_response(
                    command=command_name,
                    code=error_code,
                    message=str(e),
                    start_time=start_time,
                )
                print_response(response)
                raise typer.Exit(1) from None

        return wrapper  # type: ignore[return-value]

    return decorator
