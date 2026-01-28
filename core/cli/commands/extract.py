"""Extract command — standalone document extraction.

Usage:
    hades extract document.pdf
    hades extract document.pdf --format text
    hades extract document.pdf --output extracted.json
"""

from __future__ import annotations

import json
from pathlib import Path

from core.cli.output import (
    CLIResponse,
    ErrorCode,
    error_response,
    progress,
    success_response,
)


def extract_file(
    file_path: str,
    output_format: str,
    output_path: str | None,
    start_time: float,
) -> CLIResponse:
    """Extract structured text from a document file.

    Args:
        file_path: Path to the document file.
        output_format: Output format ('json' or 'text').
        output_path: Optional output file path.
        start_time: Command start timestamp.

    Returns:
        CLIResponse with extraction result or error.
    """
    path = Path(file_path)

    if not path.exists():
        return error_response(
            command="extract",
            code=ErrorCode.FILE_NOT_FOUND,
            message=f"File not found: {file_path}",
            start_time=start_time,
        )

    progress(f"Extracting {path.name}...")

    try:
        from core.tools.extract import extract_document

        result = extract_document(
            path,
            extract_tables=True,
            extract_equations=True,
            extract_images=True,
        )

        if output_format == "text":
            output = result["text"]
        else:
            output = result

        # Write to file if requested
        if output_path:
            out = Path(output_path)
            if output_format == "text":
                out.write_text(output)
            else:
                out.write_text(json.dumps(output, indent=2))
            progress(f"Written to {output_path}")

        return success_response(
            command="extract",
            data={"result": output if output_format == "json" else {"text": output}},
            start_time=start_time,
            source_file=str(path),
            format=output_format,
            extraction_time=result.get("extraction_time", 0),
            tables=len(result.get("tables", [])),
            equations=len(result.get("equations", [])),
            images=len(result.get("images", [])),
        )

    except ImportError as e:
        return error_response(
            command="extract",
            code=ErrorCode.CONFIG_ERROR,
            message=f"Extraction dependencies not available: {e}",
            start_time=start_time,
        )
    except Exception as e:
        return error_response(
            command="extract",
            code=ErrorCode.PROCESSING_FAILED,
            message=str(e),
            start_time=start_time,
        )
