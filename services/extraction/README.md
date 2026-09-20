# Extraction limits

The LaTeX backend accepts plain `.tex`, plain gzip, and gzip-compressed tar input.
Plain/gzipped source is limited to 200 MiB of bytes before UTF-8 decoding; this
includes all concatenated gzip members. Invalid UTF-8 retains the existing
replacement-character behavior.

Tar input is first decompressed into a private temporary file with a 201 MiB
limit covering payload, headers, and padding. Enumeration then allows at most
500 members, including rejected entries, and at most 200 MiB of accepted regular
file payload. Links, special files, and unsafe paths are excluded; duplicate
normalized regular-file paths fail explicitly. Only the largest accepted regular
`.tex` member is read. Archive paths, ownership, and permissions are never
materialized. Temporary files are removed on success or failure.

Limit failures return an extraction error with no partial text. These are input
and expansion controls, not process RSS or parsing-time guarantees. Decoded text,
regular-expression parsing, and concurrent requests consume additional resources;
Docling/model-backed formats have separate resource requirements.

Run the CPU-only fixtures with:

```bash
python -m pytest services/tests/test_latex_limits.py
```

Raw tar headers are checked before PAX/GNU metadata parsing on every supported
Python version: at most 1 MiB per extended header and 4 MiB total metadata.
The 500-header budget includes extended headers, even those hidden by tarfile
iteration. Oversized declared lengths are rejected before payload allocation.
