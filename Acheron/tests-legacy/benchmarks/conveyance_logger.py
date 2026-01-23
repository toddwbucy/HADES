"""CLI wrapper for :mod:`core.logging.conveyance` utilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.logging.conveyance import (
    ConveyanceContext,
    TIME_UNITS,
    log_conveyance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Benchmark JSON path")
    parser.add_argument("--label", required=True, help="Label for this record")
    parser.add_argument(
        "--benchmark-key",
        default="get",
        help="Top-level key holding stats (e.g., get, insert, query)",
    )
    parser.add_argument(
        "--time-source",
        default="e2e",
        help="Stat bucket to read (e.g., e2e, ttfb)",
    )
    parser.add_argument(
        "--time-metric",
        default="p95",
        help="Metric within the bucket (e.g., avg, p95, p99)",
    )
    parser.add_argument(
        "--time-units",
        default="ms",
        choices=sorted(TIME_UNITS.keys()),
        help="Units of the metric (converted to seconds)",
    )
    parser.add_argument("--what", type=float, required=True, help="W factor (0-1)")
    parser.add_argument("--where", type=float, required=True, help="R factor (0-1)")
    parser.add_argument("--who", type=float, required=True, help="H factor (0-1)")
    parser.add_argument("--ctx-l", type=float, required=True, help="Context L component")
    parser.add_argument("--ctx-i", type=float, required=True, help="Context I component")
    parser.add_argument("--ctx-a", type=float, required=True, help="Context A component")
    parser.add_argument("--ctx-g", type=float, required=True, help="Context G component")
    parser.add_argument(
        "--weight-l",
        type=float,
        default=0.25,
        help="Weight for L component (defaults to 0.25)",
    )
    parser.add_argument(
        "--weight-i",
        type=float,
        default=0.25,
        help="Weight for I component (defaults to 0.25)",
    )
    parser.add_argument(
        "--weight-a",
        type=float,
        default=0.25,
        help="Weight for A component (defaults to 0.25)",
    )
    parser.add_argument(
        "--weight-g",
        type=float,
        default=0.25,
        help="Weight for G component (defaults to 0.25)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.7,
        help="Context amplification exponent (default 1.7)",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Optional free-form notes captured with the record",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSONL file to append the record to",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, do not append to file even when --output is provided",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    context = ConveyanceContext(
        L=args.ctx_l,
        I=args.ctx_i,
        A=args.ctx_a,
        G=args.ctx_g,
        weight_L=args.weight_l,
        weight_I=args.weight_i,
        weight_A=args.weight_a,
        weight_G=args.weight_g,
    )

    record = log_conveyance(
        input_path=args.input,
        label=args.label,
        benchmark_key=args.benchmark_key,
        time_source=args.time_source,
        time_metric=args.time_metric,
        time_units=args.time_units,
        what=args.what,
        where=args.where,
        who=args.who,
        context=context,
        alpha=args.alpha,
        notes=args.notes,
        output_path=args.output,
        dry_run=args.dry_run,
    )

    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
