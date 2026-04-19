from __future__ import annotations

import argparse

import notebook_support as ns


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract partial benchmark tables and figures from a saved run session. "
            "By default this reads the latest running session and writes the snapshot into that session folder."
        )
    )
    parser.add_argument(
        "--run",
        help="Run name or session id. Defaults to the latest running session.",
    )
    parser.add_argument(
        "--thesis-pack",
        action="store_true",
        help="Also export the thesis-ready figure pack for the current partial snapshot.",
    )
    parser.add_argument(
        "--skip-algorithm-sections",
        action="store_true",
        help="Skip per-algorithm tables, panels, and deep dives.",
    )
    parser.add_argument(
        "--write-global",
        action="store_true",
        help="Write into results/ instead of the session-specific run_sessions/<id>/ folder.",
    )
    parser.add_argument(
        "--context-points",
        type=int,
        default=1200,
        help="Context points to show around deep-dive anomalies.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    payload = ns.export_saved_run_snapshot_artifacts(
        args.run,
        include_algorithm_sections=not args.skip_algorithm_sections,
        include_thesis_pack=args.thesis_pack,
        write_global=args.write_global,
        context_points=args.context_points,
    )

    print(f"Run: {payload['run_name']} [{payload['session_id']}]")
    print(
        "Coverage: "
        f"{payload['completed_dataset_count']}/{payload['selected_dataset_count']} datasets "
        f"(pending {payload['pending_dataset_count']})"
    )
    print(f"Partial snapshot: {payload['is_partial']}")
    print(f"Artifacts: {payload['artifact_root']}")
    if payload["thesis_payload"] is not None:
        print(f"Thesis catalog: {payload['thesis_payload']['catalog_path']}")
        print(f"Thesis captions: {payload['thesis_payload']['captions_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
