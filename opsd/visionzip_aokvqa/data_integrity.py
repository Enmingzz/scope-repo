"""Fail-closed validation for decontaminated training-data manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_decontaminated_training_data(
    train_path: str | Path,
    manifest_path: str | Path,
    expected_rows: int = 10_000,
    minimum_benchmarks: int = 15,
) -> dict[str, Any]:
    train_path = Path(train_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_sha256 = sha256_file(train_path)

    audit_meta = manifest.get("independent_postbuild_audit", {})
    audit_path = Path(str(audit_meta.get("path", ""))).resolve()
    processor_meta = manifest.get("qwen_processor_token_check", {})
    processor_path = Path(str(processor_meta.get("path", ""))).resolve()
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else {}
    processor_check = (
        json.loads(processor_path.read_text(encoding="utf-8")) if processor_path.is_file() else {}
    )

    audit_zero_fields = (
        "strict_question_match_rows",
        "relaxed_question_match_rows",
        "encoded_image_match_rows",
        "pixel_image_match_rows",
        "perceptual_image_match_rows",
        "any_image_match_rows",
        "image_and_question_match_rows",
        "image_payload_errors",
    )
    audit_benchmarks = audit.get("benchmarks", [])
    audit_train_sets = audit.get("train_sets", [])
    manifest_snapshots = {
        row.get("benchmark"): row.get("file_sha256")
        for row in manifest.get("benchmark_decontamination", {}).get("benchmarks", [])
    }
    audit_snapshots = {
        row.get("benchmark"): row.get("file_sha256") for row in audit_benchmarks
    }
    caps = manifest.get("caps", {})
    checks = {
        "status": manifest.get("status") == "passed",
        "path": Path(str(manifest.get("output_jsonl", ""))).resolve() == train_path,
        "sha256": manifest.get("output_sha256") == actual_sha256,
        "rows": int(manifest.get("selected_rows", -1)) == expected_rows,
        "sample_ids": int(manifest.get("unique_sample_ids", -1)) == expected_rows,
        "source_ids": int(manifest.get("unique_source_ids", -1)) == expected_rows,
        "image_questions": int(manifest.get("unique_image_question_pairs", -1)) == expected_rows,
        "benchmark_snapshots": len(manifest_snapshots) >= minimum_benchmarks,
        "audit_metadata": (
            audit_meta.get("status") == "passed"
            and int(audit_meta.get("benchmark_count", 0)) >= minimum_benchmarks
            and all(
                int(audit_meta.get(field, -1)) == 0
                for field in (
                    "strict_question_matches",
                    "relaxed_question_matches",
                    "encoded_image_matches",
                    "decoded_pixel_matches",
                    "perceptual_image_matches",
                    "image_and_question_matches",
                    "image_payload_errors",
                    "missing_training_images",
                    "duplicate_sample_id_excess",
                    "duplicate_source_id_excess",
                    "duplicate_image_question_excess",
                )
            )
        ),
        "audit_file_sha256": (
            audit_path.is_file() and audit_meta.get("file_sha256") == sha256_file(audit_path)
        ),
        "audit_snapshot_hashes": (
            len(audit_snapshots) >= minimum_benchmarks and audit_snapshots == manifest_snapshots
        ),
        "audit_zero_overlap": (
            len(audit_benchmarks) >= minimum_benchmarks
            and all(
                int(row.get(field, -1)) == 0
                for row in audit_benchmarks
                for field in audit_zero_fields
            )
        ),
        "audit_train_integrity": (
            len(audit_train_sets) == 1
            and Path(str(audit_train_sets[0].get("path", ""))).resolve() == train_path
            and audit_train_sets[0].get("file_sha256") == actual_sha256
            and int(audit_train_sets[0].get("missing_image_count", -1)) == 0
            and int(audit_train_sets[0].get("duplicate_id_excess", -1)) == 0
            and int(audit_train_sets[0].get("duplicate_source_id_excess", -1)) == 0
            and int(audit_train_sets[0].get("duplicate_image_question_excess", -1)) == 0
        ),
        "processor_metadata": (
            processor_meta.get("status") == "passed"
            and int(processor_meta.get("checked_samples", 0)) >= 256
            and int(processor_meta.get("mismatch_count", -1)) == 0
        ),
        "processor_check_file_sha256": (
            processor_path.is_file()
            and processor_meta.get("file_sha256") == sha256_file(processor_path)
        ),
        "processor_check": (
            processor_check.get("status") == "passed"
            and Path(str(processor_check.get("dataset", ""))).resolve() == train_path
            and int(processor_check.get("checked_samples", 0)) >= 256
            and int(processor_check.get("mismatch_count", -1)) == 0
            and int(processor_check.get("max_actual_image_tokens", -1))
            <= int(caps.get("max_prompt_image_tokens", -1))
            and int(processor_check.get("min_pixels", -1)) == int(caps.get("min_pixels", -2))
            and int(processor_check.get("max_pixels", -1)) == int(caps.get("max_pixels", -2))
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Training-data decontamination manifest failed: {checks}")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "train_path": str(train_path),
        "train_sha256": actual_sha256,
        "audit_path": str(audit_path),
        "audit_sha256": sha256_file(audit_path),
        "processor_check_path": str(processor_path),
        "processor_check_sha256": sha256_file(processor_path),
        "checks": checks,
    }
