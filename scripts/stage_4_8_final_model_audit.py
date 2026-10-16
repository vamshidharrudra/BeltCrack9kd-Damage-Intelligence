
from pathlib import Path
import json
import time

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "audit_outputs" / "stage_4_8_final_model_audit"

MANIFEST = ROOT / "canonical_dataset" / "canonical_manifest.csv"
CHECKPOINT = (
    ROOT / "models" / "segmentation"
    / "beltcrack9kd_cpu_pilot_best.pt"
)

ARTIFACTS = {
    "canonical_manifest": MANIFEST,
    "segmentation_checkpoint": CHECKPOINT,
    "mask_generation_summary": (
        ROOT / "audit_outputs" / "stage_4_5_crack_masks"
        / "stage_4_5_mask_generation_summary.json"
    ),
    "pilot_training_history": (
        ROOT / "audit_outputs" / "stage_4_6b_cpu_pilot"
        / "stage_4_6b_pilot_history.csv"
    ),
    "pilot_training_summary": (
        ROOT / "audit_outputs" / "stage_4_6b_cpu_pilot"
        / "stage_4_6b_pilot_summary.json"
    ),
    "corrected_validation_thresholds": (
        ROOT / "audit_outputs"
        / "stage_4_6e_corrected_validation_diagnostic"
        / "stage_4_6e_corrected_threshold_results.csv"
    ),
    "qualitative_inspection_metrics": (
        ROOT / "audit_outputs" / "stage_4_6f_qualitative_inspection"
        / "stage_4_6f_sample_metrics.csv"
    ),
    "final_test_summary": (
        ROOT / "audit_outputs" / "stage_4_7_final_test"
        / "stage_4_7_test_summary.json"
    ),
    "final_test_per_image_metrics": (
        ROOT / "audit_outputs" / "stage_4_7_final_test"
        / "stage_4_7_test_image_metrics.csv"
    ),
}

OPTIONAL_ARTIFACTS = {
    "crack_detection_audit": (
        ROOT / "audit_outputs" / "stage_4_4_crack_detection_audit"
    ),
    "sequential_video_outputs": (
        ROOT / "audit_outputs" / "stage_4_2_sequential_video"
    ),
    "maintenance_intelligence_outputs": (
        ROOT / "audit_outputs" / "stage_4_3_maintenance_intelligence"
    ),
    "offline_application_outputs": (
        ROOT / "audit_outputs" / "stage_4_0_offline_application"
    ),
}


def read_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def main():
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("STAGE 4.8 - FINAL MODEL AUDIT")
    print("=" * 68)
    print("Training: NO")
    print("Model modification: NO")
    print("New test inference: NO")
    print()

    artifact_results = {}

    for name, path in ARTIFACTS.items():
        exists = path.is_file()
        artifact_results[name] = {
            "path": str(path),
            "exists": exists,
            "size_bytes": path.stat().st_size if exists else None,
        }
        status = "FOUND" if exists else "MISSING"
        print(f"[{status}] {name}: {path}")

    optional_results = {}
    print()
    print("OPTIONAL PRIOR-STAGE OUTPUTS")

    for name, path in OPTIONAL_ARTIFACTS.items():
        exists = path.exists()
        optional_results[name] = {
            "path": str(path),
            "exists": exists,
        }
        status = "FOUND" if exists else "NOT FOUND"
        print(f"[{status}] {name}: {path}")

    manifest_summary = None
    if MANIFEST.is_file():
        manifest = pd.read_csv(MANIFEST)
        required_columns = {
            "split",
            "sequence",
            "canonical_image_filename",
        }
        missing_columns = sorted(required_columns - set(manifest.columns))

        manifest_summary = {
            "rows": int(len(manifest)),
            "columns": list(manifest.columns),
            "missing_required_columns": missing_columns,
            "split_counts": {
                str(key): int(value)
                for key, value in manifest["split"].value_counts().items()
            } if "split" in manifest.columns else {},
        }

    test_summary = None
    test_summary_path = ARTIFACTS["final_test_summary"]

    if test_summary_path.is_file():
        try:
            test_summary = read_json(test_summary_path)
        except Exception as exc:
            test_summary = {
                "read_error": str(exc),
            }

    test_metrics = {}
    if isinstance(test_summary, dict):
        metric_keys = [
            "evaluation_split",
            "test_records",
            "resolved_pairs",
            "training_performed",
            "threshold_tuning_performed",
            "threshold",
            "checkpoint_best_epoch",
            "mean_image_dice",
            "mean_image_precision",
            "mean_image_recall",
            "global_pixel_dice",
            "global_pixel_precision",
            "global_pixel_recall",
            "mean_predicted_coverage",
            "elapsed_seconds",
        ]
        test_metrics = {
            key: test_summary[key]
            for key in metric_keys
            if key in test_summary
        }

    missing_required = [
        name for name, result in artifact_results.items()
        if not result["exists"]
    ]

    audit_pass = (
        len(missing_required) == 0
        and manifest_summary is not None
        and not manifest_summary["missing_required_columns"]
        and isinstance(test_summary, dict)
        and "read_error" not in test_summary
    )

    project_summary = {
        "project": "BeltCrack9kd_Damage_Intelligence",
        "stage": "4.8",
        "audit_status": "PASS" if audit_pass else "INCOMPLETE",
        "training_performed": False,
        "model_modified": False,
        "new_test_inference_performed": False,
        "model_type": "Small U-Net crack segmentation pilot",
        "checkpoint_path": str(CHECKPOINT),
        "checkpoint_best_epoch": (
            test_summary.get("checkpoint_best_epoch")
            if isinstance(test_summary, dict)
            else None
        ),
        "manifest_summary": manifest_summary,
        "final_test_metrics": test_metrics,
        "required_artifacts": artifact_results,
        "optional_prior_stage_outputs": optional_results,
        "missing_required_artifacts": missing_required,
        "important_limitation": (
            "The segmentation pilot was trained on a subset of the training "
            "data for two epochs. Test metrics are specific to this checkpoint "
            "and preprocessing setup. The current results do not establish "
            "production-ready crack segmentation or validated end-to-end "
            "physical conveyor damage prediction."
        ),
        "elapsed_seconds": round(time.time() - started, 2),
    }

    summary_path = OUT / "FINAL_PROJECT_SUMMARY.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(project_summary, file, indent=2)

    lines = [
        "BELTCRACK9KD DAMAGE INTELLIGENCE",
        "FINAL MODEL AUDIT",
        "=" * 48,
        f"Audit status: {project_summary['audit_status']}",
        f"Required artifacts missing: {len(missing_required)}",
        "",
        "FINAL TEST METRICS",
        "-" * 48,
    ]

    if test_metrics:
        for key, value in test_metrics.items():
            lines.append(f"{key}: {value}")
    else:
        lines.append("Final test summary could not be loaded.")

    lines.extend([
        "",
        "ARTIFACT CHECK",
        "-" * 48,
    ])

    for name, result in artifact_results.items():
        status = "FOUND" if result["exists"] else "MISSING"
        lines.append(f"{status}: {name} -> {result['path']}")

    lines.extend([
        "",
        "LIMITATIONS",
        "-" * 48,
        "The segmentation model is an experimental pilot, not a production model.",
        "The current metrics do not validate an end-to-end physical damage predictor.",
        "Do not tune the model or threshold using the final test results.",
        "",
        "Training performed during audit: NO",
        "New test inference performed during audit: NO",
    ])

    results_path = OUT / "FINAL_RESULTS.txt"
    results_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print("=" * 68)
    print(f"AUDIT STATUS: {project_summary['audit_status']}")
    print(f"Missing required artifacts: {len(missing_required)}")
    print(f"Project summary: {summary_path}")
    print(f"Final results text: {results_path}")
    print(f"Elapsed seconds: {project_summary['elapsed_seconds']}")
    print("STAGE 4.8 COMPLETED")


if __name__ == "__main__":
    main()
