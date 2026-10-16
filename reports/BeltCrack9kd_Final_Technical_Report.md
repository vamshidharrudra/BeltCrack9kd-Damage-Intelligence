# BeltCrack9kd Damage Intelligence

**Final Technical Report**

Report generated: 2026-10-15

## 1. Project Overview

BeltCrack9kd Damage Intelligence is a conveyor-belt crack analysis project built around dataset auditing, crack-mask generation, segmentation experimentation, sequential analysis, and maintenance decision intelligence.

The project includes an experimental image-segmentation model and separate feature-based damage intelligence components. The current segmentation pilot has not demonstrated production-ready crack segmentation or a validated end-to-end physical damage predictor.

## 2. Dataset and Data Integrity

The canonical dataset manifest contains 14,087 image records.

| Split | Images |
|---|---:|
| Train | 9,819 |
| Validation | 2,102 |
| Test | 2,166 |
| **Total** | **14,087** |

The canonical split assignments were audited for duplicate-image leakage across splits. The recorded split design avoids component and sequence conflicts.

The dataset audit identified 37,306 crack polygons. The mask generation stage produced 14,087 masks, with no reported image/mask dimension mismatches or mask-generation errors.

Source images include multiple resolutions, including 3840x2160, 1920x1080, and 2160x3840. The segmentation pipeline resizes images while preserving aspect ratio and adds centered padding.

## 3. Crack Mask Generation

Crack polygons from the LabelMe annotations were rasterized into binary masks. Nearest-neighbor interpolation was used when resizing masks so that the binary labels were preserved.

The generated masks were organized by split and sequence. The resulting mask dataset was used for the segmentation pilot.

## 4. Segmentation Model

A compact U-Net-style convolutional neural network was used for the CPU segmentation pilot.

- Input resolution: 256x256.
- Training device: CPU.
- Pilot training subset: 400 training images.
- Pilot validation subset: 100 validation images.
- Training duration: 2 epochs.
- Checkpoint: `models/segmentation/beltcrack9kd_cpu_pilot_best.pt`.
- Preprocessing: aspect-ratio-preserving resize, centered padding, RGB normalization, and binary masks.

This was a feasibility pilot, not a full-dataset training run. A separate larger baseline training attempt did not complete, so its partial results are not presented as a completed experiment.

## 5. Validation and Threshold Selection

The pilot checkpoint was evaluated on the validation split using the same resize-and-pad preprocessing as training. No training was performed during this diagnostic.

The selected threshold was 0.10, based on the highest mean-image Dice among the tested validation thresholds. The threshold was fixed before final test evaluation.

Validation threshold selection is a model-selection step. The validation metrics are not an independent estimate of final generalization performance.

## 6. Final Test Evaluation

The existing pilot checkpoint was evaluated on all 2,166 test images. No training or threshold tuning was performed on the test split.

| Metric | Test result |
|---|---:|
| Mean image Dice | 0.160466 |
| Mean image precision | 0.135608 |
| Mean image recall | 0.237736 |
| Global pixel Dice | 0.241551 |
| Global pixel precision | 0.232126 |
| Global pixel recall | 0.251773 |
| Mean predicted coverage | 0.004034 |

The global pixel Dice was approximately 0.2416. The global pixel precision and recall were approximately 23.21% and 25.18%, respectively. These results indicate that the current pilot has substantial segmentation errors and should not be considered production-ready.

The per-image results are available in `audit_outputs/stage_4_7_final_test/stage_4_7_test_image_metrics.csv`.

## 7. Qualitative Inspection

Twenty validation images were inspected by comparing the input image, ground-truth mask, predicted mask, and prediction overlay. The sample metrics are diagnostic and should not be treated as an independent test result.

Visual comparisons are stored in `audit_outputs/stage_4_6f_qualitative_inspection/comparisons/`.

## 8. Sequential and Maintenance Intelligence

Earlier project stages produced offline analysis, sequential-video outputs, and maintenance decision outputs. These components should be interpreted according to their own inputs and evaluation evidence.

The existing feature-based future crack-area prediction component is distinct from the image-segmentation model. The current results do not establish that segmentation predictions have been integrated into and validated as a complete physical damage forecasting system.

## 9. Limitations

1. The segmentation checkpoint was trained on a limited subset for only two epochs.
2. Test segmentation metrics are low, with substantial false positive and false negative predictions.
3. The threshold was selected on validation data and is not independently optimized for the test split.
4. Pixel-level crack segmentation does not directly measure physical crack depth, belt strength, or remaining useful life.
5. Reliable sequential crack tracking and real-world damage forecasting require further validation.
6. The current pilot should not be used alone for operational maintenance or safety decisions.

## 10. Reproducibility and Artifacts

Important project outputs include:

- `canonical_dataset/canonical_manifest.csv`
- `canonical_dataset/masks/`
- `models/segmentation/beltcrack9kd_cpu_pilot_best.pt`
- `audit_outputs/stage_4_5_crack_masks/`
- `audit_outputs/stage_4_6e_corrected_validation_diagnostic/`
- `audit_outputs/stage_4_6f_qualitative_inspection/`
- `audit_outputs/stage_4_7_final_test/`
- `audit_outputs/stage_4_8_final_model_audit/`

The Stage 4.8 audit reported PASS with zero missing required artifacts at the time it was run.

## 11. Conclusion

The project establishes an audited dataset workflow, generated crack masks, a reproducible segmentation pilot, validation-based threshold selection, final test metrics, and supporting damage intelligence artifacts.

The segmentation experiment is a baseline feasibility result rather than a production solution. Further training, error analysis, independent evaluation, and validation against real physical damage measurements are required before operational use.

---

This report is generated from the recorded project outputs. Missing source summaries are not replaced with fabricated results.
