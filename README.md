# BeltCrack9kd Damage Intelligence

A machine-learning research project exploring conveyor-belt crack detection, sequential image analysis, temporal damage indicators, and maintenance-oriented damage intelligence using the **BeltCrack9kd dataset**.

> **Important:** This is a research prototype. Sequential images can support analysis of observed changes, but they do not automatically provide ground-truth labels for future physical crack growth or remaining useful life.

## Project Goals

- Audit dataset structure and JSON annotation-to-image mappings.
- Investigate duplicate images and potential train/validation/test leakage.
- Build spatial and temporal features for crack/damage analysis.
- Experiment with machine-learning models for damage-related indicators.
- Evaluate crack-segmentation approaches.
- Develop a Streamlit dashboard for exploring damage-intelligence experiments.

## Dataset Used

**Dataset:** BeltCrack9kd (the downloaded multi-part dataset, supplied as packages such as `BeltCrack9kd1` through `BeltCrack9kd13`).

**Role in this project:** The images and their JSON annotations are used for dataset auditing, image/annotation pairing, duplicate and split-leakage analysis, crack-feature extraction, temporal/sequential experiments, and segmentation/model experiments.

### Dataset audit notes

The project's audit scripts recorded 13 dataset packages and checked JSON annotations and image pairing. One audit run found 36,357 valid JSON files, and a later pairing audit reported image matches and dimension matches for all 36,357 JSON files in that audit scope. A separate package-level audit of `BeltCrack9kd1` reported 14,087 images/JSON files and a train/validation/test split. **These are results from specific audit scopes, not a claim that the complete downloaded dataset contains only 14,087 images.** Refer to the scripts and reports in this repository for the scope and details of each audit.

### Access and licensing

The raw dataset is **not included** in this GitHub repository because the download is very large (approximately 162 GB in the local project setup). Obtain BeltCrack9kd from the original dataset publisher/source, and follow its license, citation requirements, and access terms. No dataset download URL is provided here because the exact canonical source URL and redistribution terms have not been verified in this README.

After downloading and extracting the dataset, configure the paths expected by the relevant scripts for your local environment. Do not assume every script runs without path configuration or generated intermediate files.

## Repository Structure

- `audit/` — Dataset, annotation, duplicate, and leakage audit scripts.
- `scripts/` — Data preparation, feature engineering, temporal experiments, model evaluation, and app-related scripts.
- `reports/` — Technical report and research findings.
- `BeltCrack9kd_Damage_Intelligence/belt_only_app.py` — Main belt-focused application file currently tracked in the repository.
- Root-level Python files — Additional audit utilities.

Generated outputs, trained model artifacts, raw data, and large intermediate files are excluded from this Git repository. Some experiments therefore require you to regenerate those artifacts locally before they can be reproduced.

## Setup

Use Python 3 and create a virtual environment from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the dependencies required by the particular script or application you intend to run. A verified, pinned `requirements.txt` is still needed for reproducible installation; dependencies should be derived from the actual imports and working environment rather than guessed.

## API Configuration

Features that use Groq may require a local `.env` file containing your own API key and model configuration. Never commit `.env`, API tokens, or other secrets. The repository's `.gitignore` is intended to exclude these files.

## Reproducibility and Limitations

- Check each script's input/output paths and configuration before running it.
- Review the audit reports before interpreting model scores, especially where duplicate images may cross dataset splits.
- Sequential image ordering alone does not prove that images represent the same physical belt region over time or provide true crack-growth ground truth.
- Model outputs are experimental indicators, not verified predictions of future physical damage, failure time, or remaining useful life.
- Real-world deployment would require validation on properly grouped, leakage-controlled data and independent operating conditions.

## Status

Research and development. The source scripts and audit documentation are tracked; full data, generated outputs, trained artifacts, and fully pinned environment setup are not bundled in this repository.
