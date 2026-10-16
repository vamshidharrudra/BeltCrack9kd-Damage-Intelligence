# BeltCrack9kd Damage Intelligence

A machine-learning research project exploring sequential conveyor-belt crack analysis, temporal damage progression, and maintenance-oriented damage intelligence using the BeltCrack9kd dataset.

## Project Goals

- Audit dataset structure, annotations, and image-to-annotation mappings.
- Investigate duplicate images and train/validation/test leakage.
- Engineer spatial and temporal features for damage analysis.
- Experiment with machine-learning models for damage progression.
- Evaluate crack segmentation approaches.
- Develop a Streamlit dashboard for damage-intelligence experiments.

## Project Structure

- `audit/` — Dataset, annotation, duplicate, and leakage audits.
- `scripts/` — Data preparation, feature engineering, experiments, model evaluation, and dashboard scripts.
- `reports/` — Technical report and research findings.
- `BeltCrack9kd_Damage_Intelligence/` — Application code.

Generated outputs, trained model artifacts, and the full dataset are excluded from this Git repository.

## Dataset

The project uses the BeltCrack9kd dataset. Obtain the dataset from its original source and follow the applicable license and usage conditions.

Dataset files are not included in this repository.

## Setup

Create and activate a Python virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the dependencies required by the script or application you intend to run. A pinned dependency file should be added after the project's actual environment has been verified.

## API Configuration

For features that use Groq, create a local `.env` file containing your own API key and model configuration. Never commit `.env` or expose API credentials in source code.

## Research Limitations

Experimental results should not be interpreted as proof of reliable real-world conveyor-belt damage prediction. Validate model performance, data leakage, segmentation quality, and generalization before operational use.

## Status

Research and development. Reproducibility instructions and dependency versions are still being finalized.