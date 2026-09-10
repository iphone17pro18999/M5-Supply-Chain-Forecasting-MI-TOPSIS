# Editor and reviewer verification guide

This repository is organized as an auditable, step-by-step reproduction pipeline.

1. Download the three original M5 files from the official Kaggle competition page.
2. Place them in `data/raw/`.
3. Create a clean Python environment and install `requirements.txt`.
4. Run `python run_all.py`, or execute Steps 01–15 individually.
5. Inspect `outputs/generated/reproducibility_audit.json`.
6. The audit compares newly generated accuracy metrics, feature rankings, fold structure, regime counts, selected-series identity, and ranking order with reference artifacts.
7. Runtime equality is **not** required because runtime depends on hardware and software environment.
8. Figures are regenerated from Python; pre-rendered paper figures are not required.

The manuscript and Supplementary Material are intentionally not stored in the repository.
