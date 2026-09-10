"""Convenience runner for the complete reproducibility pipeline."""
from pathlib import Path
import subprocess, sys

steps = [
    "01_config.py","02_verify_data.py","03_prepare_category_store.py",
    "04_category_store_features.py","05_category_store_mi.py",
    "06_category_store_forecasting.py","07_entropy_topsis.py",
    "08_item_store_regime_classification.py","09_item_store_features_mi.py",
    "10_item_store_regime_forecasting.py","11_generate_main_tables.py",
    "12_generate_supplementary_tables.py","13_generate_main_figures.py",
    "14_generate_supplementary_figures.py","15_reproducibility_audit.py",
]

for s in steps:
    print(f"\n=== Running {s} ===")
    subprocess.run([sys.executable, str(Path("src")/s)], check=True)
