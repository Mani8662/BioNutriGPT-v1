"""
Merge NHANES 2005-2016 (from BioNutriGPT_final_dataset_2005_2018.csv, excluding
the overlapping 2017-2018 cycle) and NHANES 2017-March 2020 Pre-Pandemic raw
data into a single unmodified raw CSV file.

No values are altered, cleaned, imputed, clipped, scaled, or filtered beyond
removing the 2017-2018 cycle rows that CDC already bundled inside 2017-2020.
"""

from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent

SRC_2005_2018 = ROOT_DIR / "P" / "BioNutriGPT_final_dataset_2005_2018.csv"
SRC_2017_2020 = BASE_DIR / "NHANES_2017_2020_PrePandemic_Raw.csv"
OUT_MERGED_RAW = BASE_DIR / "BioNutriGPT_merged_raw_2005_2020.csv"


def merge_raw_datasets(
    src_2005_2018: Path = SRC_2005_2018,
    src_2017_2020: Path = SRC_2017_2020,
    out_csv: Path = OUT_MERGED_RAW,
) -> pd.DataFrame:
    """Concatenate 2005-2016 and 2017-2020 raw CSVs via outer column union."""
    print(f"Loading {src_2005_2018} ...")
    df_old = pd.read_csv(src_2005_2018, low_memory=False)
    print(f"  Loaded shape: {df_old.shape}")

    overlap_mask = df_old["NHANES_CYCLE"] == "2017-2018"
    n_overlap = int(overlap_mask.sum())
    df_2005_2016 = df_old.loc[~overlap_mask].copy()
    print(f"  Excluded overlapping '2017-2018' rows: {n_overlap}")
    print(f"  Retained '2005-2016' shape: {df_2005_2016.shape}")

    print(f"Loading {src_2017_2020} ...")
    df_2017_2020 = pd.read_csv(src_2017_2020, low_memory=False)
    print(f"  Loaded shape: {df_2017_2020.shape}")

    merged = pd.concat(
        [df_2005_2016, df_2017_2020],
        axis=0,
        ignore_index=True,
        sort=False,
    )
    print(f"Merged raw shape (outer column union): {merged.shape}")
    print("NHANES_CYCLE breakdown:")
    print(merged["NHANES_CYCLE"].value_counts(dropna=False).to_string())

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    size_mb = out_csv.stat().st_size / (1024 * 1024)
    print(f"Saved: {out_csv} ({size_mb:.2f} MB)")
    return merged


if __name__ == "__main__":
    merge_raw_datasets()
