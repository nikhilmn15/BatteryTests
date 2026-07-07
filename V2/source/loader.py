"""
Very short code block here to clean and weed out corrupted data if any and to eliminate
early cycle number inflations to help the model be more accurate
"""

import pandas as pd


def load_raw(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cycle 1 registers near-zero IR/QD/QC across most batteries (sensor
    handshake artifact, not a real reading) then drop it. Fill rare missing
    protocol metadata with the per-column median rather than dropping rows."""
    df = df.copy()
    df = df[df["cycle"] > 1]

    for col in ["C1", "Q1", "C2"]:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    if "cycle_life" in df.columns and df["cycle_life"].isna().any():
        df["cycle_life"] = df["cycle_life"].fillna(df.groupby("battery_id")["cycle"].transform("max"))

    return df


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/Lithium-Ion Battery Cycle Life.csv"
    df = clean_data(load_raw(path))
    print(f"{len(df)} rows, {df['battery_id'].nunique()} batteries after cleaning")
