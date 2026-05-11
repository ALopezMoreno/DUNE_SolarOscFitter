#!/usr/bin/env python3
"""
Convert inputs_neutronOnly_may2026 files to the format expected by SolarOscFitter.

Changes applied to every file:
  - Rename energy columns: E_max / E_reco → Ereco, E_true → Etrue
  - Convert energy units: keV → GeV (divide by 1e6)
  - Convert mask: integer 0/1 → string False/True
  - Add weights column of 1.0 where absent

Output mirrors the subdirectory structure under
  inputs/inputs_neutronOnly_may2026_converted/
"""

import pandas as pd
import pathlib
import sys

SRC_DIR  = pathlib.Path("inputs/inputs_neutronOnly_may2026")
DEST_DIR = pathlib.Path("inputs/inputs_neutronOnly_may2026_converted")

ENERGY_SCALE = 1e-6  # keV → GeV

def convert_file(src: pathlib.Path, dest: pathlib.Path) -> None:
    df = pd.read_csv(src)

    # --- Rename energy columns -----------------------------------------------
    rename_map = {}
    for col in df.columns:
        if col in ("E_max", "E_reco"):
            rename_map[col] = "Ereco"
        elif col == "E_true":
            rename_map[col] = "Etrue"
    if rename_map:
        df = df.rename(columns=rename_map)

    # --- Unit conversion: keV → GeV ------------------------------------------
    for col in ("Etrue", "Ereco"):
        if col in df.columns:
            df[col] = df[col] * ENERGY_SCALE

    # --- Mask: 0/1 int → False/True string ------------------------------------
    if "mask" in df.columns:
        df["mask"] = df["mask"].map({0: "False", 1: "True"})

    # --- Add weights = 1.0 if missing -----------------------------------------
    if "weights" not in df.columns:
        df.insert(df.columns.get_loc("mask"), "weights", 1.0)

    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    print(f"  {src.relative_to(SRC_DIR.parent.parent)}")
    print(f"    → {dest.relative_to(SRC_DIR.parent.parent)}")


def main() -> None:
    files = sorted(SRC_DIR.rglob("*.csv"))
    if not files:
        print(f"No CSV files found under {SRC_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"Converting {len(files)} files …\n")
    for src in files:
        rel  = src.relative_to(SRC_DIR)
        dest = DEST_DIR / rel
        convert_file(src, dest)

    print(f"\nDone. Converted files are in {DEST_DIR}/")


if __name__ == "__main__":
    main()
