#!/usr/bin/env python3

"""
Flip BlockPush CSV replay files.

Expected CSV columns:
0 st_block_x
1 st_block_y
2 st_block_or
3 st_ee_x
4 st_ee_y
5 st_ee_tgt_x
6 st_ee_tgt_y
7 act_x
8 act_y
9 qval  # optional
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def wrap_angle_pi(theta):
    new_theta = theta + np.pi
    new_theta[new_theta >= np.pi] -= 2 * np.pi
    return new_theta


def flip_csv_file(src_csv: Path, tgt_csv: Path, verbose: bool = False):
    df = pd.read_csv(src_csv)

    # Same as original replay-buffer script:
    # flip_dims = [0, 3, 5] + [7]
    cols_to_flip = [
        df.columns[0],  # block x
        df.columns[3],  # ee x
        df.columns[5],  # ee target x
        df.columns[7],  # action x
    ]

    if verbose:
        print(f"\n{src_csv.name}")
        print("flipping columns:", cols_to_flip)

    df[cols_to_flip] = -df[cols_to_flip]

    # Same block orientation logic
    ori_col = df.columns[2]
    df[ori_col] = wrap_angle_pi(df[ori_col].to_numpy())

    tgt_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(tgt_csv, index=False)

    if verbose:
        print(f"saved -> {tgt_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src_dir", help="folder containing source CSV files")
    parser.add_argument("tgt_dir", help="folder to save flipped CSV files")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    tgt_dir = Path(args.tgt_dir)

    assert src_dir.is_dir(), f"source folder does not exist: {src_dir}"
    assert not tgt_dir.exists(), f"target folder already exists: {tgt_dir}"

    tgt_dir.mkdir(mode=0o775, parents=True, exist_ok=False)

    csv_files = sorted(src_dir.glob("*.csv"))
    print(f"found {len(csv_files)} CSV files in {src_dir}")

    for src_csv in csv_files:
        tgt_csv = tgt_dir / src_csv.name
        flip_csv_file(src_csv, tgt_csv, verbose=args.verbose)

    print(f"done. saved flipped CSVs to {tgt_dir}")


if __name__ == "__main__":
    main()