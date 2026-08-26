# Usage: python random_subset.py data.csv --n 1000 --seed 42 --outdir output/
#   input_csv       path to the source CSV file
#   --n             number of rows to randomly sample (default: all rows)
#   --seed          random seed for reproducibility (default: none)
#   --outdir        folder to save train.csv/test.csv (default: current directory)
#   --train-ratio   fraction of sampled rows used for training (default: 0.8, i.e. 4:1)
#
# Note: the first row of input_csv is always treated as the header row.
# It is excluded from --n and written to both train.csv and test.csv.

import argparse
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--outdir", type=str, default=".")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    n = args.n if args.n is not None else len(df)
    n = min(n, len(df))

    sample = df.sample(n=n, random_state=args.seed)

    train_df = sample.sample(frac=args.train_ratio, random_state=args.seed)
    test_df = sample.drop(train_df.index)

    os.makedirs(args.outdir, exist_ok=True)
    train_df.to_csv(os.path.join(args.outdir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(args.outdir, "test.csv"), index=False)


if __name__ == "__main__":
    main()
