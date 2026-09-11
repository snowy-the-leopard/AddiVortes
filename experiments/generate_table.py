#!/usr/bin/env python3
"""
Generate a LaTeX table* from a CSV of factorial design results.

- Sorts rows by test number (first CSV column) ascending
- Drops the columns listed in DROP_COLUMNS (nu, q, mcmcBurnin, and the
  individual m/omega/lambda/mcmcIter columns)
- Adds a new second column listing which parameters differ from their
  default values, formatted per PARAM_FORMAT
- Writes the .tex snippet to stdout (or to a file if OUT_PATH is set)
"""

import csv
import sys
from pathlib import Path

# ---------------- CONFIG ----------------
PARENT = Path(__file__).resolve().parent
CSV_PATH = PARENT / "settings_results_full.csv"          # path to your input CSV
OUT_PATH = PARENT / "table.tex"         # set to None to print to stdout instead

# Columns to drop entirely from the output table.
DROP_COLUMNS = {"nu", "q", "mcmcBurnin", "m", "omega", "lambda", "mcmcIter"}

# Default value for each parameter that feeds the "changed" column.
# Order here also controls the order they're listed in when several
# differ from default in the same row.
DEFAULTS = {
    "lambda": 25,
    "m": 200,
    "mcmcIter": 1200,
    "omega": 3,
}

# How to render each parameter's name when it differs from default.
PARAM_FORMAT = {
    "lambda": r"$\lambda$",
    "m": r"$m$",
    "mcmcIter": r"\texttt{mcmcIter}",
    "omega": r"$\omega$",
}

CHANGED_COL_HEADER = "Parameters changed"
# -----------------------------------------


def load_rows(csv_path):
    # utf-8-sig strips a leading BOM if present (common when a CSV is saved
    # from Excel/Windows tools), which otherwise attaches itself to the
    # first header name, e.g. "\ufefftest_no" instead of "test_no".
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        rows = list(reader)
    return rows


def changed_params_cell(row):
    """Return a LaTeX string listing params that differ from default."""
    changed = []
    for param in DEFAULTS:  # dict preserves insertion order: lambda, m, mcmcIter, omega
        default_val = DEFAULTS[param]
        raw_val = row[param].strip()
        try:
            differs = float(raw_val) != float(default_val)
        except ValueError:
            differs = raw_val != str(default_val)
        if differs:
            changed.append(PARAM_FORMAT[param])
    return ", ".join(changed)


def num_changed_params(row):
    """Count how many params differ from their default value in this row."""
    count = 0
    for param, default_val in DEFAULTS.items():
        raw_val = row[param].strip()
        try:
            differs = float(raw_val) != float(default_val)
        except ValueError:
            differs = raw_val != str(default_val)
        if differs:
            count += 1
    return count


def build_table(rows):
    if not rows:
        raise ValueError("No rows found in CSV.")

    all_columns = list(rows[0].keys())
    test_no_col = all_columns[0]

    # Sort by number of changed parameters ascending (0 changed first, then
    # 1, then 2, ...). Ties are broken by the original test number so the
    # ordering is stable and reproducible.
    rows_sorted = sorted(
        rows,
        key=lambda r: (num_changed_params(r), int(r[test_no_col])),
    )

    # Test numbers are reassigned 1..N in this new order, since the original
    # numbering no longer reflects anything meaningful.

    # Remaining columns after dropping, excluding the test-number column
    # (which is placed first) and the params folded into the new column.
    other_columns = [
        c for c in all_columns if c != test_no_col and c not in DROP_COLUMNS
    ]

    # Final column layout: test_no, [changed params], then everything else
    display_columns = [test_no_col, CHANGED_COL_HEADER] + other_columns

    n_cols = len(display_columns)
    col_spec = "|" + "c|" * n_cols

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\hline")

    header = " & ".join(rf"\textbf{{{c}}}" for c in display_columns)
    lines.append(header + r" \\")
    lines.append(r"\hline")

    for new_test_no, row in enumerate(rows_sorted, start=1):
        cells = [str(new_test_no), changed_params_cell(row)]
        cells += [row[c].strip() for c in other_columns]
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


def main():
    rows = load_rows(CSV_PATH)
    table_tex = build_table(rows)

    if OUT_PATH:
        with open(OUT_PATH, "w") as f:
            f.write(table_tex + "\n")
        print(f"Wrote LaTeX table to {OUT_PATH}", file=sys.stderr)
    else:
        print(table_tex)


if __name__ == "__main__":
    main()