# BA

A small analysis toolkit for exploring a loan dataset (`lc_large.csv`). The repo contains scripts that generate summary tables, figures, and an ANOVA/t-test Excel workbook suitable for quick reporting and review.

## Quickstart

1. Ensure the dataset `lc_large.csv` is at the repository root.
2. Create and activate a virtual environment and install dependencies (see Installation).
3. Run quick commands:

```bash
# generate figures and CSV summaries
python main.py

# generate ANOVA / t-test workbook (sheets for `term` and `home_ownership`)
python build_workbook.py
```

## What this repo produces

- Figures: PNG files written to `./figures/`.
- CSV summaries (produced by `main.py`): `univariate_summary.csv` and `bivariate_summary.csv` (output to `./tables/`).
- ANOVA / t-test workbook: `anova_tables.xlsx` (written to `./tables/`). The workbook contains one sheet per feature (by default `term` and `home_ownership`) with rows [between, within, total] and columns [DF, SS, MS, corr_val, stat, p_val].

## Installation

Requirements
- Python 3.10+ (the code uses recent libraries; 3.10+ is recommended). See `pyproject.toml` for specific dependency pins.
- Recommended: a virtual environment.

Install and activate a venv, then install runtime dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate   # Windows (PowerShell / CMD)

pip install -e .
# or explicitly:
# pip install numpy pandas matplotlib seaborn scipy openpyxl
```

If you don't want editable install, the explicit pip line above is sufficient.

## Usage

Run the main analysis and plotting pipeline:

```bash
python main.py
```

This will:
- Read `lc_large.csv` from the repository root.
- Cast certain columns to categorical (`term`, `home_ownership`, etc.).
	- Save PNG figures to `./figures/`.
	- Write `univariate_summary.csv` and `bivariate_summary.csv` to `./tables/`.

Generate the ANOVA/t-test workbook:

```bash
python build_workbook.py
```

This script currently produces sheets for `term` and `home_ownership`. Each sheet contains:
- Rows: `between`, `within`, `total`.
- Columns: `DF`, `SS`, `MS`, `corr_val`, `stat`, `p_val`.

Notes on values:
- `corr_val`: effect-size metric (Cohen's d for `term`, eta² for `home_ownership`).
- `stat`: test statistic (t for `term`, F for `home_ownership`) — placed on the `between` row.
- `p_val`: p-value for the corresponding test — placed on the `between` row.

## Project layout

- `main.py` — core plotting and summary routines (functions like `get_mean`, `get_group_means`, `create_univariate_table`).
- `build_workbook.py` — creates ANOVA/t-test sheets and saves to Excel.
- `lc_large.csv` — expected input dataset (place in repo root).
- `figures/` — generated PNG figures.
- `tables/anova_tables.xlsx` — output workbook (tables directory).
- `tables/univariate_summary.csv`, `tables/bivariate_summary.csv` — CSV outputs (tables directory).
- `pyproject.toml` — project metadata and dependencies.

## Troubleshooting

- File not found errors: verify `lc_large.csv` is in the repository root and that scripts are run from the repo root. Also check for accidental spaces in hard-coded paths (e.g. `/Users/.../BA ` vs `/Users/.../BA`).
- Plots missing / empty: confirm `figures/` exists (scripts will create it) and that the input CSV has non-empty columns used by the analysis.
- Dependency issues: if installation fails, try installing the explicit requirements individually (`numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`, `openpyxl`).

Quick smoke test using a small slice of data:

```bash
python -c "import pandas as pd; print(pd.read_csv('lc_large.csv').head())"
```

If that prints rows, you can run `python main.py` safely.

## Development

Recommended developer workflow and checks:

```bash
# create venv and install
python -m venv .venv
source .venv/bin/activate
pip install -e .

# run scripts
python main.py
python build_workbook.py

# lint / format (examples)
# pip install ruff black
# ruff check .
# black --check .

# tests (if/when added)
# pytest
```

Replace the lint/test commands with your preferred tools (flake8, ruff, black, pytest, etc.).

## Contributing

1. Fork the repo.
2. Create a feature branch: `git checkout -b feature/name`.
3. Commit and push your changes.
4. Open a pull request with a short description and tests if applicable.

Please follow consistent code style and include tests for larger changes.

## License

This project is MIT-licensed. See `LICENSE` (or add one) for full terms.

## Contact / Maintainership

Project maintained by the repository owner. For issues or feature requests, open an issue or create a pull request.

---
