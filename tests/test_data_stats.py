from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# Helpers (duplicated here to keep tests self-contained)
def _cohen_d(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan
    m1, m2 = a.mean(), b.mean()
    s1, s2 = a.std(ddof=1), b.std(ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    return (m1 - m2) / pooled if pooled > 0 else np.nan


def _eta_squared(groups):
    groups = [np.asarray(g, dtype=float) for g in groups]
    groups = [g[np.isfinite(g)] for g in groups if g.size > 0]
    if not groups:
        return np.nan
    all_data = np.concatenate(groups)
    if all_data.size == 0:
        return np.nan
    grand_mean = all_data.mean()
    ss_between = sum([len(g) * (g.mean() - grand_mean) ** 2 for g in groups])
    ss_total = ((all_data - grand_mean) ** 2).sum()
    return ss_between / ss_total if ss_total > 0 else np.nan


def _repo_root():
    # repo root assumed to be parent of this tests directory
    return Path(__file__).resolve().parents[1]


def _load_data():
    csv_path = _repo_root() / "lc_large.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected data file not found: {csv_path}")
    # Read with pandas; rely on existing numeric formatting in the dataset
    df = pd.read_csv(csv_path)
    return df


def _finite_pair(df, xcol, ycol):
    pair = df[[xcol, ycol]].dropna()
    x = np.asarray(pair[xcol]).reshape(-1)
    y = np.asarray(pair[ycol]).reshape(-1)
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def _assert_close(val, expected, tol, label="value"):
    assert np.isfinite(val), f"{label} is not finite: {val}"
    diff = abs(float(val) - float(expected))
    assert diff <= tol, f"{label} mismatch: got {val}, expected {expected} (diff={diff}, tol={tol})"


def test_data_int_rate_pearson_r_matches():
    df = _load_data()
    x, y = _finite_pair(df, "int_rate", "loan_status_numeric")
    # Guard against constant arrays
    assert len(x) >= 2 and not np.all(x == x[0]) and not np.all(y == y[0])
    r, p = stats.pearsonr(x, y)
    # Reference values confirmed during analysis runs
    _assert_close(r, -0.177695, tol=5e-7, label="Pearson r (int_rate vs loan_status_numeric)")
    # R^2 also check
    _assert_close(r * r, 0.031575, tol=5e-6, label="R^2 for int_rate")
    # p-value threshold should be <0.001
    assert p < 0.001


def test_data_term_welch_t_and_cohens_d():
    df = _load_data()
    # Two groups by term (expect exactly two categories like '36 months' and '60 months')
    terms = [t for t in df["term"].dropna().unique()]
    assert len(terms) >= 2
    # pick the two most common categories to be robust if rare labels exist
    counts = df["term"].value_counts()
    top_two = list(counts.index[:2])
    g1 = df.loc[df["term"] == top_two[0], "loan_status_numeric"].dropna().to_numpy()
    g2 = df.loc[df["term"] == top_two[1], "loan_status_numeric"].dropna().to_numpy()
    t_stat, p = stats.ttest_ind(g1, g2, equal_var=False)
    d = _cohen_d(g1, g2)
    # Reference targets from prior runs
    _assert_close(t_stat, 25.934, tol=1e-3, label="Welch t (term)")
    assert p < 0.001
    _assert_close(abs(d), 0.146, tol=2e-3, label="Cohen's d |term|")


def test_data_home_ownership_anova_and_eta2():
    df = _load_data()
    cats = [c for c in df["home_ownership"].dropna().unique()]
    assert len(cats) >= 2
    groups = [df.loc[df["home_ownership"] == c, "loan_status_numeric"].dropna().to_numpy() for c in cats]
    # Filter out any empty groups just in case
    groups = [g for g in groups if len(g) > 0]
    f_stat, p = stats.f_oneway(*groups)
    eta2 = _eta_squared(groups)
    _assert_close(f_stat, 192.859, tol=1e-2, label="ANOVA F (home_ownership)")
    assert p < 0.001
    # eta^2 is small (~0.003xx)
    assert np.isfinite(eta2) and 0.002 <= float(eta2) <= 0.004, f"eta^2 out of expected range: {eta2}"
