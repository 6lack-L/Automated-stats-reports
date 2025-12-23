import numpy as np
from build_workbook import cohen_d, regression_anova_from_corr, anova_for_categories


def test_cohen_d_known():
    # Group A mean=20, Group B mean=30, both sd=10 -> d = (20-30)/10 = -1.0
    a = np.array([10.0, 20.0, 30.0])
    b = np.array([20.0, 30.0, 40.0])
    d = cohen_d(a, b)
    assert np.isfinite(d)
    assert abs(d - (-1.0)) < 1e-6


def test_regression_anova_perfect_corr():
    # x perfectly correlates with y -> r=1, SSR = SST, SSE = 0
    y = np.array([1.0, 2.0, 3.0, 4.0])
    r = 1.0
    table = regression_anova_from_corr(y, r)
    between = table['between']
    within = table['within']
    total = table['total']

    # SST for y should be 5.0 (as computed manually)
    assert total['DF'] == 3
    assert pytest_isfinite_or_equal(total['SS'], 5.0)

    # SSR = r^2 * SST = 5.0, SSE = 0
    assert between['DF'] == 1
    assert pytest_isfinite_or_equal(between['SS'], 5.0)
    assert within['DF'] == 2
    assert pytest_isfinite_or_equal(within['SS'], 0.0)


def test_anova_for_categories_eta2():
    # two groups: [1,2] and [3,4]
    g1 = np.array([1.0, 2.0])
    g2 = np.array([3.0, 4.0])
    table = anova_for_categories(np.concatenate([g1, g2]), [g1, g2])
    between = table['between']
    total = table['total']

    # Expected: ss_between = 4, ss_total = 5, eta2 = 4/5 = 0.8
    assert between['DF'] == 1
    assert total['DF'] == 3
    assert pytest_isfinite_or_equal(between['SS'], 4.0)
    assert pytest_isfinite_or_equal(total['SS'], 5.0)
    assert abs(between['corr_val'] - 0.8) < 1e-9


def pytest_isfinite_or_equal(val, expected, tol=1e-9):
    try:
        if np.isnan(val) and np.isnan(expected):
            return True
    except Exception:
        pass
    return abs(float(val) - float(expected)) <= tol
