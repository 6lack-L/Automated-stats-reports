# Bivariate Analysis Summary

Total features analyzed: 5

## loan_amnt

Test: R Squared

![loan_amnt](figures/loan_amnt_scatter_regression.png)

| Metric | Value |
|:---:|:---:|
| r | -0.034 |
| R² | 0.001 |
| p-value | <0.001 |
| Effect size | -0.550 |
| N | 167608 |

---

## term

Test: t-test

![term](figures/term_categorical_summary.png)

| Stat | Value |
|:---:|:---:|
| t | 25.934 |
| p-value | <0.001 |
| Effect size | 0.146 |
| Groups | 2 |
| N | 167608 |

| 36 months | 60 months |
| :---: | :---: |
| 4.872 (n=120813) | 4.649 (n=46795) |

---

## int_rate

Test: R Squared

![int_rate](figures/int_rate_scatter_regression.png)

| Metric | Value |
|:---:|:---:|
| r | -0.178 |
| R² | 0.032 |
| p-value | <0.001 |
| Effect size | 0.015 |
| N | 167608 |

---

## home_ownership

Test: ANOVA

![home_ownership](figures/home_ownership_categorical_summary.png)

| Stat | Value |
|:---:|:---:|
| F | 192.859 |
| p-value | <0.001 |
| Effect size | 0.003 |
| Groups | 4 |
| N | 167608 |

| ANY | MORTGAGE | OWN | RENT |
| :---: | :---: | :---: | :---: |
| 4.709 (n=127) | 4.897 (n=81332) | 4.799 (n=20333) | 4.705 (n=65816) |

---

## dti

Test: R Squared

![dti](figures/dti_scatter_regression.png)

| Metric | Value |
|:---:|:---:|
| r | -0.030 |
| R² | 0.001 |
| p-value | <0.001 |
| Effect size | 0.042 |
| N | 167402 |

---
