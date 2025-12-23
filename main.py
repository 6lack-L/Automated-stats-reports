import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats



def create_univariate_table(df, column):
    x = column
    print(df[x].dtype.name == 'category')
    if df[x].dtype.name == 'category':
        table_df = {
            'feature': x,
            'type': df[x].dtype,
            'count': get_count(df, x),
            'unique_values': get_unique_values(df, x),
            'missing_values': get_missing_values(df, x),
        }
        table_df = pd.DataFrame.from_dict(table_df, orient='index', columns =[x])
        print(table_df)
        return table_df
    else:
        table_df = {
            'feature': x,
            'type': df[x].dtype,
            'count': get_count(df, x),
            'unique_values': get_unique_values(df, x),
            'missing_values': get_missing_values(df, x),
            'mean': get_mean(df, x),
            'std': std(df, x),
            'min': max_min(df, x)[1],
            '25%': quantiles(df, x)[0.25],
            '50%': quantiles(df, x)[0.5],
            '75%': quantiles(df, x)[0.75],
            'max': max_min(df, x)[0],
            'skewness': skewness(df, x),
            'kurtosis': kurtosis(df, x),
        }
    table_df = pd.DataFrame.from_dict(table_df, orient='index', columns =[x])
    print(table_df)
    return table_df

def cohen_d(group_a, group_b):
    """Compute Cohen's d between two 1-D numeric arrays/Series."""
    a = np.asarray(group_a)
    b = np.asarray(group_b)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan
    m1, m2 = a.mean(), b.mean()
    s1, s2 = a.std(ddof=1), b.std(ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    return (m1 - m2) / pooled if pooled > 0 else np.nan


def eta_squared(groups):
    """Estimate eta-squared for ANOVA given a list of groups (arrays/Series)."""
    groups = [np.asarray(g)[~np.isnan(g)] for g in groups]
    all_data = np.concatenate(groups) if len(groups) > 0 else np.array([])
    if all_data.size == 0:
        return np.nan
    grand_mean = all_data.mean()
    ss_between = sum([len(g) * (g.mean() - grand_mean) ** 2 for g in groups])
    ss_total = ((all_data - grand_mean) ** 2).sum()
    return ss_between / ss_total if ss_total > 0 else np.nan


def display_table(table_df, display_table):
    display_table.append(table_df)
    for item in display_table:
        for col in item.columns:
            print(f"{col}: {item[col].values[0]}")
    return display_table


def get_count(df, column):
    count = df[column].count()
    if not pd.api.types.is_numeric_dtype(df[column]):
        categorical_count = df[column].value_counts()
        return categorical_count.to_dict()
    return int(count)

def get_unique_values(df, column):
    unique_values = df[column].nunique()
    return int(unique_values)

def get_missing_values(df, column):
    missing_values = df[column].isnull().sum()
    return int(missing_values)

def quantiles(df, column):
    quantiles = df[column].quantile([0.25, 0.5, 0.75])
    return quantiles

def max_min(df, column):
    max_value = df[column].max()
    min_value = df[column].min()
    return max_value, min_value

def get_mean(df, column):
    s = df[column]
    if pd.api.types.is_numeric_dtype(s):
        return s.mean()
    # attempt coercion from category/string to numeric
    s_num = pd.to_numeric(s.astype(str).str.replace(',', ''), errors='coerce')
    if s_num.notna().any():
        return s_num.mean()
    return np.nan  # or None, or skip including 'mean' in your table

def get_group_means(df, column):
    group_means = df.groupby(column)['loan_status_numeric'].mean()
    print(group_means)
    return group_means

def std(df, column):
    std_value = df[column].std()
    return float(std_value)

def skewness(df, column):
    skewness_value = df[column].skew()
    return float(skewness_value)

def kurtosis(df, column):
    kurtosis_value = df[column].kurtosis()
    return float(kurtosis_value)

## removed unused t_test helper

def p_value_threshold(p):
    """Return a compact threshold label for a p-value.

    - '<0.001' if p < 0.001
    - '<0.05' if p < 0.05
    - '>=0.05' otherwise
    - '' if p is NaN or cannot be parsed
    """
    try:
        if pd.isna(p):
            return ""
        p = float(p)
    except Exception:
        return ""
    if p < 0.001:
        return "<0.001"
    if p < 0.05:
        return "<0.05"
    return ">=0.05"




def plot_scatter_regression(df, column, out_path, corr, pval, effect_size):
    """Improved scatter plot with regression line and annotation."""
    plt.figure(figsize=(10, 6))
    
    # Use scatterplot for more control over points
    sns.scatterplot(data=df, x=column, y='loan_status_numeric', 
                    color="black", marker="D", s=15, alpha=0.6, legend=False)

    # Overlay a regression line without the scatter points, and disable CI band (red shading)
    sns.regplot(data=df, x=column, y='loan_status_numeric', y_jitter=0.03,
                scatter=False, color='red', line_kws={'linewidth': 2}, ci=None)

    plt.title(f"Scatter Plot of {column} vs. Loan Status", fontsize=16)
    plt.xlabel(column, fontsize=12)
    plt.ylabel("Loan Status (Numeric)", fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Add correlation, R², and p-threshold as an annotation
    try:
        r_val = float(corr)
        r2 = r_val * r_val
    except Exception:
        r2 = np.nan
    pthr = p_value_threshold(pval)
    p_label = pthr if str(pthr).strip() else "n/a"
    stats_text = f'R²: {r2:.3f}\np-value: {p_label}\nEffect size (d): {effect_size:.3f}'
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_boxplot_group_means(df, column, out_path, stats_text):
    """Improved boxplot with statistical annotation (caller provides stats_text)."""
    plt.figure(figsize=(12, 7))

    sns.boxplot(x=column, y='loan_status_numeric', data=df, hue=column, palette='viridis', legend=False)

    plt.title(f"Boxplot of Loan Status by {column}", fontsize=16)
    plt.xlabel(column, fontsize=12)
    plt.ylabel("Loan Status (Numeric)", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', linewidth=0.5)

    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def delete_png_files(dir_path: str):
    """Delete all .png files inside the given directory path."""
    if not os.path.isdir(dir_path):
        return
    for name in os.listdir(dir_path):
        if name.lower().endswith(".png"):
            full_path = os.path.join(dir_path, name)
            try:
                os.remove(full_path)
            except OSError:
                pass


def main():
    # Use repo-relative directories (assumes scripts are run from repo root)
    cwd = os.getcwd()
    figures_dir = os.path.join(cwd, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    # Ensure a central tables directory for CSV/MD outputs
    tables_dir = os.path.join(cwd, 'tables')
    os.makedirs(tables_dir, exist_ok=True)
    # clean figures dir
    delete_png_files(figures_dir)
    file = os.path.join(cwd, 'lc_large.csv')
    lst = [
            'loan_status_numeric',
            'loan_amnt',
            'term',
            'int_rate',
            'emp_length',
            'home_ownership',
            'annual_inc',
            'verification_status',
            'mths_since_last_delinq',
            'open_acc',
            'dti'
        ]
    lst2 = [
            'term',
            'emp_length',
            'home_ownership',
            'verification_status'
        ]
    assignment = [
        'loan_status_numeric',
        'loan_amnt',
        'term',
        'int_rate',
        'home_ownership',
        'dti'
    ]
    df = pd.read_csv(file)

    for x in df.columns:
        if x in lst2:
            df[x] = df[x].astype('category')
        if x not in lst:
            df = df.drop(columns=x)
            continue

    rows = []
    for x in df.columns :
        row = {
            "feature": x,
            "type": str(df[x].dtype),
            "count": get_count(df, x),
            "unique_values": get_unique_values(df, x),
            "missing_values": get_missing_values(df, x),
        }
        if pd.api.types.is_numeric_dtype(df[x]):
            row.update({
                "mean": get_mean(df, x),
                "std": std(df, x),
                "min": max_min(df, x)[1],
                "25%": quantiles(df, x)[0.25],
                "50%": quantiles(df, x)[0.5],
                "75%": quantiles(df, x)[0.75],
                "max": max_min(df, x)[0],
                "skewness": skewness(df, x),
                "kurtosis": kurtosis(df, x),
            })
        rows.append(row)

    df_rows = pd.DataFrame(rows).set_index("feature")
    combined_df = df_rows
    

    rounded_df = combined_df.round(3)
    uni_path = os.path.join(tables_dir, "univariate_summary.csv")
    rounded_df.to_csv(uni_path)

    print(rounded_df.head())
    print(f"Univariate analysis completed. Summary saved to '{uni_path}'.")

    # Collect rows for a bivariate summary CSV
    bivariate_rows = []

    for col in df.columns:
        if col == 'loan_status_numeric':
            continue
        if col not in assignment:
            df = df.drop(columns=col)
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            out_scatter = os.path.join(figures_dir, f"{col}_scatter_regression.png")
            # Pairwise drop NA to compute correlation robustly
            pair = df[[col, 'loan_status_numeric']].dropna()
            if len(pair) >= 2:
                # Ensure 1-D numpy arrays and filter non-finite values
                x = np.asarray(pair[col]).reshape(-1)
                y = np.asarray(pair['loan_status_numeric']).reshape(-1)
                finite_mask = np.isfinite(x) & np.isfinite(y)
                x = x[finite_mask]
                y = y[finite_mask]
                if len(x) >= 2 and not (np.all(x == x[0]) or np.all(y == y[0])):
                    try:
                        corr, pval = stats.pearsonr(x, y)
                    except Exception:
                        corr, pval = np.nan, np.nan
                else:
                    corr, pval = np.nan, np.nan
            else:
                corr, pval = np.nan, np.nan

            # R^2 from simple linear relationship equals squared Pearson r
            if isinstance(corr, (int, float, np.floating)):
                r2 = float(corr) ** 2
                if not np.isfinite(r2):
                    r2 = np.nan
            else:
                r2 = np.nan

            # Cohen's d between the two loan_status groups on the numeric feature
            grp0 = df.loc[df['loan_status_numeric'] == 0, col].dropna().to_numpy()
            grp1 = df.loc[df['loan_status_numeric'] == 1, col].dropna().to_numpy()
            d_val = cohen_d(grp1, grp0)
            # Also log the raw Pearson correlation for clarity
            try:
                corr_print = f"{float(corr):.6f}" if isinstance(corr, (int, float, np.floating)) else "nan"
            except Exception:
                corr_print = "nan"
            print(f"{col}: r={corr_print}, R^2={r2:.3f}, p={p_value_threshold(pval)}, effect size d={d_val:.3f}")
            # Pass raw numeric p-value to plotting; the function will convert to threshold for display
            plot_scatter_regression(df, col, out_scatter, corr, pval, d_val)

            bivariate_rows.append({
                'feature': col,
                'feature_type': 'numeric',
                'test': 'R Squared',
                'r': float(corr) if isinstance(corr, (int, float, np.floating)) else np.nan,
                'r2': r2,
                't': np.nan,
                'F': np.nan,
                'p': p_value_threshold(pval),
                'effect_size_value': d_val,
                'n_total': int(len(pair)),
                'n_groups': np.nan,
                'group_means_counts': ''
            })
        else:
            out_categorical = os.path.join(figures_dir, f"{col}_categorical_summary.png")

            # For categorical features: handle 'term' with t-test, 'home_ownership' with ANOVA
            if col == 'term':
                groups = [df[df[col] == category]['loan_status_numeric'].dropna() for category in df[col].cat.categories]
                groups = [g for g in groups if len(g) > 0]
                if len(groups) == 2:
                    t_stat, pval = stats.ttest_ind(groups[0], groups[1], equal_var=False)
                    cohend = cohen_d(groups[0], groups[1])
                    # Build group means and counts
                    gm = df.groupby(col, observed=True)['loan_status_numeric'].mean()
                    gc = df.groupby(col, observed=True)['loan_status_numeric'].size()
                    gm_str = ", ".join([f"{idx}:{val:.3f}(n={int(gc.get(idx, 0))})" for idx, val in gm.items()])
                    stats_text = f"t={t_stat:.3f}, p {p_value_threshold(pval)}\nCohen's d={cohend:.3f}\n{gm_str}"
                    print(f"{col} (term) t-test: t={t_stat:.3f}, p={p_value_threshold(pval)}, d={cohend:.3f}, {gm_str}")
                    plot_boxplot_group_means(df, col, out_categorical, stats_text)
                    bivariate_rows.append({
                        'feature': col,
                        'feature_type': 'categorical',
                        'test': 't-test',
                        't': t_stat,
                        'F': np.nan,
                        'p': p_value_threshold(pval),
                        'effect_size_value': cohend,
                        'n_total': int(sum(len(g) for g in groups)),
                        'n_groups': 2,
                        'group_means_counts': gm_str
                    })
                else:
                    gm = df.groupby(col, observed=True)['loan_status_numeric'].mean()
                    gc = df.groupby(col, observed=True)['loan_status_numeric'].size()
                    gm_str = ", ".join([f"{idx}:{val:.3f}(n={int(gc.get(idx, 0))})" for idx, val in gm.items()])
                    stats_text = f"t-test skipped: found {len(groups)} groups (need 2).\n{gm_str}"
                    print(f"Skipping t-test for '{col}': found {len(groups)} groups (need 2).")
                    plot_boxplot_group_means(df, col, out_categorical, stats_text)
                    bivariate_rows.append({
                        'feature': col,
                        'feature_type': 'categorical',
                        'test': 'group-means',
                        'r': np.nan,
                        't': np.nan,
                        'F': np.nan,
                        'p': p_value_threshold(np.nan),
                        'effect_size': np.nan,
                        'n_total': int(sum(len(g) for g in groups)),
                        'n_groups': len(groups),
                        'group_means_counts': gm_str,
                        'notes': 't-test skipped: need exactly 2 groups'
                    })

            elif col == 'home_ownership':
                groups = [df[df[col] == category]['loan_status_numeric'].dropna() for category in df[col].cat.categories]
                groups = [g for g in groups if len(g) > 0]
                if len(groups) > 1:
                    f_stat, pval = stats.f_oneway(*groups)
                    eta2 = eta_squared(groups)
                    gm = df.groupby(col, observed=True)['loan_status_numeric'].mean()
                    gc = df.groupby(col, observed=True)['loan_status_numeric'].size()
                    gm_str = ", ".join([f"{idx}:{val:.3f}(n={int(gc.get(idx, 0))})" for idx, val in gm.items()])
                    stats_text = f"ANOVA F={f_stat:.3f}, p {p_value_threshold(pval)}\neta²={eta2:.3f}\n{gm_str}"
                    print(f"{col} (home_ownership) ANOVA: F={f_stat:.3f}, p={p_value_threshold(pval)}, eta²={eta2:.3f}, {gm_str}")
                    plot_boxplot_group_means(df, col, out_categorical, stats_text)
                    bivariate_rows.append({
                        'feature': col,
                        'feature_type': 'categorical',
                        'test': 'ANOVA',
                        'r': np.nan,
                        't': np.nan,
                        'F': f_stat,
                        'p': p_value_threshold(pval),
                        'effect_size_value': eta2,
                        'n_total': int(sum(len(g) for g in groups)),
                        'n_groups': len(groups),
                        'group_means_counts': gm_str
                    })
                else:
                    gm = df.groupby(col, observed=True)['loan_status_numeric'].mean()
                    gc = df.groupby(col, observed=True)['loan_status_numeric'].size()
                    gm_str = ", ".join([f"{idx}:{val:.3f}(n={int(gc.get(idx, 0))})" for idx, val in gm.items()])
                    stats_text = f"ANOVA skipped: not enough groups.\n{gm_str}"
                    print(f"Skipping ANOVA for '{col}': not enough groups.")
                    plot_boxplot_group_means(df, col, out_categorical, stats_text)
                    bivariate_rows.append({
                        'feature': col,
                        'feature_type': 'categorical',
                        'test': 'group-means',
                        't': np.nan,
                        'F': np.nan,
                        'p': p_value_threshold(np.nan),
                        'effect_size': np.nan,
                        'n_total': int(sum(len(g) for g in groups)),
                        'n_groups': len(groups),
                        'group_means_counts': gm_str,
                        'notes': 'ANOVA skipped: need at least 2 groups'
                    })

            else:
                # Other categorical columns: show group means and skip inferential test
                gm = df.groupby(col, observed=True)['loan_status_numeric'].mean()
                gc = df.groupby(col, observed=True)['loan_status_numeric'].size()
                gm_str = ", ".join([f"{idx}:{val:.3f}(n={int(gc.get(idx, 0))})" for idx, val in gm.items()])
                stats_text = f"Group means\n{gm_str}"
                print(f"{col} group means: {gm_str}")
                plot_boxplot_group_means(df, col, out_categorical, stats_text)

                bivariate_rows.append({
                    'feature': col,
                    'feature_type': 'categorical',
                    'test': 'group-means',
                    't': np.nan,
                    'F': np.nan,
                    'p': p_value_threshold(np.nan),
                    'effect_size': np.nan,
                    'n_total': int(len(df[col].dropna())),
                    'n_groups': int(df[col].nunique(dropna=True)),
                    'group_means_counts': gm_str
                })

    # Write bivariate summary CSV, rounded to 3 decimals where numeric
    if bivariate_rows:
        bivar_df = pd.DataFrame(bivariate_rows)
        num_cols = bivar_df.select_dtypes(include=[np.number]).columns
        bivar_df[num_cols] = bivar_df[num_cols].round(3)
        bivar_path = os.path.join(tables_dir, 'bivariate_summary.csv')
        bivar_df.to_csv(bivar_path, index=False)
        # Also write a Markdown report with per-feature sections
        md_lines = [
            "# Bivariate Analysis Summary",
            "",
            f"Total features analyzed: {len(bivar_df)}",
            ""
        ]

        for _, row in bivar_df.iterrows():
            feat = str(row.get('feature', ''))
            ftype = str(row.get('feature_type', ''))
            test = str(row.get('test', ''))
            pthr = str(row.get('p', ''))  # now stores threshold label like '<0.05'
            img_rel = f"figures/{feat}_scatter_regression.png" if ftype == 'numeric' else f"figures/{feat}_categorical_summary.png"
            md_lines.append(f"## {feat}")
            md_lines.append("")
            md_lines.append(f"Test: {test}")
            md_lines.append("")
            md_lines.append(f"![{feat}]({img_rel})")
            md_lines.append("")

            if ftype == 'numeric':
                r2 = row.get('r2', np.nan)
                n_total = row.get('n_total', np.nan)
                eff_val = row.get('effect_size_value', np.nan)
                md_lines.append("| Metric | Value |")
                md_lines.append("|:---:|:---:|")
                r_val = row.get('r', np.nan)
                r_cell = "" if (r_val is None or (isinstance(r_val, float) and np.isnan(r_val))) else format(float(r_val), ".3f")
                md_lines.append("| r | " + r_cell + " |")
                md_lines.append("| R² | " + ("" if pd.isna(r2) else format(float(r2), ".3f")) + " |")
                md_lines.append("| p-value | " + pthr + " |")
                md_lines.append("| Effect size | " + ("" if pd.isna(eff_val) else format(float(eff_val), ".3f")) + " |")
                md_lines.append("| N | " + ("" if pd.isna(n_total) else str(int(n_total))) + " |")
                md_lines.append("")
            else:
                # Categorical: include test stats/effect size and a category table with means and counts
                stat_t = row.get('t', np.nan)
                stat_f = row.get('F', np.nan)
                # eff_name not used in label; always show as 'Effect size'
                eff_val = row.get('effect_size_value', np.nan)
                n_groups = row.get('n_groups', np.nan)
                n_total = row.get('n_total', np.nan)

                # Compact stats table
                md_lines.append("| Stat | Value |")
                md_lines.append("|:---:|:---:|")
                if not pd.isna(stat_t):
                    md_lines.append(f"| t | {float(stat_t):.3f} |")
                if not pd.isna(stat_f):
                    md_lines.append(f"| F | {float(stat_f):.3f} |")
                md_lines.append("| p-value | " + pthr + " |")
                if not pd.isna(eff_val):
                    md_lines.append("| Effect size | " + ("" if pd.isna(eff_val) else format(float(eff_val), ".3f")) + " |")
                if not pd.isna(n_groups):
                    md_lines.append(f"| Groups | {int(n_groups)} |")
                if not pd.isna(n_total):
                    md_lines.append(f"| N | {int(n_total)} |")
                md_lines.append("")

                # Parse group means/counts to make columns as headers
                gm_str = str(row.get('group_means_counts', '')).strip()
                if gm_str:
                    cols = []
                    vals = []
                    for seg in [s.strip() for s in gm_str.split(',') if s.strip()]:
                        parts = seg.split(':', 1)
                        if len(parts) != 2:
                            continue
                        cat = parts[0].strip()
                        rest = parts[1].strip()
                        mean_val = ''
                        n_val = ''
                        if '(' in rest and rest.endswith(')'):
                            try:
                                mean_part, n_part = rest.split('(', 1)
                                mean_val = f"{float(mean_part):.3f}" if mean_part.strip() else ''
                                n_inner = n_part[:-1]  # drop trailing ')'
                                n_val = n_inner.split('=')[-1].strip()
                            except Exception:
                                mean_val = rest
                                n_val = ''
                        else:
                            try:
                                mean_val = f"{float(rest):.3f}"
                            except Exception:
                                mean_val = rest
                        cols.append(cat)
                        if n_val:
                            vals.append(f"{mean_val} (n={n_val})")
                        else:
                            vals.append(f"{mean_val}")

                    if cols and vals:
                        # Header row with categories
                        md_lines.append("| " + " | ".join(cols) + " |")
                        md_lines.append("| " + " | ".join([":---:"] * len(cols)) + " |")
                        md_lines.append("| " + " | ".join(vals) + " |")
                        md_lines.append("")

            # Separator between features
            md_lines.append("---")
            md_lines.append("")

        md_path = os.path.join(tables_dir, 'bivariate_summary.md')
        with open(md_path, 'w') as f:
            f.write("\n".join(md_lines))

    print(f"Bivariate analysis completed. Summaries saved to '{os.path.join(tables_dir,'bivariate_summary.csv')}' and '{os.path.join(tables_dir,'bivariate_summary.md')}'.")

if __name__ == "__main__":
    main()
