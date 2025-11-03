import os
import numpy as np
import pandas as pd


ASSIGNMENT_COLS = [
	'loan_status_numeric',
	'loan_amnt',
	'term',
	'int_rate',
	'home_ownership',
	'dti',
]


def ss_total(y: np.ndarray) -> float:
	y = y[~np.isnan(y)]
	if y.size == 0:
		return np.nan
	ym = np.nanmean(y)
	return float(np.nansum((y - ym) ** 2))


def regression_anova_from_corr(y: np.ndarray, r: float) -> dict:
	"""Return between/within/total ANOVA-like table for simple regression.

	between ~ regression (SSR), within ~ residual (SSE), total ~ SST
	DF_between=1, DF_within=n-2, DF_total=n-1
	corr_val = R^2 (on 'between' row) to match visuals
	"""
	y_clean = y[~np.isnan(y)]
	n = y_clean.size
	if n < 2 or r is None or not np.isfinite(r):
		return {
			'between': {'DF': np.nan, 'SS': np.nan, 'MS': np.nan, 'corr_val': ''},
			'within': {'DF': np.nan, 'SS': np.nan, 'MS': np.nan, 'corr_val': ''},
			'total': {'DF': np.nan, 'SS': np.nan, 'MS': np.nan, 'corr_val': ''},
		}

	sst = ss_total(y_clean)
	# SSR = r^2 * SST (simple linear regression)
	r2 = float(r) ** 2
	ssr = r2 * sst if np.isfinite(sst) else np.nan
	sse = sst - ssr if np.isfinite(ssr) else np.nan

	df_between = 1
	df_within = max(n - 2, 0)
	df_total = max(n - 1, 0)

	ms_between = ssr / df_between if df_between > 0 and np.isfinite(ssr) else np.nan
	ms_within = sse / df_within if df_within > 0 and np.isfinite(sse) else np.nan
	ms_total = sst / df_total if df_total > 0 and np.isfinite(sst) else np.nan

	return {
		'between': {'DF': df_between, 'SS': ssr, 'MS': ms_between, 'corr_val': r2},
		'within': {'DF': df_within, 'SS': sse, 'MS': ms_within, 'corr_val': ''},
		'total': {'DF': df_total, 'SS': sst, 'MS': ms_total, 'corr_val': ''},
	}


def anova_for_categories(y: np.ndarray, groups: list[np.ndarray]) -> dict:
	"""Return between/within/total ANOVA table for categorical predictor.

	corr_val set on 'between' row to eta^2 (effect size used in the project).
	"""
	# Clean groups and y are already numeric arrays with NaNs handled by caller
	valid_groups = [g[~np.isnan(g)] for g in groups if g.size > 0]
	if len(valid_groups) < 2:
		return {
			'between': {'DF': np.nan, 'SS': np.nan, 'MS': np.nan, 'corr_val': ''},
			'within': {'DF': np.nan, 'SS': np.nan, 'MS': np.nan, 'corr_val': ''},
			'total': {'DF': np.nan, 'SS': np.nan, 'MS': np.nan, 'corr_val': ''},
		}

	all_data = np.concatenate(valid_groups)
	n = all_data.size
	if n == 0:
		return {
			'between': {'DF': np.nan, 'SS': np.nan, 'MS': np.nan, 'corr_val': ''},
			'within': {'DF': np.nan, 'SS': np.nan, 'MS': np.nan, 'corr_val': ''},
			'total': {'DF': np.nan, 'SS': np.nan, 'MS': np.nan, 'corr_val': ''},
		}

	grand = np.nanmean(all_data)
	ss_between = float(sum(len(g) * (np.nanmean(g) - grand) ** 2 for g in valid_groups))
	ss_total = float(np.nansum((all_data - grand) ** 2))
	ss_within = ss_total - ss_between

	k = len(valid_groups)
	df_between = k - 1
	df_within = n - k
	df_total = n - 1

	ms_between = ss_between / df_between if df_between > 0 else np.nan
	ms_within = ss_within / df_within if df_within > 0 else np.nan
	ms_total = ss_total / df_total if df_total > 0 else np.nan

	eta2 = (ss_between / ss_total) if ss_total > 0 else np.nan

	return {
		'between': {'DF': df_between, 'SS': ss_between, 'MS': ms_between, 'corr_val': eta2},
		'within': {'DF': df_within, 'SS': ss_within, 'MS': ms_within, 'corr_val': ''},
		'total': {'DF': df_total, 'SS': ss_total, 'MS': ms_total, 'corr_val': ''},
	}


def cohen_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
	"""Compute Cohen's d between two 1-D numeric arrays."""
	a = group_a[~np.isnan(group_a)]
	b = group_b[~np.isnan(group_b)]
	n1, n2 = a.size, b.size
	if n1 < 2 or n2 < 2:
		return np.nan
	m1, m2 = np.nanmean(a), np.nanmean(b)
	s1, s2 = np.nanstd(a, ddof=1), np.nanstd(b, ddof=1)
	pooled = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
	return (m1 - m2) / pooled if pooled > 0 else np.nan


def main():
	# Use repo-relative paths (run from repo root)
	cwd = os.getcwd()
	csv_path = os.path.join(cwd, 'lc_large.csv')
	# write outputs into a tables/ directory
	tables_dir = os.path.join(cwd, 'tables')
	os.makedirs(tables_dir, exist_ok=True)
	out_xlsx = os.path.join(tables_dir, 'anova_tables.xlsx')

	df = pd.read_csv(csv_path)
	# Keep only relevant columns
	df = df[[c for c in df.columns if c in ASSIGNMENT_COLS]].copy()

	# Cast categories similar to main.py
	for cat_col in ['term', 'home_ownership']:
		if cat_col in df.columns:
			df[cat_col] = df[cat_col].astype('category')

	if 'loan_status_numeric' not in df.columns:
		raise RuntimeError('loan_status_numeric column not found')

	y = pd.to_numeric(df['loan_status_numeric'], errors='coerce').to_numpy()

	# Only produce sheets for 'term' (t-stat) and 'home_ownership' (F-stat)
	features = [f for f in ['term', 'home_ownership'] if f in df.columns]
	with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:
		for col in features:
			cats = df[col]
			groups = []
			if isinstance(cats.dtype, pd.CategoricalDtype):
				for cat in cats.cat.categories:
					vals = df.loc[cats == cat, 'loan_status_numeric']
					grp = np.asarray(pd.to_numeric(vals, errors='coerce'))
					groups.append(grp)
			else:
				for cat in pd.unique(cats.dropna()):
					vals = df.loc[cats == cat, 'loan_status_numeric']
					grp = np.asarray(pd.to_numeric(vals, errors='coerce'))
					groups.append(grp)

			table = anova_for_categories(y, groups)

			# For 'term': store Cohen's d in corr_val and t-stat/p-value in stat/p_val
			if col == 'term':
				non_empty = [g for g in groups if g.size > 0]
				if len(non_empty) == 2:
					from scipy import stats as _stats
					# Cohen's d between the two term groups
					dval = cohen_d(non_empty[0], non_empty[1])
					table['between']['corr_val'] = dval if np.isfinite(dval) else np.nan
					# Welch t-test for t-stat and p-value
					t_res = _stats.ttest_ind(non_empty[0], non_empty[1], equal_var=False)
					try:
						t_stat = getattr(t_res, 'statistic', None)
						p_val = getattr(t_res, 'pvalue', None)
						if t_stat is None:
							t_stat = t_res[0]
						if p_val is None:
							p_val = t_res[1]
						t_stat = float(t_stat)
						p_val = float(p_val)
					except Exception:
						t_stat = np.nan
						p_val = np.nan
					table['between']['stat'] = t_stat if np.isfinite(t_stat) else np.nan
					table['between']['p_val'] = p_val if np.isfinite(p_val) else np.nan
				else:
					table['between']['corr_val'] = ''

			# For 'home_ownership': place eta^2 in corr_val and F-stat/p-value in stat/p_val
			if col == 'home_ownership':
				valid_groups = [g for g in groups if g.size > 0]
				if len(valid_groups) > 1:
					from scipy import stats as _stats
					# eta2 already computed by anova_for_categories and stored in corr_val
					eta2 = table['between'].get('corr_val', np.nan)
					table['between']['corr_val'] = eta2 if np.isfinite(eta2) else np.nan
					f_res = _stats.f_oneway(*valid_groups)
					try:
						f_stat = getattr(f_res, 'statistic', None)
						p_val = getattr(f_res, 'pvalue', None)
						if f_stat is None:
							f_stat = f_res[0]
						if p_val is None:
							p_val = f_res[1]
						f_stat = float(f_stat)
						p_val = float(p_val)
					except Exception:
						f_stat = np.nan
						p_val = np.nan
					table['between']['stat'] = f_stat if np.isfinite(f_stat) else np.nan
					table['between']['p_val'] = p_val if np.isfinite(p_val) else np.nan
				else:
					table['between']['corr_val'] = ''

			# Build DataFrame in requested layout and include p-value and separate stat column
			rows = ['between', 'within', 'total']
			data = {
				'DF': [table[r]['DF'] for r in rows],
				'SS': [table[r]['SS'] for r in rows],
				'MS': [table[r]['MS'] for r in rows],
				# corr_val will store effect size (eta^2 for home_ownership, Cohen's d for term)
				'corr_val': [table[r].get('corr_val', '') for r in rows],
				# stat will store the test statistic (t or F) on the between row
				'stat': [table[r].get('stat', '') for r in rows],
				'p_val': [table[r].get('p_val', '') for r in rows],
			}
			out_df = pd.DataFrame(data, index=rows)
			# Round numeric columns for readability: SS/MS/corr/stat to 3 decimals, p_val to 6
			for c in ['SS', 'MS', 'corr_val', 'stat']:
				if c in out_df.columns:
					out_df[c] = pd.to_numeric(out_df[c], errors='coerce').round(3)
			if 'p_val' in out_df.columns:
				out_df['p_val'] = pd.to_numeric(out_df['p_val'], errors='coerce').round(6)
			safe_name = str(col)[:31]
			out_df.to_excel(writer, sheet_name=safe_name)

	print(f"Wrote ANOVA-style tables to {out_xlsx}")


if __name__ == '__main__':
	main()

