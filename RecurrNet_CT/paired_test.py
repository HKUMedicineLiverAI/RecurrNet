from itertools import combinations
import warnings

import numpy as np
from scipy import stats
import pandas as pd

from lifelines import utils
from lifelines.utils import (
	group_survival_table_from_events,
	string_rjustify,
	format_p_value,
	format_floats,
	# interpolate_at_times_and_return_pandas,
	_expected_value_of_survival_up_to_t,
	_expected_value_of_survival_squared_up_to_t,
)

def _to_1d_array(x): #  -> np.ndarray
	v = np.atleast_1d(x)
	try:
		if v.shape[0] > 1 and v.shape[1] > 1:
			raise ValueError("Wrong shape (2d) given to _to_1d_array")
	except IndexError:
		pass
	return v
def interpolate_at_times(df_or_series, new_times): #  -> ndarray
	"""


	Parameters
	-----------
	df_or_series: Pandas DataFrame or Series
	new_times: iterable, numpy array
		 can be a n-d array, list.

	"""

	assert df_or_series.index.is_monotonic_increasing, "df_or_series must be a increasing index."
	t = df_or_series.index.values
	y = df_or_series.values.reshape(df_or_series.shape[0])
	return np.interp(new_times, t, y)


def interpolate_at_times_and_return_pandas(df_or_series, new_times): #  -> Union[pd.Series, pd.DataFrame]
	"""
	TODO
	"""
	new_times = _to_1d_array(new_times)
	try:
		cols = df_or_series.columns
	except AttributeError:
		cols = None

	return pd.DataFrame(interpolate_at_times(df_or_series, new_times), index=new_times, columns=cols).squeeze()


# string_rjustify = lambda width: lambda s: s.rjust(width, " ")
# string_ljustify = lambda width: lambda s: s.ljust(width, " ")

from lifelines import KaplanMeierFitter

# def survival_difference_at_fixed_point_in_time_test(point_in_time, fitterA, fitterB, **result_kwargs): # -> StatisticalResult
def survival_difference_at_fixed_point_in_time_test_for_paired_data(point_in_time, ymvi1_t, ymvi1_s,yp1_t, yp1_s, **result_kwargs): # -> StatisticalResult
	"""
	Often analysts want to compare the survival-ness of groups at specific times, rather than comparing the entire survival curves against each other.
	For example, analysts may be interested in 5-year survival. Statistically comparing the naive Kaplan-Meier points at a specific time
	actually has reduced power (see [1]). By transforming the survival function, we can recover more power. This function uses
	the log(-log(·)) transformation.


	Parameters
	----------
	point_in_time: float,
		the point in time to analyze the survival curves at.

	fitterA:
		A lifelines univariate model fitted to the data. This can be a ``KaplanMeierFitter``, ``WeibullFitter``, etc.

	fitterB:
		the second lifelines model to compare against.

	result_kwargs:
		add keywords and meta-data to the experiment summary

	Returns
	-------

	StatisticalResult
	  a StatisticalResult object with properties ``p_value``, ``summary``, ``test_statistic``, ``print_summary``

	Examples
	--------
	.. code:: python

		T1 = [1, 4, 10, 12, 12, 3, 5.4]
		E1 = [1, 0, 1,  0,  1,  1, 1]
		kmf1 = KaplanMeierFitter().fit(T1, E1)

		T2 = [4, 5, 7, 11, 14, 20, 8, 8]
		E2 = [1, 1, 1, 1,  1,  1,  1, 1]
		kmf2 = KaplanMeierFitter().fit(T2, E2)

		from lifelines.statistics import survival_difference_at_fixed_point_in_time_test
		results = survival_difference_at_fixed_point_in_time_test(12.0, kmf1, kmf2)

		results.print_summary()
		print(results.p_value)        # 0.77
		print(results.test_statistic) # 0.09

	Notes
	-----
	1. Other transformations are possible, but Klein et al. [1] showed that the log(-log(·)) transform has the most desirable
	statistical properties.

	2. The API of this function changed in v0.25.3. This new API allows for right, left and interval censoring models to be tested.


	References
	-----------

	[1] Klein, J. P., Logan, B. , Harhoff, M. and Andersen, P. K. (2007), Analyzing survival curves at a fixed point in time. Statist. Med., 26: 4505-4519. doi:10.1002/sim.2864

	"""
	paired_t = []
	paired_s = []
	ymvi1_t, ymvi1_s = (list(t) for t in zip(*sorted(zip(ymvi1_t, ymvi1_s))))
	yp1_t, yp1_s = (list(t) for t in zip(*sorted(zip(yp1_t, yp1_s))))
	for j in range(len(ymvi1_t)):
		for k in range(len(yp1_t)):
			if ymvi1_t[j] == yp1_t[k] and ymvi1_s[j] == yp1_s[k]:
				paired_t.append(ymvi1_t[j])
				paired_s.append(ymvi1_s[j])
	fitterA = KaplanMeierFitter().fit(ymvi1_t, ymvi1_s)
	fitterB = KaplanMeierFitter().fit(yp1_t, yp1_s)
	n = max(len(ymvi1_s),len(yp1_s))

	if type(fitterB) != type(fitterA):
		warnings.warn(
			"This test compares survival functions, but your fitters are estimating the survival functions differently. This means that this test is also testing the different ways to estimate the survival function and will be unreliable.",
			UserWarning,
		)

	log = np.log
	clog = lambda s: log(-log(s))

	sA_t = fitterA.predict(point_in_time)
	sB_t = fitterB.predict(point_in_time)

	from lifelines.fitters import NonParametricUnivariateFitter, ParametricUnivariateFitter

	if isinstance(fitterA, NonParametricUnivariateFitter):
		sigma_sqA = interpolate_at_times_and_return_pandas(fitterA._cumulative_sq_, point_in_time)
	elif isinstance(fitterA, ParametricUnivariateFitter):
		sigma_sqA = fitterA._compute_variance_of_transform(fitterA._survival_function, [point_in_time]).squeeze()

	if isinstance(fitterB, NonParametricUnivariateFitter):
		sigma_sqB = interpolate_at_times_and_return_pandas(fitterB._cumulative_sq_, point_in_time)
	elif isinstance(fitterB, ParametricUnivariateFitter):
		sigma_sqB = fitterB._compute_variance_of_transform(fitterB._survival_function, [point_in_time]).squeeze()
	# print(clog(sA_t))
	# print((clog(sA_t) - clog(sB_t)) ** 2)
	# print((sigma_sqA / log(sA_t) ** 2 + sigma_sqB / log(sB_t) ** 2))


	# X = (clog(sA_t) - clog(sB_t)) ** 2 / (sigma_sqA / log(sA_t) ** 2 + sigma_sqB / log(sB_t) ** 2)

	T1 = ymvi1_t
	Y1 = list(map(int, ymvi1_s))
	T2 = yp1_t
	Y2 = list(map(int, yp1_s))
	PT = paired_t
	PY = list(map(int, paired_s))

	if len(PT) != 0:
		G12_sum = 0.0
		for r in range(len(T1)):
			if T1[r] > point_in_time:
				break
			for s in range(len(T2)):
				if T2[s] > point_in_time:
					break
				pi1r = sum(Y1[r:])
				pi2s = sum(Y2[s:])
				for k in range(len(PT)):
					if PT[k] >= max(T1[r],T2[s]):
						break
				pirs = sum(PY[k:])
				if pirs == 0:
					break

				if T1[r] == T2[s]:
					qrs = 1
				else:
					qrs = 0
				if Y2[s] == 1:
					for k in range(len(PT)):
						qs_r = 0
						if PT[k] >= T1[r] and T1[r] <= T2[s] and PY[k]==1:
							qs_r += 1
				else:
					qs_r = 0

				if Y1[r] == 1:
					for k in range(len(PT)):
						qr_s = 0
						if PT[k] >= T2[s] and T2[s] <= T1[r] and PY[k]==1:
							qr_s += 1
				else:
					qr_s = 0

				if Y1[r] == 1:
					q1r = 1
				else:
					q1r = 0

				if Y2[s] == 1:
					q2s = 1
				else:
					q2s = 0

				# print(pi1r)
				# print(pi2s)
				# print(pirs)

				G12 = (pirs*n)/(pi1r*pi2s) * (qrs/pirs - (qr_s*q2s)/(pirs*pi2s) - (qs_r*q1r)/(pirs*pi1r) + (q1r*q2s)/(pi1r*pi2s))

				G12_sum += G12

		cov = G12_sum/n
	else:
		cov = 0



	# cov = (/n) / (/n**2)
	X = (sA_t - sB_t) ** 2 / (sA_t ** 2 * sigma_sqA ** 2 + sB_t ** 2 * sigma_sqB ** 2) - 2*cov
	# res = stats.chi2_contingency(np.array([[0.7142857142857143, 1-0.7142857142857143], [1.0, 0.0]]))
	# print(res[1])
	p_value = stats.chi2.sf(X, 1)

	print(p_value)
	print(X)

	return p_value

	# return StatisticalResult(
	#     p_value,
	#     X,
	#     null_distribution="chi squared",
	#     degrees_of_freedom=1,
	#     point_in_time=point_in_time,
	#     test_name="survival_difference_at_fixed_point_in_time_test",
	#     fitterA=fitterA,
	#     fitterB=fitterB,
	#     **result_kwargs
	# )

def _chisq_test_p_value(U, degrees_freedom) -> float:
	p_value = stats.chi2.sf(U, degrees_freedom)
	return p_value


# # T1 = [1, 4, 10, 12, 12, 3, 5.4]
# # E1 = [1, 0, 1,  0,  1,  1, 1]
# T1 = [1.435616438, 0.473972603, 10.04788732, 5.002739726, 2.18630137, 0.345205479, 1.090410959, 3.01369863, 0.230136986, 0.353424658, 0.745205479, 0.430136986, 0.545205479, 0.123287671, 0.345205479, 0.260273973, 1.715068493, 5.354929577, 2.736986301, 0.252054795, 4.095774648, 2.679452055, 0.326027397, 2.821917808, 1.58630137, 12.49577465, 0.816438356, 12.31830986, 0.616438356, 0.750684932, 5.307042254, 13.14929577, 5.430985915]
# E1 = [True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, False, True, True, True, True, False, True, False, True, True, False, False, False]
# kmf1 = KaplanMeierFitter().fit(T1, E1)

# T2 = [1.893150685, 0.652054795, 12.91267606, 0.320547945, 0.334246575, 0.364383562, 0.473972603, 0.287671233, 2.052054795, 0.345205479, 1.090410959, 1.052054795, 0.230136986, 0.353424658, 0.745205479, 0.430136986, 0.123287671, 0.260273973, 3.346478873, 1.715068493, 4.411267606, 0.252054795, 0.769863014, 1.016438356, 0.208219178, 0.326027397, 0.394520548, 2.821917808, 0.632876712, 0.249315068, 0.816438356, 0.509589041, 0.616438356, 0.843835616, 0.750684932, 0.178082192]
# E2 = [False, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, False, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True]
# # T2 = [1, 2, 3, 4]
# # E2 = [1, 0, 1, 0]
# kmf2 = KaplanMeierFitter().fit(T2, E2)
# # from lifelines.statistics import survival_difference_at_fixed_point_in_time_test
# # p_value,X = survival_difference_at_fixed_point_in_time_test(3, kmf1, kmf2)
# p_value = survival_difference_at_fixed_point_in_time_test_for_paired_data(3, T1,E1,T2,E2)

# results.print_summary()
# print(results.p_value)        # 0.77
# print(results.test_statistic) # 0.09


# # Example of calculating the mcnemar test
# from statsmodels.stats.contingency_tables import mcnemar
# # define contingency table
# table = [[4, 2],
#  [1, 3]]
# # calculate mcnemar test
# result = mcnemar(table, exact=True)
# # summarize the finding
# print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
# # interpret the p-value
# alpha = 0.05
# if result.pvalue > alpha:
#  print('Same proportions of errors (fail to reject H0)')
# else:
#  print('Different proportions of errors (reject H0)')
