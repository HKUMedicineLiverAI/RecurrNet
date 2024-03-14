def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import os
import numpy as np
import pandas as pd
# import xgboost as xgb
import random 
import scipy.stats
import operator
import pickle
# import pickle5 as pickle
import sksurv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
import matplotlib.pyplot as plt
from sksurv.datasets import load_whas500
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn.utils import check_consistent_length
import pickle
import matplotlib.pyplot as plt
from sksurv.nonparametric import kaplan_meier_estimator
from sklearn.impute import KNNImputer
from sklearn.pipeline import make_pipeline
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.preprocessing import OneHotEncoder, encode_categorical
from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv

from sklearn import set_config
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from sksurv.datasets import load_gbsg2
from sksurv.preprocessing import OneHotEncoder
from sksurv.ensemble import RandomSurvivalForest
from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM
from sksurv.metrics import concordance_index_censored
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sksurv.metrics import concordance_index_censored
from sklearn.metrics import auc, accuracy_score, precision_score,confusion_matrix, mean_squared_error, recall_score, classification_report, roc_curve
from paired_test import survival_difference_at_fixed_point_in_time_test_for_paired_data
import math


def _compute_counts(event, time, order=None):
    n_samples = event.shape[0]

    if order is None:
        order = np.argsort(time, kind="mergesort")

    uniq_times = np.empty(n_samples, dtype=time.dtype)
    uniq_events = np.empty(n_samples, dtype=int)
    uniq_counts = np.empty(n_samples, dtype=int)

    i = 0
    prev_val = time[order[0]]
    j = 0
    while True:
        count_event = 0
        count = 0
        while i < n_samples and prev_val == time[order[i]]:
            if event[order[i]]:
                count_event += 1

            count += 1
            i += 1

        uniq_times[j] = prev_val
        uniq_events[j] = count_event
        uniq_counts[j] = count
        j += 1

        if i == n_samples:
            break

        prev_val = time[order[i]]

    times = np.resize(uniq_times, j)
    n_events = np.resize(uniq_events, j)
    total_count = np.resize(uniq_counts, j)
    n_censored = total_count - n_events

    # offset cumulative sum by one
    total_count = np.r_[0, total_count]
    n_at_risk = n_samples - np.cumsum(total_count)

    return times, n_events, n_at_risk[:-1], n_censored



class StepFunction:
    def __init__(self, x, y, *, a=1.0, b=0.0, domain=(0, None)):
        check_consistent_length(x, y)
        self.x = x
        self.y = y
        self.a = a
        self.b = b
        domain_lower = self.x[0] if domain[0] is None else domain[0]
        domain_upper = self.x[-1] if domain[1] is None else domain[1]
        self._domain = (float(domain_lower), float(domain_upper))

    @property
    def domain(self):
        return self._domain

    def __call__(self, x):
        x = np.atleast_1d(x)
        if not np.isfinite(x).all():
            raise ValueError("x must be finite")
        if np.min(x) < self._domain[0] or np.max(x) > self.domain[1]:
            raise ValueError(f"x must be within [{self.domain[0]:f}; {self.domain[1]:f}]")

        # x is within the domain, but we need to account for self.domain[0] <= x < self.x[0]
        x = np.clip(x, a_min=self.x[0], a_max=None)

        i = np.searchsorted(self.x, x, side="left")
        not_exact = self.x[i] != x
        i[not_exact] -= 1
        value = self.a * self.y[i] + self.b
        if value.shape[0] == 1:
            return value[0]
        return value

    def __repr__(self):
        return f"StepFunction(x={self.x!r}, y={self.y!r}, a={self.a!r}, b={self.b!r})"

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return all(self.x == other.x) and all(self.y == other.y) and self.a == other.a and self.b == other.b
        return False

class BreslowEstimator:
    def fit(self, linear_predictor, event, time):
        risk_score = np.exp(linear_predictor)
        order = np.argsort(time, kind="mergesort")
        risk_score = risk_score[order]
        uniq_times, n_events, n_at_risk, _ = _compute_counts(event, time, order)

        divisor = np.empty(n_at_risk.shape, dtype=float)
        value = np.sum(risk_score)
        divisor[0] = value
        k = 0
        for i in range(1, len(n_at_risk)):
            d = n_at_risk[i - 1] - n_at_risk[i]
            value -= risk_score[k : (k + d)].sum()
            k += d
            divisor[i] = value

        assert k == n_at_risk[0] - n_at_risk[-1]

        y = np.cumsum(n_events / divisor)
        self.cum_baseline_hazard_ = StepFunction(uniq_times, y)
        self.baseline_survival_ = StepFunction(uniq_times, np.exp(-y))
        self.unique_times_ = uniq_times
        return self

    # def get_cumulative_hazard_function(self, linear_predictor):
    #     risk_score = np.exp(linear_predictor)
    #     n_samples = risk_score.shape[0]
    #     funcs = np.empty(n_samples, dtype=object)
    #     for i in range(n_samples):
    #         funcs[i] = StepFunction(x=self.cum_baseline_hazard_.x, y=self.cum_baseline_hazard_.y, a=risk_score[i])
    #     return funcs

    def get_survival_function(self, linear_predictor):
        risk_score = np.exp(linear_predictor)
        n_samples = risk_score.shape[0]
        funcs = np.empty(n_samples, dtype=object)
        for i in range(n_samples):
            funcs[i] = StepFunction(x=self.baseline_survival_.x, y=np.power(self.baseline_survival_.y, risk_score[i]))
        return funcs


def _predict_function(func_name, baseline_model, prediction, return_array=False):
    fns = getattr(baseline_model, func_name)(prediction)

    if not return_array:
        return fns

    times = baseline_model.unique_times_
    arr = np.empty((prediction.shape[0], times.shape[0]), dtype=float)
    for i, fn in enumerate(fns):
        arr[i, :] = fn(times)
    return arr

if __name__ == '__main__':
    # X, y = load_whas500()
    # X = X.astype(float)
    # estimator = CoxPHSurvivalAnalysis().fit(X, y)
    # e = [i[0] for i in y]
    # t = [i[1] for i in y]
    # _baseline_model = BreslowEstimator().fit(estimator.predict(X),np.asarray(e),np.asarray(t))
    # # surv_funcs = estimator.predict_survival_function(X.iloc[:10])
    # surv_funcs = _predict_function("get_survival_function", _baseline_model, estimator.predict(X.iloc[:1]))

    # for fn in surv_funcs:
    #     plt.step(fn.x, fn(fn.x), where="post")
    #     plt.ylim(0, 1)
    #     plt.show()

    import numpy as np
    from matplotlib import pyplot as plt
    import scipy.stats as st
    from sklearn import metrics
    from lifelines.statistics import multivariate_logrank_test
    from lifelines.statistics import logrank_test
    import copy
    from scipy.stats import chi2
    from lifelines.statistics import survival_difference_at_fixed_point_in_time_test
    from lifelines import KaplanMeierFitter

    class DelongTest():
        def __init__(self,preds1,preds2,label,threshold=0.05):
            self._preds1=preds1
            self._preds2=preds2
            self._label=label
            self.threshold=threshold
            self._show_result()

        def _auc(self,X, Y)->float:
            return 1/(len(X)*len(Y)) * sum([self._kernel(x, y) for x in X for y in Y])

        def _kernel(self,X, Y)->float:
            '''
            Mann-Whitney statistic
            '''
            return .5 if Y==X else int(Y < X)

        def _structural_components(self,X, Y)->list:
            V10 = [1/len(Y) * sum([self._kernel(x, y) for y in Y]) for x in X]
            V01 = [1/len(X) * sum([self._kernel(x, y) for x in X]) for y in Y]
            return V10, V01

        def _get_S_entry(self,V_A, V_B, auc_A, auc_B)->float:
            return 1/(len(V_A)-1) * sum([(a-auc_A)*(b-auc_B) for a,b in zip(V_A, V_B)])
        
        def _z_score(self,var_A, var_B, covar_AB, auc_A, auc_B):
            return (auc_A - auc_B)/((var_A + var_B - 2*covar_AB )**(.5)+ 1e-8)

        def _group_preds_by_label(self,preds, actual)->list:
            X = [p for (p, a) in zip(preds, actual) if a]
            Y = [p for (p, a) in zip(preds, actual) if not a]
            return X, Y

        def _compute_z_p(self):
            X_A, Y_A = self._group_preds_by_label(self._preds1, self._label)
            X_B, Y_B = self._group_preds_by_label(self._preds2, self._label)

            V_A10, V_A01 = self._structural_components(X_A, Y_A)
            V_B10, V_B01 = self._structural_components(X_B, Y_B)

            auc_A = self._auc(X_A, Y_A)
            auc_B = self._auc(X_B, Y_B)

            # Compute entries of covariance matrix S (covar_AB = covar_BA)
            var_A = (self._get_S_entry(V_A10, V_A10, auc_A, auc_A) * 1/len(V_A10)+ self._get_S_entry(V_A01, V_A01, auc_A, auc_A) * 1/len(V_A01))
            var_B = (self._get_S_entry(V_B10, V_B10, auc_B, auc_B) * 1/len(V_B10)+ self._get_S_entry(V_B01, V_B01, auc_B, auc_B) * 1/len(V_B01))
            covar_AB = (self._get_S_entry(V_A10, V_B10, auc_A, auc_B) * 1/len(V_A10)+ self._get_S_entry(V_A01, V_B01, auc_A, auc_B) * 1/len(V_A01))

            # Two tailed test
            z = self._z_score(var_A, var_B, covar_AB, auc_A, auc_B)
            p = st.norm.sf(abs(z))*2

            return z,p

        def _show_result(self):
            z,p=self._compute_z_p()

    def bootstrap_auc(y, pred, y1,bootstraps = 1000, fold_size = 50):
        statistics = []
        statistics_PPV = []
        statistics_NPV = []
        statistics_sen = []
        statistics_spc = []
          # for c in range(len(classes)):
        df = pd.DataFrame(columns=['y', 'pred', 'y1'])
        df.loc[:, 'y'] = y
        df.loc[:, 'pred'] = pred
        df.loc[:, 'y1'] = y1
        df_pos = df[df.y == 1]
        df_neg = df[df.y == 0]
        prevalence = len(df_pos) / len(df)
        for i in range(bootstraps):
            pos_sample = df_pos.sample(n = int(fold_size * prevalence), replace=True)
            neg_sample = df_neg.sample(n = int(fold_size * (1-prevalence)), replace=True)
            y_sample = np.concatenate([pos_sample.y.values, neg_sample.y.values])
            pred_sample = np.concatenate([pos_sample.pred.values, neg_sample.pred.values])
            y1_sample = np.concatenate([pos_sample.y1.values, neg_sample.y1.values])

            fpr, tpr, thresholds = roc_curve(y_sample,y1_sample,pos_label=False)
            score = auc(fpr, tpr)

            y_sample = y_sample.astype(int)
            pred_sample = pred_sample.astype(int)
            tn, fp, fn, tp = metrics.confusion_matrix(y_sample, pred_sample).ravel()
            ppv = tp / (tp + fp)
            npv = tn / (tn + fn)
            sen = tp / (tp + fn)
            spc = tn / (tn + fp)
              
            statistics.append(score)
            statistics_PPV.append(ppv)
            statistics_NPV.append(npv)
            statistics_sen.append(sen)
            statistics_spc.append(spc)

        CI95 = np.percentile(statistics, (2.5, 97.5))
        CI95_PPV = np.percentile(statistics_PPV, (2.5, 97.5))
        CI95_NPV = np.percentile(statistics_NPV, (2.5, 97.5))
        CI95_sen = np.percentile(statistics_sen, (2.5, 97.5))
        CI95_spc = np.percentile(statistics_spc, (2.5, 97.5))
        return str(CI95),str(CI95_PPV),str(CI95_NPV),str(CI95_sen),str(CI95_spc)


    def bootstrap_auc_MVI(y, pred, bootstraps = 1000, fold_size = 50):
        statistics = []
        statistics_PPV = []
        statistics_NPV = []
        statistics_sen = []
        statistics_spc = []
        df = pd.DataFrame(columns=['y', 'pred'])
        df.loc[:, 'y'] = y
        df.loc[:, 'pred'] = pred
        df_pos = df[df.y == 1]
        df_neg = df[df.y == 0]
        prevalence = len(df_pos) / len(df)
        for i in range(bootstraps):
            pos_sample = df_pos.sample(n = int(fold_size * prevalence), replace=True)
            neg_sample = df_neg.sample(n = int(fold_size * (1-prevalence)), replace=True)
            y_sample = np.concatenate([pos_sample.y.values, neg_sample.y.values])
            pred_sample = np.concatenate([pos_sample.pred.values, neg_sample.pred.values])


            score = metrics.roc_auc_score(y_sample, pred_sample)
            y_sample = y_sample.astype(int)
            pred_sample = pred_sample.astype(int)
            tn, fp, fn, tp = metrics.confusion_matrix(y_sample, pred_sample).ravel()
            ppv = tp / (tp + fp)
            npv = tn / (tn + fn)
            sen = tp / (tp + fn)
            spc = tn / (tn + fp)
              
            statistics.append(score)
            statistics_PPV.append(ppv)
            statistics_NPV.append(npv)
            statistics_sen.append(sen)
            statistics_spc.append(spc)

        CI95 = np.percentile(statistics, (2.5,97.5))
        CI95_PPV = np.percentile(statistics_PPV, (2.5,97.5))
        CI95_NPV = np.percentile(statistics_NPV, (2.5,97.5))
        CI95_sen = np.percentile(statistics_sen, (2.5, 97.5))
        CI95_spc = np.percentile(statistics_spc, (2.5, 97.5))
        return str(CI95),str(CI95_PPV),str(CI95_NPV),str(CI95_sen),str(CI95_spc)

    df1 = pd.read_csv('rs.csv',header=0)
    df2 = pd.read_csv('wh_up_final_death.csv',header=0)
    df = df1.merge(df2, how='inner', on='Pseudo ID')

    with open('sl_A.pkl', 'rb') as f:
    	sl = pickle.load(f)
    train_list = sl['train']
    test_list = sl['test']
    val_list = sl['val']

    train_prediction = df[df['Pseudo ID'].isin(train_list)]['rs'].values
    test_prediction = df[df['Pseudo ID'].isin(test_list)]['rs'].values
    val_prediction = df[df['Pseudo ID'].isin(val_list)]['rs'].values
    MVI_train = df[df['Pseudo ID'].isin(train_list)]['Histology_Lymphovascular invasion'].values
    MVI_test = df[df['Pseudo ID'].isin(test_list)]['Histology_Lymphovascular invasion'].values
    MVI_val = df[df['Pseudo ID'].isin(val_list)]['Histology_Lymphovascular invasion'].values
    e = df[df['Pseudo ID'].isin(train_list)]['status'].values
    t = df[df['Pseudo ID'].isin(train_list)]['t'].values


    train = df[df['Pseudo ID'].isin(train_list)] # ,'VGHTPE'
    y_train = train.loc[:,['status','t']]
    y_train_d = train.loc[:,['death','death_t']]
    X_train= train.drop(columns = ['status','t','Hospital','1','2','3','4','5','death']) # ,'death_t'
    y_train.loc[:,['status']] = y_train.loc[:,['status']].astype(bool)
    aux = [(e1,e2) for e1,e2 in y_train.to_numpy()]
    y_train = np.array(aux, dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
    y_train_d .loc[:,['death']] = y_train_d .loc[:,['death']].astype(bool)
    aux_d  = [(e1,e2) for e1,e2 in y_train_d .to_numpy()]
    y_train_d  = np.array(aux_d , dtype=[('Status', '?'), ('Survival_in_days', '<f8')])

    test = df[df['Pseudo ID'].isin(test_list)] # ,'VGHTPE'
    y_test = test.loc[:,['status','t']]
    y_test_d = test.loc[:,['death','death_t']]
    X_test= test.drop(columns = ['status','t','Hospital','1','2','3','4','5','death']) # ,'death_t'
    y_test.loc[:,['status']] = y_test.loc[:,['status']].astype(bool)
    aux = [(e1,e2) for e1,e2 in y_test.to_numpy()]
    y_test = np.array(aux, dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
    y_test_d .loc[:,['death']] = y_test_d .loc[:,['death']].astype(bool)
    aux_d  = [(e1,e2) for e1,e2 in y_test_d .to_numpy()]
    y_test_d  = np.array(aux_d , dtype=[('Status', '?'), ('Survival_in_days', '<f8')])


    val = df[df['Pseudo ID'].isin(val_list)] # ,'VGHTPE'
    y_val = val.loc[:,['status','t']]
    y_val_d = val.loc[:,['death','death_t']]
    X_val= val.drop(columns = ['status','t','Hospital','1','2','3','4','5','death']) # ,'death_t'
    y_val.loc[:,['status']] = y_val.loc[:,['status']].astype(bool)
    aux = [(e1,e2) for e1,e2 in y_val.to_numpy()]
    y_val = np.array(aux, dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
    y_val_d .loc[:,['death']] = y_val_d .loc[:,['death']].astype(bool)
    aux_d  = [(e1,e2) for e1,e2 in y_val_d .to_numpy()]
    y_val_d  = np.array(aux_d , dtype=[('Status', '?'), ('Survival_in_days', '<f8')])

    _baseline_model = BreslowEstimator().fit(train_prediction,e,t)
    surv_funcs_train = _predict_function("get_survival_function", _baseline_model, train_prediction)
    surv_funcs_test = _predict_function("get_survival_function", _baseline_model, test_prediction)
    surv_funcs_val = _predict_function("get_survival_function", _baseline_model, val_prediction)
    strain = np.empty([train_prediction.shape[0],5], dtype=float)
    stest = np.empty([test_prediction.shape[0],5], dtype=float)
    sval = np.empty([val_prediction.shape[0],5], dtype=float)

    for i, (fn) in enumerate(surv_funcs_train):
    	strain[i] = fn([1,2,3,4,5])
    for i, (fn) in enumerate(surv_funcs_test):
        stest[i] = fn([1,2,3,4,5])
    for i, (fn) in enumerate(surv_funcs_val):
    	sval[i] = fn([1,2,3,4,5])


    dict_list = []
    # HK
    thd = []
    thd = [0.6595733017707814, 0.5157901534760587, 0.4573565885617464, 0.4173351495428791, 0.37742299143777254]

    dict_tabel = {}

    for k in [2,5]:
        dict_hk = {}
        MVI = [i for i in MVI_test]
        y_test11 = y_test.tolist()
        y_test11_d = y_test_d.tolist()
        y_test1 = [1 if i[0] and i[1]<k else 0 for i in y_test]
        y_test_t = []
        for i, (j) in enumerate(y_test):
            if j[0]==False and j[1]<k:
                y_test_t.append(i)

        y1  = stest[:,k-1].tolist()

        MVI_s = [i for i in MVI_test]
        y1_s = stest[:,k-1].tolist()

        for i in reversed(y_test_t):
            del MVI[i]
            del y_test1[i]
            del y1[i]
        y_train1 = [1 if i[0] and i[1]<k else 0 for i in y_train]
        y_train_t = []
        for i, (j) in enumerate(y_train):
            if j[0]==False and j[1]<k:
                y_train_t.append(i)
        yt1  = strain[:,k-1].tolist()
        for i in reversed(y_train_t):
            del y_train1[i]
            del yt1[i]



        fpr, tpr, thresholds = roc_curve(y_train1,yt1,pos_label=False)
        tfpr, ttpr, tthresholds = roc_curve(y_test1,y1,pos_label=False)

        train_auc = auc(fpr, tpr)
        tauc = auc(tfpr, ttpr)
        ei, e = max(enumerate([tpr[i]-fpr[i] for i in range(1,len(tpr))]), key=operator.itemgetter(1))
        
        # predictions = (y1 <= thresholds[ei+1]).astype(bool)
        # predictions_s = (y1 <= thresholds[ei+1]).astype(bool)
        # thd.append(thresholds[ei+1])
        predictions = (y1 <= np.asarray(thd[k-1])).astype(bool)
        predictions_s = (y1_s <= np.asarray(thd[k-1])).astype(bool)
        re = classification_report(y_test1, predictions)
        tn, fp, fn, tp = metrics.confusion_matrix(y_test1, predictions).ravel()
        print(tp,tn,fn,fp)
        ppv = tp / (tp + fp)
        npv = tn / (tn + fn)
        sen = tp / (tp + fn)
        spc = tn / (tn + fp)
        CI95,CI95_PPV,CI95_NPV,CI95_sen,CI95_spc = bootstrap_auc(y_test1,predictions,y1,fold_size=len(y_test1))
        dict_hk['prediction AUC'] = str(tauc)
        dict_hk['prediction AUC 95% CI'] = str(CI95)
        dict_hk['prediction PPV'] = str(ppv)
        dict_hk['prediction PPV 95% CI'] = str(CI95_PPV)
        dict_hk['prediction NPV'] = str(npv)
        dict_hk['prediction NPV 95% CI'] = str(CI95_NPV)
        dict_hk['prediction sensitivity'] = str(sen)
        dict_hk['prediction sensitivity 95% CI'] = str(CI95_sen)
        dict_hk['prediction specificity'] = str(spc)
        dict_hk['prediction specificity 95% CI'] = str(CI95_spc)
        # print('AUC 95% CI: ',str(CI95),'PPV 95% CI: ',str(CI95_PPV),'NPV 95% CI: ',str(CI95_NPV))


        mfpr, mtpr, mthresholds = roc_curve(y_test1,MVI)
        mvi_auc = auc(mfpr, mtpr)
        tnm, fpm, fnm, tpm = metrics.confusion_matrix(y_test1, MVI).ravel()
        ppvm = tpm / (tpm + fpm)
        npvm = tnm / (tnm + fnm)
        senm = tpm / (tpm + fnm)
        spcm = tnm / (tnm + fpm)
        CI95m,CI95_PPVm,CI95_NPVm,CI95_senm,CI95_spcm = bootstrap_auc_MVI(y_test1,MVI,fold_size=len(y1))
        dict_hk['MVI AUC'] = str(mvi_auc)
        dict_hk['MVI AUC 95% CI'] = str(CI95m)
        dict_hk['MVI PPV'] = str(ppvm)
        dict_hk['MVI PPV 95% CI'] = str(CI95_PPVm)
        dict_hk['MVI NPV'] = str(npvm)
        dict_hk['MVI NPV 95% CI'] = str(CI95_NPVm)
        dict_hk['MVI sensitivity'] = str(senm)
        dict_hk['MVI sensitivity 95% CI'] = str(CI95_senm)
        dict_hk['MVI specificity'] = str(spcm)
        dict_hk['MVI specificity 95% CI'] = str(CI95_spcm)

        z, p = DelongTest(np.array(y1),np.array(MVI),np.array(y_test1))._compute_z_p()
        dict_hk['Delongs test z score'] = str(z)
        dict_hk['Delongs test p value'] = str(p)

        plt.plot(tfpr, ttpr, label='High-risk of recurrence on model')
        plt.plot(mfpr, mtpr, label='Microvascular invasion')
        plt.plot([], [], ' ', label="p < 0.001")
        plt.plot([0, 1], [0, 1], linestyle='--') # 
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right", frameon=False,fontsize="9")
        plt.savefig('./KM/HKROC' + str(k)+ '.png')
        plt.clf()


        dt = []
        gt = []
        et = []
        ymvi_s,ymvi_t = [],[]
        ymvi1_s,ymvi1_t = [],[]
        ymvi0_s,ymvi0_t = [],[]
        ymvi1_s_d1,ymvi1_t_d1 = [],[]
        for i, (j) in enumerate(MVI_s):
            ymvi_s.append(y_test11[i][0])
            ymvi_t.append(y_test11[i][1])
            if j == 1:
                ymvi1_s.append(y_test11[i][0])
                ymvi1_t.append(y_test11[i][1])
                ymvi1_s_d1.append(y_test11_d[i][0])
                ymvi1_t_d1.append(y_test11_d[i][1])
                dt.append(y_test11[i][1])
                gt.append(1)
                if y_test11[i][0]:
                  et.append(1)
                else:
                  et.append(0)
            elif j == 0:
                ymvi0_s.append(y_test11[i][0])
                ymvi0_t.append(y_test11[i][1])

        yp_s,yp_t = [],[]
        yp1_s,yp1_t = [],[]
        yp0_s,yp0_t = [],[]
        yp1_s_d1,yp1_t_d1 = [],[]
        for i, (j) in enumerate(predictions_s):
            yp_s.append(y_test11[i][0])
            yp_t.append(y_test11[i][1])
            if j == True:
                yp1_s.append(y_test11[i][0])
                yp1_t.append(y_test11[i][1])
                yp1_s_d1.append(y_test11_d[i][0])
                yp1_t_d1.append(y_test11_d[i][1])
                dt.append(y_test11[i][1])
                gt.append(3)
                if y_test11[i][0]:
                  et.append(1)
                else:
                  et.append(0)
            elif j == False:
                yp0_s.append(y_test11[i][0])
                yp0_t.append(y_test11[i][1])
        df = pd.DataFrame({
        'durations': dt,
        'groups': gt, # could be strings too
        'events': et,
        })
        results = logrank_test(ymvi1_t, yp1_t, event_observed_A=ymvi1_s, event_observed_B=yp1_s)
        dict_hk['Logrank test pvalue'] = str(results.p_value)
        result = survival_difference_at_fixed_point_in_time_test_for_paired_data(k, ymvi1_t, ymvi1_s,yp1_t, yp1_s)
        dict_hk['survival_difference_at_fixed_point_in_time_test_for_paired_data() pvalue'] = str(result)

        timep0, survival_probp0 = kaplan_meier_estimator(yp0_s,yp0_t)
        dict_hk['prediction = 0'] = str(1-survival_probp0[np.array([np.where(timep0 < k)]).max()])
        
        timep1, survival_probp1 = kaplan_meier_estimator(yp1_s,yp1_t)
        plt.step([0] + timep1.tolist(), [0] + [1-i for i in survival_probp1], where="post",label='High-risk of recurrence on model')
        dict_hk['prediction = 1'] = str(1-survival_probp1[np.array([np.where(timep1 < k)]).max()])

        dict_tabel['hkpre'+str(k)] = [len([i for i in yp1_t if i > p]) for p in range(k+1)]+[float(dict_hk['prediction = 1'])]
        print([len([i for i in yp1_t if i > p]) for p in range(k+1)]+[float(dict_hk['prediction = 1'])])





        time0, survival_prob0 = kaplan_meier_estimator(ymvi0_s,ymvi0_t)
        dict_hk['MVI = 0'] = str(1-survival_prob0[np.array([np.where(time0 < k)]).max()])
        
        time1, survival_prob1 = kaplan_meier_estimator(ymvi1_s,ymvi1_t)
        plt.step([0] + time1.tolist(), [0] + [1-i for i in survival_prob1], where="post",label='Microvascular invasion')
        dict_hk['MVI = 1'] = str(1-survival_prob1[np.array([np.where(time1 < k)]).max()])
        print([len([i for i in ymvi1_t if i > p]) for p in range(k+1)]+[float(dict_hk['MVI = 1'])])
        dict_tabel['hkMVI'+str(k)] = [len([i for i in ymvi1_t if i > p]) for p in range(k+1)]+[float(dict_hk['MVI = 1'])]

        
        plt.plot([], [], ' ', label="p < 0.001")

        ax = plt.gca()
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        plt.ylabel("Cumulative probability")
        plt.xlabel("Time (years)")
        plt.legend(loc="upper left", frameon=False,fontsize="9")
        plt.axis([0, int(k), 0, 1.0])
        plt.xticks(np.arange(0, k+0.00001, 1))
        # plt.text(0.06, 0.81, "p < 0.001")
        # plt.text(4, 0.08, "p = 0.011")
        # plt.savefig('./KM/HK' + str(k) + '.png')
        plt.clf()
        dict_list.append(dict_hk)


    print(thd)
    # TW
    # for k in range(1,6):
    for k in [2,5]:
      dict_tw = {}
      MVI = [i for i in MVI_val]
      y_val11 = y_val.tolist()
      y_val11_d = y_val_d.tolist()
      y_val1 = [1 if i[0] and i[1]<k else 0 for i in y_val]
      y_val_t = []
      for i, (j) in enumerate(y_val):
          if j[0]==False and j[1]<k:
              y_val_t.append(i)
      y1  = sval[:,k-1].tolist()

      MVI_s = [i for i in MVI_val]
      y1_s = sval[:,k-1].tolist()
      for i in reversed(y_val_t):
          del MVI[i]
          del y_val1[i]
          del y1[i]
      y_train1 = [1 if i[0] and i[1]<k else 0 for i in y_train]
      y_train_t = []
      for i, (j) in enumerate(y_train):
          if j[0]==False and j[1]<k:
              y_train_t.append(i)
      yt1  = strain[:,k-1].tolist()
      for i in reversed(y_train_t):
          del y_train1[i]
          del yt1[i]

      fpr, tpr, thresholds = roc_curve(y_train1,yt1,pos_label=False)
      tfpr, ttpr, tthresholds = roc_curve(y_val1,y1,pos_label=False)
      
      train_auc = auc(fpr, tpr)
      tauc = auc(tfpr, ttpr)
      ei, e = max(enumerate([tpr[i]-fpr[i] for i in range(1,len(tpr))]), key=operator.itemgetter(1))
      predictions = (y1 <= np.asarray(thd[k-1])).astype(bool)
      predictions_s = (y1_s <= np.asarray(thd[k-1])).astype(bool)
      re = classification_report(y_val1, predictions)
      tn, fp, fn, tp = metrics.confusion_matrix(y_val1, predictions).ravel()
      print(tp,tn,fn,fp)
      ppv = tp / (tp + fp)
      npv = tn / (tn + fn)
      sen = tp / (tp + fn)
      spc = tn / (tn + fp)
      CI95,CI95_PPV,CI95_NPV,CI95_sen,CI95_spc = bootstrap_auc(y_val1,predictions,y1,fold_size=len(y1))
      dict_tw['prediction AUC'] = str(tauc)
      dict_tw['prediction AUC 95% CI'] = str(CI95)
      dict_tw['prediction PPV'] = str(ppv)
      dict_tw['prediction PPV 95% CI'] = str(CI95_PPV)
      dict_tw['prediction NPV'] = str(npv)
      dict_tw['prediction NPV 95% CI'] = str(CI95_NPV)
      dict_tw['prediction sensitivity'] = str(sen)
      dict_tw['prediction sensitivity 95% CI'] = str(CI95_sen)
      dict_tw['prediction specificity'] = str(spc)
      dict_tw['prediction specificity 95% CI'] = str(CI95_spc)
      dict_tw['Delongs test z score'] = str(z)
      dict_tw['Delongs test p value'] = str(p)

      mfpr, mtpr, mthresholds = roc_curve(y_val1,MVI)
      mvi_auc = auc(mfpr, mtpr)
      tnm, fpm, fnm, tpm = metrics.confusion_matrix(y_val1, MVI).ravel()
      ppvm = tpm / (tpm + fpm)
      npvm = tnm / (tnm + fnm)
      senm = tpm / (tpm + fnm)
      spcm = tnm / (tnm + fpm)
      CI95m,CI95_PPVm,CI95_NPVm,CI95_senm,CI95_spcm = bootstrap_auc_MVI(y_val1,MVI,fold_size=len(y1))
      dict_tw['MVI AUC'] = str(mvi_auc)
      dict_tw['MVI AUC 95% CI'] = str(CI95m)
      dict_tw['MVI PPV'] = str(ppvm)
      dict_tw['MVI PPV 95% CI'] = str(CI95_PPVm)
      dict_tw['MVI NPV'] = str(npvm)
      dict_tw['MVI NPV 95% CI'] = str(CI95_NPVm)
      dict_tw['MVI sensitivity'] = str(senm)
      dict_tw['MVI sensitivity 95% CI'] = str(CI95_senm)
      dict_tw['MVI specificity'] = str(spcm)
      dict_tw['MVI specificity 95% CI'] = str(CI95_spcm)

      plt.plot(tfpr, ttpr, label='High-risk of recurrence on model')
      plt.plot(mfpr, mtpr, label='Microvascular invasion')
      plt.plot([], [], ' ', label="p < 0.001")
      plt.plot([0, 1], [0, 1], linestyle='--')
      plt.xlabel('False Positive Rate')
      plt.ylabel('True Positive Rate')
      plt.legend(loc="lower right", frameon=False,fontsize="9")
      # plt.show()
      plt.savefig('./KM/TWROC' + str(k)+ '.png')
      plt.clf()

      dv = []
      gv = []
      ev = []
      ymvi1_s,ymvi1_t = [],[]
      ymvi0_s,ymvi0_t = [],[]
      ymvi1_s_d1,ymvi1_t_d1 = [],[]
      for i, (j) in enumerate(MVI_s):
          if j == 1:
              ymvi1_s.append(y_val11[i][0])
              ymvi1_t.append(y_val11[i][1])
              ymvi1_s_d1.append(y_val11_d[i][0])
              ymvi1_t_d1.append(y_val11_d[i][1])
              dv.append(y_val11[i][1])
              gv.append(1)
              if y_val11[i][0]:
                ev.append(1)
              else:
                ev.append(0)
          elif j == 0:
              ymvi0_s.append(y_val11[i][0])
              ymvi0_t.append(y_val11[i][1])

      yp1_s,yp1_t = [],[]
      yp0_s,yp0_t = [],[]
      yp1_s_d1,yp1_t_d1 = [],[]
      for i, (j) in enumerate(predictions_s):
          if j == True:
              yp1_s.append(y_val11[i][0])
              yp1_t.append(y_val11[i][1])
              yp1_s_d1.append(y_val11_d[i][0])
              yp1_t_d1.append(y_val11_d[i][1])
              dv.append(y_val11[i][1])
              gv.append(3)
              if y_val11[i][0]:
                ev.append(1)
              else:
                ev.append(0)
          elif j == False:
              yp0_s.append(y_val11[i][0])
              yp0_t.append(y_val11[i][1])

      df = pd.DataFrame({
        'durations': dv,
        'groups': gv, # could be strings too
        'events': ev,
        })
      results = logrank_test(ymvi1_t, yp1_t, event_observed_A=ymvi1_s, event_observed_B=yp1_s)
      # print(results.p_value)
      dict_tw['Logrank test pvalue'] = str(results.p_value)
      result = survival_difference_at_fixed_point_in_time_test_for_paired_data(k, ymvi1_t, ymvi1_s,yp1_t, yp1_s)
      dict_tw['survival_difference_at_fixed_point_in_time_test_for_paired_data() pvalue'] = str(result)
      predictions_s = [1 if i==True else 0 for i in predictions_s]

      timep0, survival_probp0 = kaplan_meier_estimator(yp0_s,yp0_t)
      dict_tw['prediction = 0'] = str(1-survival_probp0[np.array([np.where(timep0 < k)]).max()])
      timep1, survival_probp1 = kaplan_meier_estimator(yp1_s,yp1_t)
      plt.step([0] + timep1.tolist(), [0] + [1-i for i in survival_probp1], where="post",label='High-risk of recurrence on model')

      dict_tw['prediction = 1'] = str(1-survival_probp1[np.array([np.where(timep1 < k)]).max()])
      print([len([i for i in yp1_t if i > p]) for p in range(k+1)]+[float(dict_tw['prediction = 1'])])
      dict_tabel['twpre'+str(k)] = [len([i for i in yp1_t if i > p]) for p in range(k+1)]+[float(dict_tw['prediction = 1'])]

      time0, survival_prob0 = kaplan_meier_estimator(ymvi0_s,ymvi0_t)
      dict_tw['MVI = 0'] = str(1-survival_prob0[np.array([np.where(time0 < k)]).max()])

      time1, survival_prob1 = kaplan_meier_estimator(ymvi1_s,ymvi1_t)
      plt.step([0] + time1.tolist(), [0] + [1-i for i in survival_prob1], where="post",label='Microvascular invasion')
      dict_tw['MVI = 1'] = str(1-survival_prob1[np.array([np.where(time1 < k)]).max()])

      print([len([i for i in ymvi1_t if i > p]) for p in range(k+1)]+[float(dict_tw['MVI = 1'])])
      dict_tabel['twmvi'+str(k)] = [len([i for i in ymvi1_t if i > p]) for p in range(k+1)]+[float(dict_tw['MVI = 1'])]
      
      plt.plot([], [], ' ', label="p < 0.001")
      ax = plt.gca()
      ax.spines['right'].set_color('none')
      ax.spines['top'].set_color('none')
      plt.ylabel("Cumulative probability")
      plt.xlabel("Time (years)")
      plt.legend(loc="upper left", frameon=False,fontsize="9")
      plt.xticks(np.arange(0, k+0.00001, 1))
      # plt.text(0.8, 0.08, "p = 0.001")
      # plt.text(4, 0.08, "p = 0.000")
      plt.savefig('./KM/TW' + str(k)+ '.png')
      plt.clf()
      # plt.show()

      dict_list.append(dict_tw)
    re = pd.DataFrame(dict_list)
    print(re['prediction AUC 95% CI'].values.tolist())
    re.T.to_csv('./KM/result_CT.csv', index=True, header=['HK2','HK5','TW2','TW5'])
    cp = pd.DataFrame.from_dict(dict_tabel, orient='index') #Convert dict to df
    cp.to_csv("./KM/cp.csv",header=False)