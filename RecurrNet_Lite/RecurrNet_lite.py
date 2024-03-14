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

# # -----------------------------------------------------------------------
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
best = 0.0

best_path = './best/'
for k in range(100):

  with open('./sl_A.pkl', 'rb') as f:
    sl_dict = pickle.load(f)
  print(k,k,k,k,k)
  all_id = sl_dict['all']
  train_id = sl_dict['train']
  test_id = sl_dict['test']
  val_id = sl_dict['val']
  # load data
  df = pd.read_csv('rn_lite_data.csv',header=0)
  # df = pd.get_dummies(data=df, columns=['Sex']) # 

  all_ = df[df['Pseudo ID'].isin(all_id)] # ,'VGHTPE'
  ny = all_.loc[:,['status','t']]
  X_train1= all_.drop(columns = ['status','t','Hospital','1','2','3','4','5','death','death_t'])
  ny.loc[:,['status']] = ny.loc[:,['status']].astype(bool)
  aux = [(e1,e2) for e1,e2 in ny.to_numpy()]
  ny = np.array(aux, dtype=[('Status', '?'), ('Survival_in_days', '<f8')])

  train = df[df['Pseudo ID'].isin(train_id)] # ,'VGHTPE'
  y_train = train.loc[:,['status','t']]
  X_train= train.drop(columns = ['status','t','Hospital','1','2','3','4','5','death','death_t'])
  y_train.loc[:,['status']] = y_train.loc[:,['status']].astype(bool)
  aux = [(e1,e2) for e1,e2 in y_train.to_numpy()]
  y_train = np.array(aux, dtype=[('Status', '?'), ('Survival_in_days', '<f8')])

  test = df[df['Pseudo ID'].isin(test_id)] # ,'VGHTPE'
  y_test = test.loc[:,['status','t']]
  X_test= test.drop(columns = ['status','t','Hospital','1','2','3','4','5','death','death_t']) 
  y_test.loc[:,['status']] = y_test.loc[:,['status']].astype(bool)
  aux = [(e1,e2) for e1,e2 in y_test.to_numpy()]
  y_test = np.array(aux, dtype=[('Status', '?'), ('Survival_in_days', '<f8')])


  val = df[df['Pseudo ID'].isin(val_id)] # ,'VGHTPE'
  y_val = val.loc[:,['status','t']]
  X_val= val.drop(columns = ['status','t','Hospital','1','2','3','4','5','death','death_t'])
  y_val.loc[:,['status']] = y_val.loc[:,['status']].astype(bool)
  aux = [(e1,e2) for e1,e2 in y_val.to_numpy()]
  y_val = np.array(aux, dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
  X_train1 = X_train1.drop(columns = ['Pseudo ID','Histology_Lymphovascular invasion'])
  X_train = X_train.drop(columns = ['Pseudo ID','Histology_Lymphovascular invasion'])
  X_test = X_test.drop(columns = ['Pseudo ID','Histology_Lymphovascular invasion'])
  X_val = X_val.drop(columns = ['Pseudo ID','Histology_Lymphovascular invasion'])

  names = list(X_val.columns.values)


  for t in range(20):
    # imp
    imp = IterativeImputer(DecisionTreeRegressor(), missing_values=np.nan,max_iter=100)#
    # X = imp.fit_transform(X_train1)
    X_train = imp.fit_transform(X_train)
    # with open('imp.pkl', 'rb') as f:
    # with open('./model/9133/imp.pkl', 'rb') as f:
    #   imp = pickle.load(f)
    X = imp.transform(X_train1)
    X = pd.DataFrame(X, columns = names)
    X_train = imp.transform(X_train)
    X_train = pd.DataFrame(X_train,columns = names)
    X_test = imp.transform(X_test)
    X_test = pd.DataFrame(X_test,columns = names)
    X_val = imp.transform(X_val)
    X_val = pd.DataFrame(X_val,columns = names)

    X_trainn = X_train
    X_testt = X_test
    X_vall = X_val
    Xx = X

    model = sksurv.ensemble.RandomSurvivalForest(n_estimators=25, max_depth=20, 
      min_samples_split=6, min_samples_leaf=3, min_weight_fraction_leaf=0.0, 
      max_features='sqrt', max_leaf_nodes=None, bootstrap=True, oob_score=False, 
      n_jobs=None, random_state=None, verbose=0, warm_start=False, max_samples=None)

    model.fit(X_trainn, y_train)

    va_times = np.arange(1, 6)
    cph_risk_scores = model.predict(X_testt)
    cph_auc, cph_mean_auc = cumulative_dynamic_auc(
        y_train, y_test, cph_risk_scores, va_times
    )
    cindex_model = model.score(X_testt, y_test)
    # val
    cph_auc_val, cph_mean_auc_val = cumulative_dynamic_auc(
        y_train, y_val, model.predict(X_vall), va_times
    )
    cindex_model_val = model.score(X_vall, y_val)
    # print('train:', round(model.score(X_trainn, y_train), 4), 'test:', round(cindex_model, 4), cph_auc)
    # print('val:', round(model.score(X_vall, y_val), 4), cumulative_dynamic_auc(y_train, y_val, model.predict(X_vall), va_times)[0])
        
    if cph_auc_val[0] > best:
        best = cph_auc_val[0]
        print('best train:', round(model.score(X_trainn, y_train), 4), 'test:', round(cindex_model, 4), cph_auc)
        print('best val:', round(model.score(X_vall, y_val), 4), cumulative_dynamic_auc(y_train, y_val, model.predict(X_vall), va_times)[0])
        with open(best_path + 'model.pkl','wb') as f:
            pickle.dump(model,f)
        with open(best_path + 'imp.pkl','wb') as f:
            pickle.dump(imp,f)
        with open(best_path + 'train.pkl', 'wb') as f:
            pickle.dump(X_train, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(y_train, f, pickle.HIGHEST_PROTOCOL)
        with open(best_path + 'test.pkl', 'wb') as f:
            pickle.dump(X_test, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(y_test, f, pickle.HIGHEST_PROTOCOL)
        with open(best_path + 'val.pkl', 'wb') as f:
            pickle.dump(X_val, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(y_val, f, pickle.HIGHEST_PROTOCOL)
    # else:
    #     print('train:', round(model.score(X_trainn, y_train), 4), 'test:', round(cindex_model, 4), cph_auc)
    #     print('val:', round(model.score(X_vall, y_val), 4), cumulative_dynamic_auc(y_train, y_val, model.predict(X_vall), va_times)[0])