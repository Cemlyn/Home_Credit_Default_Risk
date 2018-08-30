#  Benchmark - Logistic Regression Model

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV , KFold
from time import time

import lightgbm as lgbm

import matplotlib.pyplot as plt

# Clearing the command console - makes it easier to read
import os
clear = lambda: os.system('cls')
clear()

################################################################################################################################
# Fitting the actual model now
################################################################################################################################

corr_limit = 0.00
colinear_limit = 0.80

def feature_selection(extract=False):

        print("Selecting features...")

        #--- Removing features with absolute correlation lower than the limit ---#
        data = pd.read_pickle("Processed_DFS_Data_v03.pkl")
        #data = pd.read_csv("Processed_DFS_Data_sampled.csv")
        train, test = data[data['TARGET'].isin([0,1])], data[pd.isnull(data['TARGET'])]

        if extract==True:
                corr=train.corr().abs()
                corr.to_pickle("correlation_matrix.pkl")
        else:
                corr=pd.read_pickle("correlation_matrix.pkl")

        #corr = train.corr()
        corr = corr.sort_values(by=['TARGET'], ascending=False)
        corr = corr[corr['TARGET']>=corr_limit]

        #--- dictionary containing correlations
        correlations_dict = corr['TARGET'].to_dict()

        feature_list = list(corr.index.values)

        print("{} features have been selected out of {} using an absolute correlation threshold of {} or greater...".format( len(feature_list)-1, len(data.columns)-1, corr_limit))

        for feature in feature_list:
                colinear_features = list(corr[corr[feature]>=colinear_limit].index.values)
                colinear_features.remove(feature)

                for feat in colinear_features:

                        if corr.loc[feature,'TARGET']>corr.loc[feat,'TARGET']:

                                if feat in feature_list:
                                        feature_list.remove(feat)

        print("Colinear features have been removed, resulting in {} features remaining".format(len(feature_list)))

        return train[feature_list], test[feature_list]

#feature_selection(extract=False)

def build_model():

        train, test = feature_selection(extract=False)

        print("Fitting the model...")

        clf = lgbm.LGBMClassifier(
                n_estimators=5000,
                learning_rate=0.0128,
                max_depth=7,
                num_leaves=11,
                min_split_gain=0.0018,
                min_child_weight=2.6880,
                colsample_bytree=0.5672,
                subsample=0.6406,
                reg_alpha=3.5025,
                reg_lambda=0.9549,
                n_jobs=-1)

        # Training Data - convering to umpy arrays for ease
        x , y = train.drop(['TARGET'], axis=1), train['TARGET']
        x, y =x.values, y.values

        kf = KFold(n_splits=4)
        kf.get_n_splits(x)

        for train_index, val_index in kf.split(x):

                x_train, x_val = x[train_index], x[val_index]
                y_train, y_val = y[train_index], y[val_index]

                clf.fit(X=x_train,y=y_train)

                pred_train = clf.predict_proba(x_train)
                pred_val = clf.predict_proba(x_val)

                print( "Train ROC:%s, Val ROC:%s" %( roc_auc_score(y_train,pred_train[:,-1]) , roc_auc_score(y_val,pred_val[:,-1])  ) )

        # Exporting outputs for kaggle
        x_test = test.drop(['TARGET'], axis=1)
        x_index = list(x_test.index.values)
        pred_test = clf.predict_proba(x_test)
        pred_test = pd.DataFrame(data={'SK_ID_CURR':x_index,'TARGET': pred_test[:,-1]})
        pred_test.to_csv("GB_GS_results.csv", index=False)

        return None

build_model()


'''
boosting_type='gbdt',
learning_rate=0.066, 
reg_alpha=21.524, 
reg_lambda=24.922, 
max_depth=88, 
num_leaves=84,
min_child_samples=90, 
colsample_bytree=0.404)
'''