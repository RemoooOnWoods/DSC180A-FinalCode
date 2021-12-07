import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import pearsonr

from numpy import loadtxt
from numpy import sort
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectFromModel

import xgboost as xgb
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer


def main(targets):

    if 'test' in targets:
        # load short version of data
        data = pd.read_csv('test/testdata/testdata.csv')
        identical_values = []
        for i in data.columns:
            if len(data[i].value_counts()) == 1:
                identical_values.append(i)
        data.drop(identical_values, axis=1, inplace = True)
        new_df = pd.get_dummies(data, columns=["cp", "restecg","ekgmo","ekgday","ekgyr","slope","thal","cmo","cday","cyr"],
            prefix=["cp", "restecg","ekgmo","ekgday","ekgyr","slope","thal","cmo","cday","cyr"])
        new_num = (data["num"] > 0).astype('int')
        new_df.drop(['num'], axis=1, inplace = True)
        new_df['num'] = new_num
        df_array = new_df.values
        X = df_array[:,0:140]
        Y = df_array[:,140]
        seed = 7
        test_size = 0.33
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

        # basic model
        model = xgb.XGBClassifier()
        model.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_test, y_test)], verbose = 0)
        y_pred = model.predict(X_test)
        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(y_test, predictions)
        print("The evaluation metric stats of the basic model:")
        print("Accuracy: %.2f%%" % (accuracy * 100.0))
        tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        print("Sensitivity: %.2f%%" % (sensitivity * 100.0))
        print("Specificity: %.2f%%" % (specificity * 100.0))
        f1 = f1_score(y_test, predictions)
        print("F1: %.2f%%" % (f1 * 100.0))
        auc = roc_auc_score(y_test, predictions)
        print("AUC: %.2f%%" % (auc * 100.0))
        print()

        # Bayesian Optimization
        bayes_cv_tuner = BayesSearchCV(
            estimator = xgb.XGBClassifier(
                objective = 'binary:logistic',
                eval_metric = 'auc',
                use_label_encoder = False
            ),
            search_spaces = {
                'learning_rate': Real(0.01, 1.0, 'log-uniform'),
                'min_child_weight': Integer(0, 5),
                'max_depth': Integer(0, 10),
                'subsample': Real(0.01, 1.0, 'uniform'),
                'colsample_bytree': Real(0.01, 1.0, 'uniform'),
                'reg_lambda': Real(0.01, 1000.0, 'log-uniform'),
                'reg_alpha': Real(0.01, 100.0, 'log-uniform'),
                'gamma': Real(0.001, 5.0, 'log-uniform'),
                'n_estimators': Integer(50, 1000),
            },
            scoring = 'roc_auc',
            n_jobs = 3,
            n_iter = 10,
            verbose = 0,
            random_state = 123
        )
        result = bayes_cv_tuner.fit(X_train, y_train)
        best_params = result.best_params_
        model_tuned = xgb.XGBClassifier(objective = 'binary:logistic', eval_metric = 'auc', use_label_encoder = False, **best_params)
        model_tuned.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_test, y_test)], verbose = 0)
        y_pred_new = model_tuned.predict(X_test)
        predictions_new = [round(value) for value in y_pred_new]
        accuracy_new = accuracy_score(y_test, predictions_new)
        print("The evaluation metric stats after Bayesian optimization")
        print("Accuracy: %.2f%%" % (accuracy_new * 100.0))
        tn_new, fp_new, fn_new, tp_new = confusion_matrix(y_test, predictions_new).ravel()
        sensitivity_new = tp_new / (tp_new + fn_new)
        specificity_new = tn_new / (tn_new + fp_new)
        print("Sensitivity: %.2f%%" % (sensitivity_new * 100.0))
        print("Specificity: %.2f%%" % (specificity_new * 100.0))
        f1_new = f1_score(y_test, predictions_new)
        print("F1: %.2f%%" % (f1_new * 100.0))
        auc_new = roc_auc_score(y_test, predictions_new)
        print("AUC: %.2f%%" % (auc_new * 100.0))
        print()

        # Feature selection
        selection = SelectFromModel(model_tuned, threshold=0.055, prefit=True)
        select_X_train = selection.transform(X_train)
        selection_model = xgb.XGBClassifier(objective = 'binary:logistic', eval_metric = 'auc', use_label_encoder = False, **best_params)
        selection_model.fit(select_X_train, y_train)
        select_X_test = selection.transform(X_test)
        predictions_select = selection_model.predict(select_X_test)
        accuracy_select = accuracy_score(y_test, predictions_select)
        print("accuracy after feature selection")
        print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (0.055, select_X_train.shape[1], accuracy_select*100.0))


if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
