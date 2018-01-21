# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 13:31:41 2017

@author: rohit
"""
from collections import defaultdict
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics.classification import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
import operator

# Load data
filePath = r"G:\IUPUI\Fall 17\IS\project\dataset_phishing\dataset_phishing_with_attributes.csv"
data = pd.read_csv(filePath)
X = pd.read_csv(filePath, usecols=[*range(0, 9)])
Y = pd.read_csv(filePath, usecols=[*range(9, 10)])
output_label = ["Phishy", "Suspicious", "Legitimate"]

# 10 fold CV
kf = KFold(n_splits=10, shuffle=True, random_state=5)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
X = X.values
Y = Y.values

i = 1
for train, test in kf.split(X):
    print("Current Iteration", i)
    i = i + 1

    X_train, X_test = X[train], X[test]
    y_train, y_test = Y[train], Y[test]

    # Normalizing the Sets
    scaler = preprocessing.StandardScaler().fit(X_train)
    scaler.transform(X_train)
    scaler.transform(X_test)

    # PCA
    # pca = PCA(n_components=None)
    # pca.fit_transform(X_train)
    # pca.transform(X_test)

    # Naive Bayes
    naiveBayesModel = GaussianNB()
    naiveBayesModel.fit(X_train, y_train)
    y_pred_naiveBayesModel = naiveBayesModel.predict(X_test)
    print('Naive Bayes Classification')
    print('Accuracy:', accuracy_score(y_test, y_pred_naiveBayesModel))
    print('F1 score:', f1_score(y_test, y_pred_naiveBayesModel, average='macro'))
    print(classification_report(y_test, y_pred_naiveBayesModel,target_names=output_label, digits=4))
    print(confusion_matrix(y_true=y_test, y_pred=y_pred_naiveBayesModel))

    # Decision Tree
    decisionTreeModel = tree.DecisionTreeClassifier()
    decisionTreeModel.fit(X_train, y_train)
    y_pred_decision_tree = decisionTreeModel.predict(X_test)
    prob_DT = decisionTreeModel.predict_proba(X_test)
    print('Decision Tree Classification')
    print('Accuracy:', accuracy_score(y_test, y_pred_decision_tree))
    print('F1 score:', f1_score(y_test, y_pred_decision_tree, average='macro'))
    print(classification_report(y_test, y_pred_decision_tree,target_names=output_label, digits=4))
    print(confusion_matrix(y_true=y_test, y_pred=y_pred_decision_tree))

    # Random Forest
    RfModel = RandomForestClassifier(n_estimators=20)
    RfModel.fit(X_train, y_train)
    y_pred_Rf = RfModel.predict(X_test)
    prob_RF = RfModel.predict_proba(X_test)
    print('Random Forest Classification')
    print('Accuracy:', accuracy_score(y_test, y_pred_Rf))
    print('F1 score:', f1_score(y_test, y_pred_Rf, average='macro'))
    print(classification_report(y_test, y_pred_Rf,target_names=output_label, digits=4))
    print(confusion_matrix(y_true=y_test, y_pred=y_pred_Rf))

    # Support Vector Machine
    svmModel = svm.SVC(decision_function_shape='ovr', C=100,probability=True)
    svmModel.fit(X_train, y_train)
    y_pred_svm = svmModel.predict(X_test)
    prob_SVM = svmModel.predict_proba(X_test)
    print('Support Vector Machine Classification')
    print('Accuracy:', accuracy_score(y_test, y_pred_svm))
    print('F1 score:', f1_score(y_test, y_pred_svm, average='macro'))
    print(classification_report(y_test, y_pred_svm,target_names=output_label, digits=4))
    print(confusion_matrix(y_true= y_test,y_pred= y_pred_svm))

    # Neural Network
    NNmodel = MLPClassifier(hidden_layer_sizes=(7, 10), solver='lbfgs', alpha=1e-5, random_state=5)
    NNmodel.fit(X_train, y_train)
    y_pred_nn = NNmodel.predict(X_test)
    print('Neural Network Classification')
    print('Accuracy:', accuracy_score(y_test, y_pred_nn))
    print('F1 score:', f1_score(y_test, y_pred_nn, average='macro'))
    print(classification_report(y_test, y_pred_nn,target_names=output_label, digits=4))
    print(confusion_matrix(y_true=y_test, y_pred=y_pred_nn))

    print('Combination of Random Forest and Decision Tree')
    y_pred_RF_with_DT = []
    for idx,val in enumerate(prob_RF):
        max_index_RF,max_value_RF = max(enumerate(prob_RF[idx]), key=operator.itemgetter(1))
        max_index_DT,max_value_DT = max(enumerate(prob_DT[idx]), key=operator.itemgetter(1))

        if(y_test[idx] == y_pred_decision_tree[idx]):
            y_pred_RF_with_DT.append(y_pred_decision_tree[idx])
        else:
            y_pred_RF_with_DT.append(y_pred_Rf[idx])

    print('Accuracy:', accuracy_score(y_test, y_pred_RF_with_DT))
    print('F1 score:', f1_score(y_test, y_pred_RF_with_DT, average='macro'))
    print(classification_report(y_test, y_pred_RF_with_DT,target_names=output_label, digits=4))
    print(confusion_matrix(y_true=y_test, y_pred=y_pred_RF_with_DT))

    print('Combination of SVM and Random Forest')
    y_pred_RF_with_SVM = []
    for idx,val in enumerate(prob_RF):
        max_index_RF,max_value_RF = max(enumerate(prob_RF[idx]), key=operator.itemgetter(1))
        max_index_DT,max_value_SVM = max(enumerate(prob_SVM[idx]), key=operator.itemgetter(1))

        if(y_test[idx] == y_pred_svm[idx]):
            y_pred_RF_with_SVM.append(y_pred_svm[idx])
        else:
            y_pred_RF_with_SVM.append(y_pred_Rf[idx])

    print('Accuracy:', accuracy_score(y_test, y_pred_RF_with_SVM))
    print('F1 score:', f1_score(y_test, y_pred_RF_with_SVM, average='macro'))
    print(classification_report(y_test, y_pred_RF_with_SVM,target_names=output_label, digits=4))
    print(confusion_matrix(y_true=y_test, y_pred=y_pred_RF_with_SVM))
