#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from os import listdir
import os
import csv
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import xgboost
import xgboost as xgb
from hyperopt import fmin, tpe, hp, rand, anneal, partial, Trials
import matplotlib.pyplot as plt
from joblib import dump,load


# In[2]:


def function(argsDict):
    colsample_bytree = argsDict["colsample_bytree"]
    max_depth = argsDict["max_depth"]
    n_estimators = argsDict['n_estimators']
    learning_rate = argsDict["learning_rate"]
    subsample = argsDict["subsample"]
    min_child_weight = argsDict["min_child_weight"]
    gamma = argsDict["gamma"]

    clf = xgb.XGBRegressor(nthread=4,    #进程数
                            colsample_bytree=colsample_bytree,
                            max_depth=int(max_depth),  #最大深度
                            n_estimators=int(n_estimators),   #树的数量
                            learning_rate=learning_rate, #学习率
                            subsample=subsample,      #采样数
                            min_child_weight=min_child_weight,   #孩子数
                            gamma=gamma,
                            random_state=int(42),
                            objective="reg:squarederror"
                            )
    clf.fit(trainX, trainY)
    prediction = clf.predict(testX)
    R2= r2_score(testY, prediction)
    #RMSE = mean_squared_error(testY, prediction)**(1/2)
    return -R2

def model_training_with_kfold_cv(X,Y,k,ite,rs):
    def function(argsDict):
        colsample_bytree = argsDict["colsample_bytree"]
        max_depth = argsDict["max_depth"]
        n_estimators = argsDict['n_estimators']
        learning_rate = argsDict["learning_rate"]
        subsample = argsDict["subsample"]
        min_child_weight = argsDict["min_child_weight"]
        gamma = argsDict["gamma"]
        model = xgb.XGBRegressor(nthread=4,    
                                colsample_bytree=colsample_bytree,
                                max_depth=int(max_depth),  
                                n_estimators=int(n_estimators),   
                                learning_rate=learning_rate, 
                                subsample=subsample,      
                                min_child_weight=min_child_weight,  
                                gamma=gamma,
                                random_state=int(42),
                                objective="reg:squarederror"
                                )
        model.fit(trainX, trainY)
        prediction = model.predict(testX)
        R2 = r2_score(testY, prediction)
        #RMSE = mean_squared_error(testY, prediction)**(1/2)
        return -R2
    best_models = []
    feature_importance = []
    overfitting = []
    testings = []
    predictions = []
    index = []
    kf = KFold(n_splits=k,shuffle=True,random_state=rs)#k-fold cross validation
    count = 0
    for train_index,test_index in kf.split(X,Y):
        count += 1
        trainX,testX,trainY,testY = [],[],[],[]
        for i in range(len(Y)):
            if i in train_index:
                trainX.append(X[i])
                trainY.append(Y[i])
            else:
                testX.append(X[i])
                testY.append(Y[i])
        print(len(trainX),len(testX))
        print(count)
        trainX = np.array(trainX)
        testX = np.array(testX)
        trials = Trials()
        best = fmin(function, parameter_space_gbr, algo=tpe.suggest, max_evals=ite, trials=trials)
        parameters = ['colsample_bytree', 'max_depth', 'n_estimators', 'learning_rate', 'gamma', 'min_child_weight']
        colsample_bytree = best["colsample_bytree"]
        max_depth = best["max_depth"]
        n_estimators = best['n_estimators']
        learning_rate = best["learning_rate"]
        subsample = best["subsample"]
        min_child_weight = best["min_child_weight"]
        gamma = best["gamma"]
        print("The_best_parameter：", best)
        best_models.append(best)
        gbr = xgb.XGBRegressor(nthread=4,   
                                    colsample_bytree=colsample_bytree,
                                    max_depth=int(max_depth),  
                                    n_estimators=int(n_estimators),   
                                    learning_rate=learning_rate, 
                                    subsample=subsample,      
                                    min_child_weight=min_child_weight,   
                                    gamma=gamma,
                                    random_state=int(42),
                                    objective="reg:squarederror"
                                    )
        gbr.fit(trainX, trainY)
        feature_importance.append(gbr.feature_importances_)
        predictY_test = gbr.predict(testX)
        predictY_train = gbr.predict(trainX)
        testings.extend(testY)
        predictions.extend(predictY_test)
        index.extend(test_index)
        print('CV_RMSE:',mean_squared_error(testings, predictions)**(1/2))
        print('CV_r2:',r2_score(testings, predictions))
        print('RMSE_testing:',mean_squared_error(testY, predictY_test)**(1/2))
        print('r2_testing:',r2_score(testY, predictY_test))
        print('RMSE_training:',mean_squared_error(trainY, predictY_train)**(1/2))
        print('r2_training:',r2_score(trainY, predictY_train))
        overfitting.append(mean_squared_error(trainY, predictY_train)**(1/2)-mean_squared_error(testY, predictY_test)**(1/2))
    index_sorted,testings_sorted,predictions_sorted = (list(t) for t in zip(*sorted(zip(index,testings,predictions))))
    return(best_models,testings_sorted,predictions_sorted,feature_importance,np.mean(overfitting))

parameter_space_gbr = {"colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
                           "max_depth": hp.quniform("max_depth", 1, 10, 1),
                           "n_estimators": hp.quniform("n_estimators", 10, 200, 1),
                           "learning_rate": hp.uniform("learning_rate", 0.01, 0.5),
                           "subsample": hp.uniform("subsample", 0.9, 1),
                           "min_child_weight": hp.uniform("min_child_weight", 0.5, 10),
                           "gamma": hp.uniform("gamma", 0.01, 0.5)
                           }

#evaluate model accuracy
def view_accuracy(testings, predictions):
    print("RMSE:",mean_squared_error(testings, predictions)**(1/2))
    print("R2:",r2_score(testings, predictions))
    plt.figure(figsize=(6,6))
    plt.ylabel('Predicted',fontsize=25)
    plt.xlabel('True',fontsize=25)
    x1 = np.linspace(min([min(testings),min(predictions)])-1,max([max(testings),max(predictions)])+1,500)#从(-1,1)均匀取50个点
    y1 = x1
    plt.plot(x1,y1,color = 'black',linewidth = 1,dashes=[6, 2])
    plt.scatter(testings, predictions, c = 'salmon',alpha=0.8,label='10-fold_cross_validation')
    plt.legend(fontsize=15)
    plt.show()


# In[3]:


X = np.load('X_example.npy')
Y = np.load('Y_example.npy')


# In[4]:


best_models,testings,predictions,feature_importance,overfitting = model_training_with_kfold_cv(X,Y,20,10,42)#model training with 20-fold cross validation,increase iterations in applications
view_accuracy(testings, predictions)


# In[22]:


import shap
def get_explainer(hyperparameter_dict,X,Y):#get explainer for a set of hyperparameters
    model = xgb.XGBRegressor(nthread=4,    
                                colsample_bytree=hyperparameter_dict['colsample_bytree'],
                                max_depth=int(hyperparameter_dict['max_depth']),  
                                n_estimators=int(hyperparameter_dict['n_estimators']),   
                                learning_rate=hyperparameter_dict['learning_rate'], 
                                subsample=hyperparameter_dict['subsample'],      
                                min_child_weight=hyperparameter_dict['min_child_weight'],   
                                gamma=hyperparameter_dict['gamma'],
                                random_state=int(42),
                                objective="reg:squarederror"
                                )
    model.fit(X,Y)
    explainer = shap.TreeExplainer(model)
    return(model,explainer)
hyperparameter_dict = dict(best_models[0])
hyperparameter_list = list(best_models[0])
for hyperparameter in hyperparameter_list:
    hyperparameter_dict[hyperparameter] = 0
for i in range(len(best_models)):
    for hyperparameter in hyperparameter_list:
        hyperparameter_dict[hyperparameter] += best_models[i][hyperparameter]/len(best_models)
model,explainer = get_explainer(hyperparameter_dict,X,Y) # 引用package并且获得解释器explainer
dump(model,'model/miuE.joblib')
shap_values = explainer.shap_values(np.array(X)) # 获取训练集data各个样本各个特征的SHAP值
y_base = explainer.expected_value


# In[23]:


import pandas as pd
i = 1
player_explainer = pd.DataFrame()
temp = []
for i in range(15):
    temp.append(str(i+1))
player_explainer['feature'] = temp+['Delta_EN','Dipole','s_electron','p_electron','d_electron','SG','Rnum','Mnum','Thickness','Atom_layers','deltaH','HM']
player_explainer['feature_value'] = np.array(X[i])
player_explainer['shap_value'] = shap_values[i]
player_explainer


# In[24]:


#shap_values = explainer.shap_values(data[cols])# only output shap values for specific features, pandas version higher than 2.0 may have 'Int64Index' problem 
shap.summary_plot(shap_values, data[cols],plot_type='bar',show = True)
shap.summary_plot(shap_values, data[cols],plot_type='violin',show = True)
shap.summary_plot(shap_values, data[cols],show = True)
#plt.savefig('/mnt/c/Users/azere/Desktop/SHAP.png',dpi = 600,bbox_inches='tight') # modify the sharp source code to save high-res figure


# In[25]:


#Partial dependence 
shap.dependence_plot('HM',shap_values,data[cols], interaction_index=None, show=False)


# In[ ]:




