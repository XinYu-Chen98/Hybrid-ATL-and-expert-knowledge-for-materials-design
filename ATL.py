#!/usr/bin/env python
# coding: utf-8

# In[3]:


#enable all magic commands
get_ipython().run_line_magic('lsmagic', '')
#import torch
import torch
import torch.nn as nn
from torch.optim import SGD,Adam
import torch.utils.data as Data
#import sklearn
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from  sklearn.preprocessing import StandardScaler
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression,Ridge,Lasso
import xgboost
import xgboost as xgb
from hyperopt import fmin, tpe, hp, rand, anneal, partial, Trials
#import basics
from os import listdir   
import os 
import pandas as pd
import csv
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm


# In[4]:


#Generat MLP classifier
def remove_outliers(data,formula,target_property,limit,limit_pos):
    tempformula = []
    tempmiue = []
    datanew = pd.DataFrame(data=None,columns=list(data))
    data_array = np.array(data)
    data_list =data_array.tolist()
    print('Processing Outliers:')
    for i in tqdm(range(len(data))):
        #remove extreme values and repeated samples
        try:
            if data[formula][i] not in tempformula and float(data[target_property][i]) > limit[0] and float(data[target_property][i]) < limit[1]:
                tempformula.append(data[formula][i])
                #Take the median for samples with diverse structures
                data_temp = data.loc[data[formula].isin([data[formula][i]])].sort_values(by = target_property)
                data_array = np.array(data_temp)
                data_list = data_array.tolist()
                if limit_pos == 'MID':
                    datanew.loc[i] = data_list[int(len(data_list)/2)]         
                if limit_pos == 'TOP':
                    datanew.loc[i] = data_list[0]  
                if limit_pos == 'BOT':
                    datanew.loc[i] = data_list[-1]  
        except Exception as e:
            print(e)
    return(datanew)

#Unify data / oversample target data / define data loader
def log_data(data):
    logdata = []
    for i in range(len(data)):
        logdata.append(np.log10(data[i]))
    return(logdata)

def view_property_distribution(target_property_s,source_data,target_property_t,target_data):
    plt.subplot(2,1,1) 
    plt.hist(source_data[target_property_s].values.tolist(),bins=20,facecolor='royalblue')
    plt.title('Source - '+target_property_s)
    plt.subplot(2,1,2) 
    plt.hist(target_data[target_property_t].values.tolist(),bins=20,facecolor='firebrick')
    plt.title('Target - '+target_property_t, y = -0.33)
    plt.show()
    
#Generat MLP classifier
class MLPmodel_C(nn.Module):
    def __init__(self,innum,a,b,c,d,e,f,g,h,i,j,tp):
        super(MLPmodel_C,self).__init__()
        self.hidden1=nn.Sequential(
            nn.Linear(innum,a),
            nn.BatchNorm1d(num_features=a),
            nn.ReLU(),  
        )
        self.hidden2=nn.Sequential(
            nn.Linear(a,b),
            nn.BatchNorm1d(num_features=b),
            nn.ReLU(),
        )
        self.hidden3=nn.Sequential(
            nn.Linear(b,c),
            nn.BatchNorm1d(num_features=c),
            nn.ReLU(),
        )
        self.hidden4=nn.Sequential(
            nn.Linear(c,d),
            nn.BatchNorm1d(num_features=d),
            nn.ReLU(),
        )
        self.hidden5=nn.Sequential(
            nn.Linear(d,e),
            nn.BatchNorm1d(num_features=e),
            nn.ReLU(),
        )
        self.hidden6=nn.Sequential(
            nn.Linear(e,f),
            nn.BatchNorm1d(num_features=f),
            nn.ReLU(),
        )
        self.hidden7=nn.Sequential(
            nn.Linear(f,g),
            nn.BatchNorm1d(num_features=g),
            nn.ReLU(),
        )
        self.hidden8=nn.Sequential(
            nn.Linear(g,h),
            nn.BatchNorm1d(num_features=h),
            nn.ReLU(),
        )
        self.hidden9=nn.Sequential(
            nn.Linear(h,i),
            nn.BatchNorm1d(num_features=i),
            nn.ReLU(),
        )
        self.hidden10=nn.Sequential(
            nn.Linear(i,j),
            nn.BatchNorm1d(num_features=j),
            nn.ReLU(),
        )
        self.classifier=nn.Sequential(
            nn.Linear(j,1),
            nn.Sigmoid(),
        )
    def forward(self,x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = self.hidden5(x)
        x = self.hidden6(x)
        x = self.hidden7(x)
        x = self.hidden8(x)
        x = self.hidden9(x)
        x = self.hidden10(x)
        output = self.classifier(x)
        return output


class MLPmodel(nn.Module):
    def __init__(self,a,b,c,d,e,f,g,h,i,j,tp):
        super(MLPmodel,self).__init__()
        self.hidden1=nn.Sequential(
            nn.Linear(268,a),
            nn.BatchNorm1d(num_features=a),
            nn.ReLU(), 
        )
        self.hidden2=nn.Sequential(
            nn.Linear(a,b),
            nn.BatchNorm1d(num_features=b),
            nn.ReLU(),
        )
        self.hidden3=nn.Sequential(
            nn.Linear(b,c),
            nn.BatchNorm1d(num_features=c),
            nn.ReLU(),
        )
        self.hidden4=nn.Sequential(
            nn.Linear(c,d),
            nn.BatchNorm1d(num_features=d),
            nn.ReLU(),
        )
        self.hidden5=nn.Sequential(
            nn.Linear(d,e),
            nn.BatchNorm1d(num_features=e),
            nn.ReLU(),
        )
        self.hidden6=nn.Sequential(
            nn.Linear(e,f),
            nn.BatchNorm1d(num_features=f),
            nn.ReLU(),
        )
        self.hidden7=nn.Sequential(
            nn.Linear(f,g),
            nn.BatchNorm1d(num_features=g),
            nn.ReLU(),
        )
        self.hidden8=nn.Sequential(
            nn.Linear(g,h),
            nn.BatchNorm1d(num_features=h),
            nn.ReLU(),
        )
        self.hidden9=nn.Sequential(
            nn.Linear(h,i),
            nn.BatchNorm1d(num_features=i),
            nn.ReLU(),
        )
        self.hidden10=nn.Sequential(
            nn.Linear(i,j),
            nn.BatchNorm1d(num_features=j),
            nn.ReLU(),
        )
        if tp == 'r':
            self.regression=nn.Sequential(
                nn.Linear(j,1),
                nn.ReLU(),
            )
        if tp == 'c':
            self.regression=nn.Sequential(
                nn.Linear(j,1),
                nn.Sigmoid(),
            )
    def forward(self,x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = self.hidden5(x)
        x = self.hidden6(x)
        x = self.hidden7(x)
        x = self.hidden8(x)
        x = self.hidden9(x)
        x = self.hidden10(x)
        output = self.regression(x)
        return output
    
#Load and unify train data
def cbfv_tolist(data):
    cbfvlist = []
    for j in range(len(data)):
        temp = np.array(data[j].strip('[').strip(']').split(','))
        fv = []
        for i in range(len(temp)):
            try:
                fv.append(float(temp[i]))
            except:
                if temp[i] == ' False':
                    fv.append(0)
                elif temp[i]:
                    fv.append(1)
        cbfvlist.append(fv)
    return(cbfvlist)

def data_unifier(data):
    ss = StandardScaler(with_mean = True,with_std = True)
    new_data = ss.fit_transform(data)
    return(new_data)

def oversample(dataX,dataY,target_size):
    print('Oversampling, target_size: ',target_size)
    dataXnew = []
    dataYnew = []
    for i in range(len(dataX)):
        dataXnew.append(dataX[i])
        dataYnew.append(dataY[i])
    for i in tqdm(range(target_size-len(dataX))):
        rand = random.randint(0,len(dataX)-1)
        dataXnew.append(dataX[rand])
        dataYnew.append(dataY[rand])
    return(dataXnew,dataYnew)

def batch_loader(dataX,dataY,batchsize):
    minvalue = min(dataY)
    print(minvalue)
    for i in range(len(dataY)):
        dataY[i] -= minvalue
    trainX, testX, trainY, testY = train_test_split(dataX, dataY, train_size = 0.85, test_size = 0.15, random_state = 42)
    trainX_tensor = torch.from_numpy(np.array(trainX).astype(np.float32))
    trainY_tensor = torch.from_numpy(np.array(trainY).astype(np.float32))
    train_data = Data.TensorDataset(trainX_tensor,trainY_tensor)
    train_loader = Data.DataLoader(
        dataset = train_data,
        batch_size = batchsize,
        shuffle = True,
        num_workers = 0,
    )
    for step,(b_x,b_y) in enumerate(train_loader):
        if step>0:
            break
    print("Check:")
    print("batch_x.shape:",b_x.shape)
    print("batch_y.shape:",b_y.shape)
    print("batch_x.dtype:",b_x.dtype)
    print("batch_y.dtype:",b_y.dtype)
    return(train_loader, trainX, testX, trainY, testY)

#Train MLP
def train_mlp(mlp,train_loader,iteration,learning_rate,modeltype,viewloss):
    optimizer=Adam(mlp.parameters(),
                    lr=learning_rate,
                    betas=(0.9, 0.999),
                    eps=1e-08,
                    weight_decay=0,
                    amsgrad=False)
    if modeltype == 'r':
        loss_func=nn.MSELoss()
    if modeltype == 'c':
        loss_func=nn.BCELoss(reduction='mean')
    train_loss_all=[]
    for epoch in range(iteration):
        for step,(b_x,b_y) in enumerate(train_loader):
            output=mlp(b_x).flatten()
            train_loss=loss_func(output,b_y)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        train_loss_all.append(train_loss.item())
        print(epoch,train_loss.item())
        if train_loss.item() < 0.0001:
            break
    if viewloss == 1:
        plt.figure()
        plt.plot(train_loss_all,"b-")
        plt.title("train loss per Ite")
        plt.show()
    if train_loss_all[0] / train_loss_all[-1]+0.000001 < 1.3:
        state = 0
    else:
        state = 1
    return(state)

#Train joint MLP

def joint_train_mlp(mlp,train_loader_s,mlp_c,iteration):
    optimizer=Adam(mlp.parameters(),
                    lr=0.00005,
                    betas=(0.9, 0.999),
                    eps=1e-08,
                    weight_decay=0,
                    amsgrad=False)
    loss_func=nn.MSELoss()
    loss_funcc=nn.BCELoss(reduction='mean')
    train_loss_all=[]
    train_loss_all_2=[]
    for epoch in range(iteration):
        for step,(b_x,b_y) in enumerate(train_loader_s):
            output=mlp(b_x).flatten()
            outputc=mlp_c(mlp.hidden10(mlp.hidden9(mlp.hidden8(mlp.hidden7(mlp.hidden6(mlp.hidden5(mlp.hidden4(mlp.hidden3(mlp.hidden2(mlp.hidden1(torch.cat([b_x, torch.from_numpy(np.array(dataX_adversarial[:1500]).astype(np.float32))], 0)))))))))))).flatten()
            train_loss1=loss_func(output,b_y)
            temp_y=np.concatenate((np.array(np.zeros((len(outputc)-1500), dtype=int)), np.array(np.ones((1500), dtype=int))), axis=0)
            train_loss2=loss_funcc(outputc,torch.from_numpy(temp_y.astype(np.float32)))
            train_loss=train_loss1-train_loss2+1
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        train_loss_all.append(train_loss.item())
        train_loss_all_2.append(train_loss2.item())
        #print(epoch,train_loss.item())
        #print(epoch,train_loss2.item())
        if train_loss.item() < 0.0001:
            break
    plt.subplot(2,1,1) 
    plt.plot(train_loss_all,"r-",label='Source')
    plt.title('Total loss')
    plt.subplot(2,1,2) 
    plt.plot(train_loss_all_2,"b-",label='Classifier')
    plt.title('Classifier loss',y = -0.33)
    plt.show()
    return(train_loss_all[-1])

#Initialize MLP model
def initialize_mlp(neuron_number_layers,train_loader,learning_rate,tp):
    mlp = MLPmodel(neuron_number_layers[0],neuron_number_layers[1],neuron_number_layers[2],neuron_number_layers[3],neuron_number_layers[4],neuron_number_layers[5],neuron_number_layers[6],neuron_number_layers[7],neuron_number_layers[8],neuron_number_layers[9],tp)
    print(mlp)
    state = train_mlp(mlp,train_loader,50,learning_rate,'r',1)
    for i in range(3):
        if state == 1:
            break
        if state == 0:
            mlp = MLPmodel(neuron_number_layers[0],neuron_number_layers[1],neuron_number_layers[2],neuron_number_layers[3],neuron_number_layers[4],neuron_number_layers[5],neuron_number_layers[6],neuron_number_layers[7],neuron_number_layers[8],neuron_number_layers[9],tp)
            state = train_mlp(mlp,train_loader, 50,learning_rate,'r',1)
    if state == 1:
        print('initialize succeed, continue')
    if state == 0:
        print('Initialize failed')
    model_save_path = '/home/xychen/Carrior Mobility/MLP_MODELS_8/model_initialized_fn'+str(neuron_number_layers[0])+'_'+str(neuron_number_layers[1])+'_'+str(neuron_number_layers[2])+'_'+str(neuron_number_layers[3])+'_'+str(neuron_number_layers[4])+'_'+str(neuron_number_layers[5])+'_'+str(neuron_number_layers[6])+'_'+str(neuron_number_layers[7])+'_'+str(neuron_number_layers[8])+'_'+str(neuron_number_layers[9])+'.pt'
    torch.save(mlp, model_save_path)
    return(model_save_path)

def initialize_mlp_C(innum,neuron_number_layers,train_loader,learning_rate,tp):
    mlp = MLPmodel_C(innum,3+int(neuron_number_layers[0]/268*innum),3+int(neuron_number_layers[1]/268*innum),3+int(neuron_number_layers[2]/268*innum),3+int(neuron_number_layers[3]/268*innum),3+int(neuron_number_layers[4]/268*innum),3+int(neuron_number_layers[5]/268*innum),3+int(neuron_number_layers[6]/268*innum),3+int(neuron_number_layers[7]/268*innum),3+int(neuron_number_layers[8]/268*innum),3+int(neuron_number_layers[9]/268*innum),tp)
    print(mlp)
    state = train_mlp(mlp,train_loader,75,learning_rate,'c',1)
    for i in range(2):
        if state == 1:
            break
        if state == 0:
            mlp = MLPmodel_C(innum,3+int(neuron_number_layers[0]/268*innum),3+int(neuron_number_layers[1]/268*innum),3+int(neuron_number_layers[2]/268*innum),3+int(neuron_number_layers[3]/268*innum),3+int(neuron_number_layers[4]/268*innum),3+int(neuron_number_layers[5]/268*innum),3+int(neuron_number_layers[6]/268*innum),3+int(neuron_number_layers[7]/268*innum),3+int(neuron_number_layers[8]/268*innum),3+int(neuron_number_layers[9]/268*innum),tp)
            state = train_mlp(mlp,train_loader, 75,learning_rate,'c',1)
    if state == 1:
        print('initialize succeed, continue')
    if state == 0:
        print('Initialize failed')
    model_save_path = '/home/xychen/Carrior Mobility/MLP_MODELS_8/model_initialized_c_fn'+str(neuron_number_layers[0])+'_'+str(neuron_number_layers[1])+'_'+str(neuron_number_layers[2])+'_'+str(neuron_number_layers[3])+'_'+str(neuron_number_layers[4])+'_'+str(neuron_number_layers[5])+'_'+str(neuron_number_layers[6])+'_'+str(neuron_number_layers[7])+'_'+str(neuron_number_layers[8])+'_'+str(neuron_number_layers[9])+'.pt'
    torch.save(mlp, model_save_path)
    return(model_save_path)

def exclude_zeros(fv):
    newfv = []
    exclude = []
    arrfv = np.array(fv)
    for i in range(len(arrfv.T)):
        if sum(arrfv.T[i].tolist()) == 0:
            exclude.append(i)
    for i in range(len(fv)):
        temp = []
        for j in range(len(fv[i])):
            if j not in exclude:
                temp.append(fv[i][j])
        newfv.append(temp)
    return(newfv)

#evaluate model accuracy
def view_model_accuracy(trainX,trainY,testX,testY,Ymin):
    plt.figure(figsize=(6,6))
    left = min([min(trainX),min(trainY),min(testX),min(testY)])-Ymin-1
    right = max([max(trainX),max(trainY),max(testX),max(testY)])-Ymin+1
    x1 = np.linspace(left,right,500)
    y1 = x1
    plt.plot(x1,y1,color = 'black',linewidth = 1,dashes=[6, 2])
    plt.ylabel('Predict Value',fontsize=25)
    plt.xlabel('True Value',fontsize=25)
    print('R2_training: ',r2_score(np.array(trainY)-Ymin,trainX-Ymin))
    print('RMSE_training:',mean_squared_error(np.array(trainY)-Ymin,trainX-Ymin)**(1/2))
    print('R2_testing: ',r2_score(np.array(testY)-Ymin,testX-Ymin))
    print('RMSE_testing:',mean_squared_error(np.array(testY)-Ymin,testX-Ymin)**(1/2))
    plt.scatter(np.array(trainY)-Ymin, trainX-Ymin, c = 'royalblue',alpha = 0.4,label = 'training')
    plt.scatter(np.array(testY)-Ymin, testX-Ymin, c = 'salmon',alpha = 0.4,label = 'testing')
    plt.legend(fontsize=20)
    plt.show()
    return(r2_score(np.array(trainY)-Ymin,trainX-Ymin),r2_score(np.array(testY)-Ymin,testX-Ymin))


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
        #R2 = r2_score(testY, prediction)
        RMSE = mean_squared_error(testY, prediction)**(1/2)
        return RMSE
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
        print('RMSE_testing:',mean_squared_error(testY, predictY_test)**(1/2))
        print('r2_testing:',r2_score(testY, predictY_test))
        print('RMSE_training:',mean_squared_error(trainY, predictY_train)**(1/2))
        print('r2_training:',r2_score(trainY, predictY_train))
        overfitting.append(mean_squared_error(trainY, predictY_train)**(1/2)-mean_squared_error(testY, predictY_test)**(1/2))
    index_sorted,testings_sorted,predictions_sorted = (list(t) for t in zip(*sorted(zip(index,testings,predictions))))
    return(best_models,testings_sorted,predictions_sorted,feature_importance,np.mean(overfitting))


def view_accuracy(testings, predictions):
    print("RMSE:",mean_squared_error(testings, predictions)**(1/2))
    print("R2:",r2_score(testings, predictions))
    plt.figure(figsize=(6,6))
    plt.ylabel('Predict Bandgap (eV)',fontsize=25)
    plt.xlabel('HSE Bandgap (eV)',fontsize=25)
    x1 = np.linspace(min([min(testings),min(predictions)])-1,max([max(testings),max(predictions)])+1,500)#从(-1,1)均匀取50个点
    y1 = x1
    plt.plot(x1,y1,color = 'black',linewidth = 1,dashes=[6, 2])
    plt.scatter(testings, predictions, c = 'salmon',alpha=0.8,label='10-fold_cross_validation')
    plt.legend(fontsize=15)
    plt.show()
    
###plot scatter of testing value and prediction value after random cv
def view_accuracy_random_cv(testings, predictions_randcv):
    meanprediction = []
    stdprediction = []
    RMSE = []
    R2 = []
    for i in range(len(predictions_randcv)):
        RMSE.append(mean_squared_error(testings, predictions_randcv[i])**(1/2))
        R2.append(r2_score(testings, predictions_randcv[i]))
    for i in range(len(testings)):
        temp = []
        for j in range(len(predictions_randcv)):
            temp.append(predictions_randcv[j][i])
        meanprediction.append(np.mean(temp))
        stdprediction.append(np.std(temp))
    print("RMSE_mean:",mean_squared_error(testings, meanprediction)**(1/2))
    print("R2_mean:",r2_score(testings, meanprediction))
    print("mean_RMSE:",np.mean(RMSE))
    print("mean_R2:",np.mean(R2))
    print("std_RMSE:",np.std(RMSE))
    print("std_R2:",np.std(R2))#show RMSE R2 and their standard error
    plt.figure(figsize=(6,6))
    plt.ylabel('Predict Bandgap (eV)',fontsize=25)
    plt.xlabel('HSE Bandgap (eV)',fontsize=25)
    x1 = np.linspace(min(testings)-1,max(testings)+1,500)
    y1 = x1
    plt.plot(x1,y1,color = 'black',linewidth = 1,dashes=[6, 2])
    plt.errorbar(testings, meanprediction,yerr = stdprediction,elinewidth=0.2,alpha=0.7,ecolor = 'firebrick',capsize=3,capthick=0.5,linestyle="none")
    plt.scatter(testings, meanprediction, c = 'salmon',alpha=0.6,label='10-fold_cross_validation')
    plt.legend(fontsize=20)
    plt.show()
    return(np.mean(R2),np.mean(RMSE))


# In[6]:


get_ipython().run_line_magic('time', '')
#Source-bulk
path_csv_source = '../property_feature_source_U.csv'
data_source = pd.read_csv(path_csv_source)
data_source_clean = remove_outliers(data_source,'Formula','Mass_eff_Electron',[1e-7,1e3],'TOP')
print('Property list:',list(data_source))
print('Sample size:',len(data_source))
print('Cleaned Sample size:',len(data_source_clean))
#Adversarial-2D
path_csv_adversarial = '../property_feature_adversarial_U.csv'
data_adversarial = pd.read_csv(path_csv_adversarial)
data_adversarial_clean = remove_outliers(data_adversarial,'formula','gap',[0,10],'BOT')
print('Property list:',list(data_adversarial))
print('Sample size:',len(data_adversarial))
print('Cleaned Sample size:',len(data_adversarial_clean))


# In[11]:


X = data_source_clean['feature'].values.tolist()
Y = log_data(data_source_clean['Mass_eff_Electron'].values.tolist())
XS,YS = [],[]
for i in range(len(X)):
    if X[i] != '0':        
        XS.append(X[i])
        YS.append(Y[i])


# In[13]:


XA = cbfv_tolist(data_adversarial_clean['feature'].values.tolist())
YA = data_adversarial_clean['gap'].values.tolist()
for i in range(len(XA)):
    for j in range(len(XA[i])):
        if np.isnan(XA[i][j]) == 1:
            XA[i][j] = 0


# In[15]:


dataX_source = data_unifier(cbfv_tolist(XS))
whereisnan=np.isnan(dataX_source)
print(dataX_source[whereisnan])
dataX_source[whereisnan]=0
dataY_source = YS
train_loader_s, trainX_s, testX_s, trainY_s, testY_s = batch_loader(dataX_source,dataY_source,1500)
dataX_adversarial,dataY_adversarial = oversample(XA,YA,len(dataX_source))
#whereisnan=np.isnan(dataX_adversarial)
#print(dataX_adversarial[whereisnan])
#dataX_adversarial[whereisnan]=0
for i in range(len(dataY_adversarial)):
    dataY_adversarial[i] = float(dataY_adversarial[i])
train_loader_a, trainX_a, testX_a, trainY_a, testY_a = batch_loader(dataX_adversarial,dataY_adversarial,1500)
#check data size
print(len(dataX_source),len(dataX_adversarial),len(dataY_source),len(dataY_adversarial))


# In[16]:


#train
loss_best = []
model_best = []
#Initialize mlp model
neuron_number_layers = [224,224,170,170,104,104,46,46,15,15]
initialized_mlp = initialize_mlp(neuron_number_layers,train_loader_s,0.0001,'r')
mlp = torch.load(initialized_mlp)
#train_mlp(mlp,train_loader_s, 100,0.0001,'r',1)
print(initialized_mlp,len(trainX_s),len(testX_s))
h1_s = np.concatenate((mlp.hidden1(torch.from_numpy(np.array(trainX_s).astype(np.float32))).detach().numpy(), mlp.hidden1(torch.from_numpy(np.array(testX_s).astype(np.float32))).detach().numpy()),axis=0)
h1_t = np.concatenate((mlp.hidden1(torch.from_numpy(np.array(trainX_a).astype(np.float32))).detach().numpy(), mlp.hidden1(torch.from_numpy(np.array(testX_a).astype(np.float32))).detach().numpy()),axis=0)
h2_s = mlp.hidden2(torch.from_numpy(h1_s.astype(np.float32))).detach().numpy()
h2_t = mlp.hidden2(torch.from_numpy(h1_t.astype(np.float32))).detach().numpy()
h3_s = mlp.hidden3(torch.from_numpy(h2_s.astype(np.float32))).detach().numpy()
h3_t = mlp.hidden3(torch.from_numpy(h2_t.astype(np.float32))).detach().numpy()
h4_s = mlp.hidden4(torch.from_numpy(h3_s.astype(np.float32))).detach().numpy()
h4_t = mlp.hidden4(torch.from_numpy(h3_t.astype(np.float32))).detach().numpy()
h5_s = mlp.hidden5(torch.from_numpy(h4_s.astype(np.float32))).detach().numpy()
h5_t = mlp.hidden5(torch.from_numpy(h4_t.astype(np.float32))).detach().numpy()
h6_s = mlp.hidden6(torch.from_numpy(h5_s.astype(np.float32))).detach().numpy()
h6_t = mlp.hidden6(torch.from_numpy(h5_t.astype(np.float32))).detach().numpy()
h7_s = mlp.hidden7(torch.from_numpy(h6_s.astype(np.float32))).detach().numpy()
h7_t = mlp.hidden7(torch.from_numpy(h6_t.astype(np.float32))).detach().numpy()
h8_s = mlp.hidden8(torch.from_numpy(h7_s.astype(np.float32))).detach().numpy()
h8_t = mlp.hidden8(torch.from_numpy(h7_t.astype(np.float32))).detach().numpy()
h9_s = mlp.hidden9(torch.from_numpy(h8_s.astype(np.float32))).detach().numpy()
h9_t = mlp.hidden9(torch.from_numpy(h8_t.astype(np.float32))).detach().numpy()
hidden_s = mlp.hidden10(torch.from_numpy(h9_s.astype(np.float32))).detach().numpy()
hidden_t = mlp.hidden10(torch.from_numpy(h9_t.astype(np.float32))).detach().numpy()
#hidden_s = h8_s
#hidden_t = h8_t
dataX_classifier = np.concatenate((hidden_s,hidden_t),axis=0).tolist()
dataY_classifier = [0 for _ in range(len(hidden_s))]
dataY_classifier.extend([1 for _ in range(len(hidden_t))])
train_loader_c, trainX_c, testX_c, trainY_c, testY_c = batch_loader(dataX_classifier,dataY_classifier,3000)
initialized_mlp_c = initialize_mlp_C(len(dataX_classifier[0]),neuron_number_layers,train_loader_c,1e-3,'c')
mlp_c = torch.load(initialized_mlp_c)

#Adversarial training
loss = []
model = []
for i in range(5):
    loss_all = joint_train_mlp(mlp,train_loader_s,mlp_c,200)
    loss.append(loss_all)
    h1_s = np.concatenate((mlp.hidden1(torch.from_numpy(np.array(trainX_s).astype(np.float32))).detach().numpy(), mlp.hidden1(torch.from_numpy(np.array(testX_s).astype(np.float32))).detach().numpy()),axis=0)
    h1_t = np.concatenate((mlp.hidden1(torch.from_numpy(np.array(trainX_a).astype(np.float32))).detach().numpy(), mlp.hidden1(torch.from_numpy(np.array(testX_a).astype(np.float32))).detach().numpy()),axis=0)
    h2_s = mlp.hidden2(torch.from_numpy(h1_s.astype(np.float32))).detach().numpy()
    h2_t = mlp.hidden2(torch.from_numpy(h1_t.astype(np.float32))).detach().numpy()
    h3_s = mlp.hidden3(torch.from_numpy(h2_s.astype(np.float32))).detach().numpy()
    h3_t = mlp.hidden3(torch.from_numpy(h2_t.astype(np.float32))).detach().numpy()
    h4_s = mlp.hidden4(torch.from_numpy(h3_s.astype(np.float32))).detach().numpy()
    h4_t = mlp.hidden4(torch.from_numpy(h3_t.astype(np.float32))).detach().numpy()
    h5_s = mlp.hidden5(torch.from_numpy(h4_s.astype(np.float32))).detach().numpy()
    h5_t = mlp.hidden5(torch.from_numpy(h4_t.astype(np.float32))).detach().numpy()
    h6_s = mlp.hidden6(torch.from_numpy(h5_s.astype(np.float32))).detach().numpy()
    h6_t = mlp.hidden6(torch.from_numpy(h5_t.astype(np.float32))).detach().numpy()
    h7_s = mlp.hidden7(torch.from_numpy(h6_s.astype(np.float32))).detach().numpy()
    h7_t = mlp.hidden7(torch.from_numpy(h6_t.astype(np.float32))).detach().numpy()
    h8_s = mlp.hidden8(torch.from_numpy(h7_s.astype(np.float32))).detach().numpy()
    h8_t = mlp.hidden8(torch.from_numpy(h7_t.astype(np.float32))).detach().numpy()
    h9_s = mlp.hidden9(torch.from_numpy(h8_s.astype(np.float32))).detach().numpy()
    h9_t = mlp.hidden9(torch.from_numpy(h8_t.astype(np.float32))).detach().numpy()
    hidden_s = mlp.hidden10(torch.from_numpy(h9_s.astype(np.float32))).detach().numpy()
    hidden_t = mlp.hidden10(torch.from_numpy(h9_t.astype(np.float32))).detach().numpy()
    #hidden_s = h8_s
    #hidden_t = h8_t
    dataX_classifier = np.concatenate((hidden_s,hidden_t),axis=0).tolist()
    dataY_classifier = [0 for _ in range(len(hidden_s))]
    dataY_classifier.extend([1 for _ in range(len(hidden_t))])
    train_loader_c, trainX_c, testX_c, trainY_c, testY_c = batch_loader(dataX_classifier,dataY_classifier,1500)
    train_mlp(mlp_c,train_loader_c, 100,0.0001,'c',1)
    model_save_path_final = '/home/xychen/Carrior Mobility/MLP_MODELS/model_T2_layers_hidden5_e-102'+str(i)+'_'+str(neuron_number_layers[0])+'_'+str(neuron_number_layers[1])+'_'+str(neuron_number_layers[2])+'_'+str(neuron_number_layers[3])+'_'+str(neuron_number_layers[4])+'.pt'
    model.append(model_save_path_final)
    torch.save(mlp, model_save_path_final)
loss_best.append(min(loss))
model_best.append(model[loss.index(min(loss))])
print(min(loss),model[loss.index(min(loss))])


# In[ ]:




