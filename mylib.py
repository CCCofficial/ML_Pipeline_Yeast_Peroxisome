import numpy as np 
import pandas as pd
import csv 
import os 
import matplotlib.pyplot as plt

from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error






def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  #plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 2])
  plt.xlabel('Epoch')
  plt.ylabel('Error')
  plt.legend()
  plt.grid(True)


def calculate_r2(Y_test,y_pred_test):
    Y_test2=np.expm1(Y_test)
    y_pred_test2=np.expm1(y_pred_test.squeeze())
    
    correlation_matrix = np.corrcoef(Y_test2, y_pred_test2)
    corr = correlation_matrix[0,1]
    r_squared = corr**2
    return r_squared

def calculate_mae(Y_test,y_pred_test):
    Y_test2=np.expm1(Y_test)
    y_pred_test2=np.expm1(y_pred_test.squeeze())
    
    mae = mean_absolute_error(Y_test2,y_pred_test2)
    return mae

def calculate_mse(Y_test,y_pred_test):
    Y_test2=np.expm1(Y_test)
    y_pred_test2=np.expm1(y_pred_test.squeeze())
    
    mse = mean_squared_error(Y_test2,y_pred_test2)
    return mse



def plot_train_test(model_name,Y_train,y_pred_train,Y_test,y_pred_test):

#max_y2=np.max(all_data.iloc[:,-1],axis=0)
#min_y2=np.min(all_data.iloc[:,-1],axis=0)
#model_name = "CNN+LSTM model (original data)"
#model_name = "Linear Regression model"

    figsize = (15,60)

    figure, ax = plt.subplots(1,2,figsize=figsize)
    figure.suptitle('{}'.format(model_name),y=0.56,size=20)


    Y_train22=np.expm1(Y_train)
    y_pred_train22=np.expm1(y_pred_train.squeeze())
    Y_test22=np.expm1(Y_test)
    y_pred_test22=np.expm1(y_pred_test.squeeze())



#plt.figure(figsize=[15,70])
#plt.subplot(1,2,1)



    ax[0].scatter(Y_train22, y_pred_train22)
    ax[0].set_xlabel('True Values', fontsize=20)
    ax[0].set_ylabel('Predictions', fontsize=20)

    ax[0].axline((0,0),slope=1,color="red")

#lims = [min_y2, max_y2]
#plt.plot(lims, lims,color='red')
    correlation_matrix = np.corrcoef(Y_train22, y_pred_train22)
    corr = correlation_matrix[0,1]
    r_squared = corr**2
    r_squared2 = r2_score(Y_train22,y_pred_train22)

#ax[0].set_title('Train\n'+'R$^2$='+str(r_squared)[:5] + '/'+str(r_squared2)[:5], fontsize=20)
    ax[0].set_title(f'Train\nR$^2$={r_squared:.3f}', fontsize=20)

#plt.title('Train\n'+'R$^2$='+str(r_squared)[:5], fontsize=20)
    ax[0].axis('square')
#plt.xticks(fontsize=20)
#plt.yticks(fontsize=20)


#plt.subplot(1,2,2)
    ax[1].scatter(Y_test22, y_pred_test22)
    ax[1].set_xlabel('True Values', fontsize=20)
#plt.ylabel('Predictions', fontsize=20)

    ax[1].axline((0,0),slope=1,color="red")

#plt.plot(lims, lims,color='red')
    correlation_matrix = np.corrcoef(Y_test22, y_pred_test22)
    corr = correlation_matrix[0,1]
    r_squared = corr**2
    r_squared2 = r2_score(Y_test22,y_pred_test22)

#ax[1].set_title('Test\n'+'R$^2$='+str(r_squared)[:5] + '/'+str(r_squared2)[:5], fontsize=20)
    ax[1].set_title(f'Test\nR$^2$={r_squared:.3f}', fontsize=20)

    ax[1].axis('square')
#plt.xticks(fontsize=20)
#plt.yticks(fontsize=20)

def plot_train_test_add_test(model_name,Y_train,y_pred_train,Y_test,y_pred_test,index,name):

#max_y2=np.max(all_data.iloc[:,-1],axis=0)
#min_y2=np.min(all_data.iloc[:,-1],axis=0)
#model_name = "CNN+LSTM model (original data)"
#model_name = "Linear Regression model"

    figsize = (15,60)

    figure, ax = plt.subplots(1,2,figsize=figsize)
    figure.suptitle('{}'.format(model_name),y=0.56,size=20)


    Y_train22=np.expm1(Y_train)
    y_pred_train22=np.expm1(y_pred_train.squeeze())
    Y_test22=np.expm1(Y_test)
    y_pred_test22=np.expm1(y_pred_test.squeeze())



#plt.figure(figsize=[15,70])
#plt.subplot(1,2,1)



    ax[0].scatter(Y_train22, y_pred_train22)
    ax[0].set_xlabel('True Values', fontsize=20)
    ax[0].set_ylabel('Predictions', fontsize=20)

    ax[0].axline((0,0),slope=1,color="red")

#lims = [min_y2, max_y2]
#plt.plot(lims, lims,color='red')
    correlation_matrix = np.corrcoef(Y_train22, y_pred_train22)
    corr = correlation_matrix[0,1]
    r_squared = corr**2
    r_squared2 = r2_score(Y_train22,y_pred_train22)

#ax[0].set_title('Train\n'+'R$^2$='+str(r_squared)[:5] + '/'+str(r_squared2)[:5], fontsize=20)
    ax[0].set_title('Train\n'+'R$^2$='+str(r_squared)[:5], fontsize=20)

#plt.title('Train\n'+'R$^2$='+str(r_squared)[:5], fontsize=20)
    ax[0].axis('square')
#plt.xticks(fontsize=20)
#plt.yticks(fontsize=20)


#plt.subplot(1,2,2)
    ax[1].scatter(Y_test22, y_pred_test22)
    ax[1].set_xlabel('True Values', fontsize=20)
#plt.ylabel('Predictions', fontsize=20)

    ax[1].axline((0,0),slope=1,color="red")

#plt.plot(lims, lims,color='red')
    correlation_matrix = np.corrcoef(Y_test22, y_pred_test22)
    corr = correlation_matrix[0,1]
    r_squared = corr**2
    r_squared2 = r2_score(Y_test22,y_pred_test22)

#ax[1].set_title('Test\n'+'R$^2$='+str(r_squared)[:5] + '/'+str(r_squared2)[:5], fontsize=20)
    ax[1].set_title('Test\n'+'R$^2$='+str(r_squared)[:5], fontsize=20)

    ax[1].axis('square')
    df_result.loc[index,name] = r_squared
#plt.xticks(fontsize=20)
#plt.yticks(fontsize=20)