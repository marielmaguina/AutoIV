#!/usr/bin/env python
# coding: utf-8

# In[169]:


import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
from matplotlib import pyplot as plt
import numpy as np
import random
import datetime
import shutil
import csv
import statsmodels.api as sm
import econtools
import econtools.metrics as mt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from tabulate import tabulate

#Look at the Github tutorial https://happygitwithr.com/


# In[170]:


#1. Helper function for data

def get_data(dataset, rep_dim=4, num=0):
    #dataset options: sin, step, abs, linear, poly2d, poly3d
    
    #1. Import data
    synthetic_data = pd.read_csv("data/"+dataset+"/"+dataset+"-train_500/data/exp"+str(num)+".csv") [[" ye"]]
    IV_representation = pd.read_csv("data/"+dataset+"/autoiv-"+dataset+"/autoiv-"+dataset+"-train_500-rep_"+str(rep_dim)+"/data/exp"+str(num)+".csv")
    True_IV = dict(np.load("data/"+dataset+"/"+dataset+"-train_500/data/exp"+str(num)+".npz"))["z"]
    True_IV = pd.DataFrame(True_IV, columns=["TrueZ_1", "TrueZ_2"])
        
    #2. Clen data 
    synthetic_data.columns = ["ye"]
    IV_representation.reset_index(inplace=True, drop=False)
    IV_representation.columns = ["x", "x_pre", "y", "y_pre", "Z_1", "Z_2", "Z_3", "Z_4", "C_1", "C_2", "C_3", "C_4"]
    
    #3. Concat data and clean
    data = pd.concat([synthetic_data, IV_representation, True_IV], axis=1)
    data = data [['x', 'x_pre', 'y', 'y_pre', 'ye', 'Z_1', 'Z_2', 'Z_3', 'Z_4', 'C_1', 'C_2', 'C_3', 'C_4', "TrueZ_1", "TrueZ_2"]]
    
    return data

#get_data(dataset="abs", rep_dim=4, num=0)


# In[196]:


#2. Helper function for graphs

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [X]')
    plt.legend()
    plt.grid(True)

def plot_x(X, Y, Y_fitted):
    plt.scatter(X, Y, label='Data')
    plt.scatter(X, Y_fitted, color='k', label='Predictions')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


# In[203]:


#3. Experiment I: Direct NN

def vanNN(dataset, output, num):
    
    #a. Get data
    data = get_data(dataset=dataset, num=num)
    
    #b. Split the data into training and test
    train_data = data.sample(frac=0.8, random_state=0)
    test_data = data.drop(train_data.index)
    
    #c. Get the LHS variables
    Y_train = train_data[["ye"]]
    Y_test = test_data[["ye"]]
    
    if output == "Y train":
        return Y_train
    elif output == "Y test":
        return Y_test
    
    #d. Get the RHS variables
    X_train = train_data[["x_pre"]]
    X_test = test_data[["x_pre"]]
        
    #e. Create layers
    model = tf.keras.Sequential([layers.Dense(units=1, activation='relu')])
    model.add(layers.Dense(units=64, activation='relu'))
    model.add(layers.Dense(units=32, activation='relu'))
    model.add(layers.Dense(units=1, activation='linear'))

    if output == "Model":
        return model
    elif output == "Model Summary":
        return X_model.summary()
        
    #f. Configure the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss='mean_absolute_error')
    
    #g. Fit model
    History = model.fit(X_train, 
                    Y_train,
                    epochs=100,
                    verbose=0,
                    validation_split = 0.2)
    
    if output == "History":
        hist = pd.DataFrame(History.history)
        hist['epoch'] = History.epoch
        return hist
    
    if output == "Loss Plot":
        plot_loss(History)
        
    #h. Fitted values
    Y_train_fitted = model.predict(X_train)
    Y_test_fitted = model.predict(X_test)
    
    if output == "Fitted Values":
        return Y_train_fitted, Y_test_fitted
    
    #i. Plot predictions and data
    if output == "Train Data and Prediction Plot":
        plot_x(X_train, Y_train, Y_train_fitted)
    elif output == "Test Data and Prediction Plot":
        plot_x(X_test, Y_test, Y_test_fitted)
        
    #f. MSE
    mse_train = mean_squared_error(train_data["y"], Y_train_fitted)
    mse_test = mean_squared_error(test_data["y"], Y_test_fitted)
    
    if output == "MSE":
        return mse_train

#vanNN(dataset="poly2d", output="Train Data and Prediction Plot", num=0)


# In[173]:


#4. Experiment II: Two Stage Least Squares with Linear Models

def twoSLS(dataset, output, num, IV):
    
    #a. Get data
    data = get_data(dataset=dataset, num=num)
    
    #b. Choose IVs
    if IV == "AutoIV":
        IVs = ["Z_1", "Z_2", "Z_3", "Z_4"]
    elif IV == "TrueIV":
        IVs = ["TrueZ_1", "TrueZ_2"]
    
    #c. Fit model
    model = mt.ivreg(data, "ye", "x_pre", IVs , ["C_1", "C_2", "C_3", "C_4"], addcons=True, iv_method="2sls")
    if output == "Model":
        return model
    
    #d. Get fitted values
    Y_fitted = model.yhat
    if output == "Fitted Values":
        return Y_fitted
    
    #e. MSE
    mse = mean_squared_error(data["y"], Y_fitted)
    if output == "MSE":
        return mse
    
    #f. Plot
    if output == "Plot":
        plot_x(X=data["x_pre"], Y=data["ye"], Y_fitted=Y_fitted)
        
#twoSLS(dataset="poly3d", output="Plot", num=1, IV="AutoIV")


# In[174]:


#5. Experiment III: Two Stage Least Square with Polynomial Basis and Ridge Regularization

def twoSLS_PolyRidge(dataset, output, num, IV):
    
    #a. Get data
    data = get_data(dataset=dataset, num=num)
    
    #b. Choose IVs
    if IV == "AutoIV":
        IVs = ["Z_1", "Z_2", "Z_3", "Z_4"]
    elif IV == "TrueIV":
        IVs = ["TrueZ_1", "TrueZ_2"]
    
    #c. Run the first stage regression and save the fitted values
    model_1 = mt.reg(data, "x_pre", IVs + ['C_1', 'C_2', 'C_3', 'C_4'], addcons=True)
    X_fitted = model_1.yhat
    
    if output == "Model 1":
        print(model_1)
    elif output == "X fitted":
        return X_fitted
    
    #d. Turn all required variables into numpy array format
    Y = np.array(data["ye"])
    X = np.array(X_fitted).reshape(-1, 1)
    C_1 = np.array(data["C_1"]).reshape(-1, 1)
    C_2 = np.array(data["C_2"]).reshape(-1, 1)
    C_3 = np.array(data["C_3"]).reshape(-1, 1)
    C_4 = np.array(data["C_4"]).reshape(-1, 1)
    RHS_variables = np.concatenate((X, C_1, C_2, C_3, C_4), axis=1)
    
    #e. Create polynomial model
    poly_features = PolynomialFeatures(degree=5)
    X_poly = poly_features.fit_transform(RHS_variables)
    
    #f. Perform polynomial regression
    poly_regression = sm.OLS(Y, X_poly)
    model_2 = poly_regression.fit()
    
    if output == "Model 2":
        print(model_2.summary())
        #Note: There are 251 coefficients
        
    #g. Create ridge regularization model
    model_3 = Ridge(alpha=10)
    model_3.fit(X,Y)
    
    #h. Obtain fitted values and calculate MSE
    Y_fitted = model_3.predict(X)
    mse = mean_squared_error(data["y"], Y_fitted)
    
    if output == "MSE":
        return mse
    
    if output == "Plot":
        plot_x(X=data["x_pre"], Y=data["ye"], Y_fitted=Y_fitted)

#twoSLS_PolyRidge(dataset="poly3d", output="X fitted vs. x_pre", num=15, IV="TrueIV")


# In[206]:


#5. Experiment IV: Two Stage Least Square with NN

def twoSLS_NN(dataset, output, num, IV):
    
    #a. Get data
    data = get_data(dataset=dataset, num=num)

    #b. Choose IVs
    if IV == "AutoIV":
        IVs = ["Z_1", "Z_2", "Z_3", "Z_4"]
    elif IV == "TrueIV":
        IVs = ["TrueZ_1", "TrueZ_2"]
    
    #c. Run the first stage regression and save the fitted values
    model_1 = mt.reg(data, "x_pre", IVs + ['C_1', 'C_2', 'C_3', 'C_4'], addcons=True) #Not sure if x or x_pre
    X_fitted = model_1.yhat
    data["X_fitted"] = X_fitted
        
    if output == "Model 1":
        print(model_1)
        
    #d. Split the data into training and test
    train_data = data.sample(frac=0.8, random_state=0)
    test_data = data.drop(train_data.index)
    
    #e.  Get the LHS variables
    Y_train = train_data[["ye"]]
    Y_test = test_data[["ye"]]
    
    if output == "Y train":
        return Y_train
    elif output == "Y test":
        return Y_test
    
    #f. Get the RHS variables
    X_train = train_data[["X_fitted"]]
    X_test = test_data[["X_fitted"]]
      
    #g. Create a layer    
    model = tf.keras.Sequential([layers.Dense(units=1, activation='relu')])
    model.add(layers.Dense(units=64, activation='relu'))
    model.add(layers.Dense(units=32, activation='relu'))
    model.add(layers.Dense(units=1, activation='linear'))
    
    #h. Configure the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss='mean_absolute_error')
    
    #i. Fit model
    History = model.fit(X_train, 
                    Y_train,
                    epochs=100,
                    verbose=0,
                    validation_split = 0.2)
    
    if output == "History":
        hist = pd.DataFrame(History.history)
        hist['epoch'] = History.epoch
        return hist
    
    if output == "Loss Plot":
        plot_loss(History)
        
    #j. Fitted values
    Y_train_fitted = model.predict(X_train)
    Y_test_fitted = model.predict(X_test)
    
    if output == "Fitted Values":
        return Y_train_fitted, Y_test_fitted
    
    #k. Plot predictions and data
    if output == "Train Data and Prediction Plot":
        plot_x(X_train, Y_train, Y_train_fitted)
    elif output == "Test Data and Prediction Plot":
        plot_x(X_test, Y_test, Y_test_fitted)
        
    #l. MSE
    mse_train = mean_squared_error(train_data["y"], Y_train_fitted)
    mse_test = mean_squared_error(Y_test, Y_test_fitted)
    
    if output == "MSE":
        return mse_train
    
    #m. 
    if output == "Plot":
        plot_x(X=train_data["x_pre"], Y=train_data["ye"], Y_fitted=Y_train_fitted)
    
#twoSLS_NN(dataset="poly3d", output="Plot", num=0, IV="AutoIV")


# In[176]:


#6. Table with results step 1 

def create_table(experiment, IV):
    
    columns = ["sin", "step", "abs", "linear", "poly2d", "poly3d"]
    df = pd.DataFrame(columns=columns)
    
    for num in range(0,20):
        empty_list = []
        
        print(num)
        
        for dataset in columns:
            
            if experiment == "VanNN":
                MSE = vanNN(dataset=dataset, output="MSE", num=num).round(4)
                
            elif experiment == "2SLS":
                MSE = twoSLS(dataset=dataset, output="MSE", num=num, IV=IV).round(4)
                
            elif experiment == "Poly-Ridge":
                MSE = twoSLS_PolyRidge(dataset=dataset, output="MSE", num=num, IV=IV).round(4)
                
            elif experiment == "2SLS-NN":
                MSE = twoSLS_NN(dataset=dataset, output="MSE", num=num, IV=IV).round(4)
                
            empty_list.append(MSE)                
        
        df.loc[len(df)] = empty_list
    
    return df

#table = create_table(experiment="Poly-Ridge")


# In[232]:


#7. Table with results step 2

def table_meanstd(IV):
    
    columns = ["experiment", "sin_mean", "step_mean", "abs_mean", "linear_mean", "poly2d_mean", "poly3d_mean", "sin_std", "step_std", "abs_std", "linear_std", "poly2d_std", "poly3d_std"]
    df = pd.DataFrame(columns=columns)
    
    
    for experiment in ["VanNN"]:#["2SLS", "Poly-Ridge", "2SLS-NN", "VanNN"]:
        
        empty_list = []
        empty_list.append(experiment)
        table = create_table(experiment=experiment, IV=IV)
        
        for dataset in ["sin", "step", "abs", "linear", "poly2d", "poly3d"]:    
            mean = table[dataset].mean()
            empty_list.append(mean)
        
        for dataset in ["sin", "step", "abs", "linear", "poly2d", "poly3d"]:
            std = table[dataset].std()
            empty_list.append(std)
            
        df.loc[len(df)] = empty_list
    return df

#table_AutoIV = table_meanstd(IV="AutoIV")
#table_TrueIV= table_meanstd(IV="TrueIV")
table_meanstd("hello")


# In[227]:


#8. Latex format:

def latex_format(df):
    
    df = df.round(2)

    # Create new columns as described
    columns_to_process = ['sin', 'step', 'abs', 'linear', 'poly2d', 'poly3d']
    for column in columns_to_process:
        mean_column = f'{column}_mean'
        std_column = f'{column}_std'
        df[column] = df[mean_column].astype(str) + ' ± ' + df[std_column].astype(str)

    # Drop the original mean and std columns
    df.drop(columns=['sin_mean', 'step_mean', 'abs_mean', 'linear_mean', 'poly2d_mean', 'poly3d_mean',
                     'sin_std', 'step_std', 'abs_std', 'linear_std', 'poly2d_std', 'poly3d_std'],
            inplace=True)

    #print(df)
    print(tabulate(df, headers='keys', tablefmt='latex'))
    
#latex_format(table_TrueIV)
#latex_format(table_AutoIV)


# In[233]:


#9. Fitted Values generated by True vs. Auto IV

model_True = twoSLS(dataset="poly3d", output="Fitted Values", num=1, IV="TrueIV")
model_Auto = twoSLS(dataset="poly3d", output="Fitted Values", num=1, IV="AutoIV")

plt.scatter(model_True, model_Auto, label='Data')
plt.xlabel('True')
plt.ylabel('Auto')
plt.title('Fitted Values generated by True vs. Auto IV')
plt.legend()


# In[234]:


#10. X_hat generated by True vs. Auto IV

X_fitted_Auto = twoSLS_PolyRidge(dataset="poly3d", output="X fitted", num=0, IV="AutoIV")
X_fitted_True = twoSLS_PolyRidge(dataset="poly3d", output="X fitted", num=0, IV="TrueIV")

plt.scatter(X_fitted_Auto, X_fitted_True, label='Data')
plt.xlabel('Auto')
plt.ylabel('True')
plt.title("X_hat generated by True vs. Auto IV")
plt.legend()

# This shows that they're able to recover the True IVs.
# Shrinkage: You eliminate error when you estimate the TrueIV.
# Idea: The estimated IV is specific to our data.


# In[236]:


def plots(num, IV):
    
    for dataset in ["sin", "step", "abs", "linear", "poly2d", "poly3d"]: 
    
        vanNN(dataset=dataset, output="Train Data and Prediction Plot", num=num)
        twoSLS(dataset=dataset, output="Plot", num=num, IV=IV)
        twoSLS_PolyRidge(dataset=dataset, output="Plot", num=num, IV=IV)
        twoSLS_NN(dataset=dataset, output="Plot", num=num, IV=IV) #use at least two layers


# In[237]:


#plots(0, IV="AutoIV")


# In[238]:


#plots(0, IV="TrueIV")


# In[ ]:





# In[ ]:




