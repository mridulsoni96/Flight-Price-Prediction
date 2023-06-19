#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Models

# #### Importing the required Libraries

# In[1]:


import pandas as pd 
import numpy as np
import keras
import matplotlib.pyplot as plt
import statsmodels.api as sm

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from config import *
from math import sqrt

from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# ### Reading the Cleaned Dataset file from Preprocessing

# In[2]:


df = pd.read_csv('cleaned_flight_dataset.csv')


# #### Dropping the Non-Required Columns for the analysis

# In[3]:


df = df.drop(['Date of Journey', 'Flight Code'], axis=1)


# #### Splitting the dataset from the last Column

# In[4]:


X = df.iloc[:,:-1]
y = df.iloc[:,-1]


# ### Applying One-Hot Encoding to the Categorical Columns

# In[5]:


ct = ColumnTransformer(transformers=[
    ('ohe',OneHotEncoder(drop='first',sparse_output=False,handle_unknown='ignore'),
    ['Airline','Source City','Destination City','Journey Class','Departure Time','Arrival Time', 'Day of Week'])
],remainder='passthrough')

# Fit and transform the model with the encoded values 
X= ct.fit_transform(X)
X


# #### Splitting the dataset into Testing and Training

# In[6]:


X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3, random_state=36)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[7]:


#Converting the y into float for prediction
y = np.array(y).astype('float32')


# In[8]:


#Converting X into Dataframe to assign the column header
X = pd.DataFrame(X)

# Getting column name that are lost after one hot encoding
X.columns=ct.get_feature_names_out()
X


# ### Calculation on VIF to see Multicollinearity

# In[9]:


# Function to calculate VIF
def calculate_vif(X):
    vif_df = pd.DataFrame(columns = ['Var', 'Vif'])
    x_var_names = X.columns
    for i in range(0, x_var_names.shape[0]):
        y = X[x_var_names[i]]
        x = X[x_var_names.drop([x_var_names[i]])]
        r_squared = sm.OLS(y,x).fit().rsquared
        vif = round(1/(1-r_squared),2)
        vif_df.loc[i] = [x_var_names[i], vif]
    return vif_df.sort_values(by = 'Vif', axis = 0, ascending=False, inplace=False)

calculate_vif(X)


# Excluding the remainder columns as they are not one hot encoded.
# VIF score is to check the multicollinearity after one hot encoding.
# If the VIF score<5, the columns are moderately correlated.
# So we can ignore the dropping of correlated columns.

# ### Checking the importance of columns in the dataset

# In[10]:


# Important feature using ExtraTreesRegressor
selection = ExtraTreesRegressor()
selection.fit(X, y)

print(selection.feature_importances_)


# #### Visualizing top 5 important columns in the dataset

# In[11]:


#plot graph of feature importances for better visualization
plt.figure(figsize = (8,4))
feat_importances = pd.Series(selection.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.xlabel('Importance ---->')
plt.ylabel('Column Names ---->')
plt.show()


# ### Creating Pipeline for the models

# In[12]:


# Creating Pipeline
# dt1 and dt2 represents model with different hyperparameters
pipeline = Pipeline([
    ('lr', LinearRegression()),
    ('dt1', DecisionTreeRegressor()),
    ('dt2', DecisionTreeRegressor()),
    ('rf', RandomForestRegressor())
])
pipeline


# #### Assigning models to variables

# In[13]:


lr = LinearRegression()
dt1 = DecisionTreeRegressor()
dt2 = DecisionTreeRegressor()
rf = RandomForestRegressor()


# ## Fitting the models

# ### Fitting Linear Regression Model

# In[14]:


lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)


# In[15]:


y_pred_lr


# ### Fitting Decision Tree Model

# ##### Using cross validation and grid search on Decision Tree

# In[16]:


cv = KFold(n_splits=10, shuffle=True, random_state=42)


# In[17]:


# Using Grid Search and fitting the model
dt1 = GridSearchCV(dt1, param_grid=param_grid, cv=cv, scoring='neg_mean_squared_error')
dt1.fit(X_train,y_train)


# #### Printing the best parameters for decision tree based on provided parameters

# In[18]:


print(f"Best hyperparameters: {dt1.best_params_}")
print(f"Best cross-validation score: {-dt1.best_score_:.2f}")


# #### Applying the best parameters on Decision Tree

# In[19]:


#Fitting the model
dt1 = DecisionTreeRegressor(max_depth=None, min_samples_split=10, min_samples_leaf= 2)
dt1.fit(X_train, y_train)

y_pred_dt = dt1.predict(X_test)
y_pred_dt


# #### Comparing with the other Decision Tree not tuned with parameters

# In[20]:


dt2 = DecisionTreeRegressor(max_depth=5, min_samples_split=10)
dt2.fit(X_train, y_train)

y_pred = dt2.predict(X_test)
y_pred


# ### Fitting Random Tree Model

# In[21]:


#Random forest model with tuned hyperparameters
rf = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=75)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_pred_rf


# ### Comparing the Results of 4 Models 

# ##### Defining a function for giving the results

# In[22]:


def evaluate_pipeline(pipeline, y_pred_all, model_name):
    score = round((pipeline.score(X_test, y_test)*100), 2)
    mse = round(mean_squared_error(y_test, y_pred_all), 3)
    rmse = round(np.sqrt(mean_squared_error(y_test, y_pred_all)), 3)
    r2 = round(r2_score(y_test, y_pred_all), 3)
    mae = mean_absolute_error(y_test, y_pred_all)

    metrics = pd.DataFrame({
        'Model Name': model_name,
        'Accuracy(%)': [score],
        'Mean Squared Error (MSE)': [mse],
        'Root Mean Squared Error (RMSE)': [rmse],
        'R-squared (R2-Score)': [r2],
        'Mean Absolute Error (MAE)': [mae]
    })
    return metrics


# In[23]:


r1 = evaluate_pipeline(lr, y_pred_lr, 'Linear Regression')
r2 = evaluate_pipeline(dt1,y_pred_dt, 'Decision Tree - Best Tuned')
r3 = evaluate_pipeline(dt2,y_pred, 'Decision Tree - Basic Tuned')
r4 = evaluate_pipeline(rf, y_pred_rf, 'Random Forest')


all_results = pd.concat([r1, r2, r3, r4], ignore_index=True)
all_results


# In[24]:


def save_plot_as_png(plot, title):
    # Replace spaces in the title with underscores and add the file extension
    filename = title.replace(' ', '_') + '.png'
    
    # Save the plot as a PNG file
    plot.savefig(filename, dpi=600, bbox_inches='tight')
    
    print(f"Plot saved as {filename}")


# ### Plotting the Comparison Scatter Plots

# In[25]:


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

# Plot the first scatter plot in the left subplot
ax1.scatter(y_test, y_pred_lr, s=2)
ax1.set_title('Linear Regression Results')
ax1.set_xlabel('True Values')
ax1.set_ylabel('Predicted Values')
ax1.set_ylim(0,125000)
ax1.set_xlim(0,125000)

# Plot the second scatter plot in the middle subplot
ax2.scatter(y_test, y_pred_dt, s=2)
ax2.set_title('Decision Tree Regression Results')
ax2.set_xlabel('True Values')
ax2.set_ylabel('Predicted Values')
ax2.set_ylim(0,125000)
ax2.set_xlim(0,125000)

# Plot the third scatter plot in the right subplot
ax3.scatter(y_test, y_pred_rf, s=2)
ax3.set_title('Random Forest Regression Results')
ax3.set_xlabel('True Values')
ax3.set_ylabel('Predicted Values')
ax3.set_ylim(0,125000)
ax3.set_xlim(0,125000)

for ax in (ax1, ax2, ax3):
    x = np.linspace(*plt.xlim())
    ax.plot(x, x, transform=ax.transAxes, ls='--', c='red')

# Adjust the spacing between the subplots
fig.subplots_adjust(wspace=0.4)

plt.suptitle("Comparison Scatter Plots of Different Regression Results", fontsize=20)

plt.show()
save_plot_as_png(fig, 'Comparison Scatter Plots of Different Regression Results')


# ### Plotting Cross-Validation Results in Box Plot for all models

# In[26]:


cv_results_lr = cross_val_score(lr, X, y, cv=5)
cv_results_dt = cross_val_score(dt1, X, y, cv=5)
cv_results_rf = cross_val_score(rf, X, y, cv=5)

# Plot the cross-validation results using boxplots
fig, ax = plt.subplots(1,3, figsize=(15, 6))
fig.subplots_adjust(wspace=0.4)

# Plot the first box plot in the left subplot
ax[0].boxplot(cv_results_lr, notch=True)
ax[0].set_title('Cross-validation -> Linear Regression')
ax[0].set_ylabel('Accuracy ---->')

# Plot the second box plot in the middle subplot
ax[1].boxplot(cv_results_dt, notch=True)
ax[1].set_title('Cross-validation -> Decision Tree')
ax[1].set_ylabel('Accuracy ---->')

# Plot the third box plot in the right subplot
ax[2].boxplot(cv_results_rf, notch=True)
ax[2].set_title('Cross-validation -> Random Forest')
ax[2].set_ylabel('Accuracy ---->')

plt.suptitle('Cross-validation Results', fontsize=18)

plt.show()
save_plot_as_png(fig, 'Cross-validation Results Box Plots')


# In[ ]:





# ## Deep Learning

# #### Doing test_train_split again for fresh data

# In[42]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3, random_state=0)


# ### test_train_split using Standard Scalar

# In[43]:


# Changing the dimension of y for scalar transformation
y = np.ravel(y).astype('float32').reshape(-1,1)
y = y.reshape(-1,1)


# In[45]:


#scaling the data before running the model
scaler_x = StandardScaler()
X_train = scaler_x.fit_transform(X_train)
X_test = scaler_x.transform(X_test)

scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)


# #### Shape of the Test Train Split

# In[46]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# ### Defining Deep Learning Model

# In[47]:


ANN_model = keras.Sequential()
ANN_model.add(Dense(100, input_dim = 35))
ANN_model.add(Activation('relu'))
ANN_model.add(Dense(150))
ANN_model.add(Activation('relu'))
ANN_model.add(Dropout(0.25))
ANN_model.add(Dense(150))
ANN_model.add(Activation('relu'))
ANN_model.add(Dense(150))
ANN_model.add(Activation('linear'))
ANN_model.add(Dense(1))         
ANN_model.compile(loss = 'mse', optimizer = 'adam')
ANN_model.summary()


# ### Visualizing the Deep Learning Model

# In[48]:


#install the visualkeras for visualization
#pip install visualkeras


# In[49]:


import visualkeras
visualkeras.layered_view(ANN_model, legend=True)


# ### Compiling the ANN model for Standard Scalar

# In[50]:


ANN_model.compile(optimizer = 'Adam', loss = 'mean_squared_error')


# In[51]:


epochs_hist = ANN_model.fit(X_train, y_train, epochs = 20, batch_size = 20, validation_split = 0.2)


# #### Getting Results from the model

# In[52]:


result = ANN_model.evaluate(X_test, y_test)
accuracy_ANN = 1 - result


# In[53]:


print(f"Accuracy: {round((accuracy_ANN)*100, 3)}%")


# ### test_train_split using MinMax Scalar

# In[54]:


#scaling the data before running the model

minmax_x = MinMaxScaler()
X_train_m = minmax_x.fit_transform(X_train)
X_test_m = minmax_x.transform(X_test)

minmax_y = MinMaxScaler()
y_train_m = minmax_y.fit_transform(y_train)
y_test_m = minmax_y.transform(y_test)


# #### Shape of the Test Train Split

# In[55]:


print(X_train_m.shape)
print(X_test_m.shape)
print(y_train_m.shape)
print(y_test_m.shape)


# ### Compiling the ANN model for MinMax Scalar

# In[56]:


epochs_hist_m = ANN_model.fit(X_train_m, y_train_m, epochs = 20, batch_size = 20, validation_split = 0.2)


# #### Getting Results from the model

# In[57]:


result_m = ANN_model.evaluate(X_test_m, y_test_m)
accuracy_ANN_m = 1 - result_m


# In[58]:


print(f"Accuracy: {round((accuracy_ANN_m)*100, 3)}%")


# ### Comparing the accuracy from Standard Scaler model and MinMaxScaler Model

# In[59]:


print(f"Accuracy (StandardScaler) : {round((accuracy_ANN)*100, 3)}%")
print(f"Accuracy (MinMaxScaler)   : {round((accuracy_ANN_m)*100, 3)}%")


# #### Plotting the Loss Progress During Different Training

# In[60]:


# create figure and axes objects for subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# plot data on the first subplot
ax1.plot(epochs_hist.history['loss'])
ax1.plot(epochs_hist.history['val_loss'])
ax1.set_title('Standard Scalar Training', fontsize=14)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training & Valid Loss')
ax1.set_xlim(0,20)
ax1.set_ylim(0,0.06)
ax1.legend(['Training Loss','Valid Loss'])

# plot data on the second subplot
ax2.plot(epochs_hist_m.history['loss'])
ax2.plot(epochs_hist_m.history['val_loss'])
ax2.set_title('MinMax Scalar Training', fontsize=14)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Training & Valid Loss')
ax2.set_xlim(0,20)
ax2.set_ylim(0,0.06)
ax2.legend(['Training Loss','Valid Loss'])

# display the plot
plt.suptitle('Loss Progress During Different Training', fontsize=18)
plt.legend(['Training Loss','Valid Loss'])
plt.show()

save_plot_as_png(fig,'Loss Progress During Different Training')


# #### Prediction Plot from the MinMaxScalar ANN Model

# In[61]:


# Predicting the model
y_predict_m = ANN_model.predict(X_test_m)

#Plotting the model
plt.plot(figsize=(15, 8))
plt.plot(y_test_m, y_predict_m, "o", color = 'xkcd:sky blue', alpha = 0.1)
x = np.linspace(*plt.xlim())
plt.plot(x, x, color='red', linestyle='--')
plt.xlabel('Model Predictions')
plt.ylabel('True Values')
plt.title('True Value v/s Predicted Value (Scaled)')


# In[ ]:





# ### Results

# In[62]:


# Inversing the scaled data
y_predict_orig_m = minmax_y.inverse_transform(y_predict_m)
y_test_orig_m = minmax_y.inverse_transform(y_test_m)


# In[63]:


# Results of Deep Learning Model
score = round((accuracy_ANN_m)*100, 3)
mse = round(mean_squared_error(y_test_orig_m, y_predict_orig_m), 3)
rmse = round(np.sqrt(mean_squared_error(y_test_orig_m, y_predict_orig_m)), 3)
r2 = round(r2_score(y_test_orig_m, y_predict_orig_m), 3)
mae = mean_absolute_error(y_test_orig_m, y_predict_orig_m)

#Getting results into Dataframe
result = pd.DataFrame({
        'Model Name': 'Artificial Neural Network',
        'Accuracy(%)': [score],
        'Mean Squared Error (MSE)': [mse],
        'Root Mean Squared Error (RMSE)': [rmse],
        'R-squared (R2-Score)': [r2],
        'Mean Absolute Error (MAE)': [mae]
    })


# In[64]:


#Appending the results to all_results
all_results = pd.concat([all_results, result], ignore_index=True)
all_results


# In[ ]:





# In[ ]:





# In[ ]:




