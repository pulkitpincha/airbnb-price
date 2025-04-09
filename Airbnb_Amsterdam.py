# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 15:00:19 2023

@author: stimp
"""

#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score
from mlxtend.plotting import plot_learning_curves
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization    
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import warnings;warnings.simplefilter(action='ignore', category=FutureWarning)
import warnings

#loading the datasets
amsterdam_weekdays = pd.read_csv("Datasets/amsterdam_weekdays.csv")
amsterdam_weekends = pd.read_csv("Datasets/amsterdam_weekends.csv")

#shape of datasets
print(f"Shape of amsterdam_weekdays: {amsterdam_weekdays.shape}")
print(f"Shape of amsterdam_weekends: {amsterdam_weekends.shape}")
print(amsterdam_weekdays.columns)
print(amsterdam_weekends.columns)

#Combining the two datasets into one
def combine(csv_1,col_1,csv_2,col_2,city):       
    csv_1['week time'] = col_1
    csv_2['week time'] = col_2
    csv_1.drop(columns = ['Unnamed: 0'],inplace=True)
    merged = pd.concat([csv_1, csv_2])
    return merged

amsterdam = combine(amsterdam_weekdays,'weekdays',amsterdam_weekends,'weekends','amsterdam')

amsterdam.head()
amsterdam.tail()
amsterdam.sample(5)
amsterdam.isna().sum()

#plotting the realSum
fig2, axs = plt.subplots(nrows=2, ncols=1, figsize=(13, 5))

sns.boxplot(y='realSum', data=amsterdam, ax=axs[0])
axs[0].set_yticks(np.arange(0,max(amsterdam['realSum']),1500))
axs[0].set_xlabel('Amsterdam')
axs[0].set_ylabel('realSum')

axs[1].hist(amsterdam['realSum'], bins=20, alpha=0.5, color='000000', density=True)
axs[1].set_xticklabels(np.arange(0,max(amsterdam['realSum']),1500))
axs[1].set_xticks(np.arange(0,max(amsterdam['realSum']),1500))
axs[1].set_xlabel('realSum')
axs[1].set_ylabel('Desnity')

#inter-quartile range of realSum
print('Inter-Quartile Range of realSum : ' + str(amsterdam['realSum'].quantile(0.75) - amsterdam['realSum'].quantile(0.25)))

#removing outliers
amsterdam_cleaned = [amsterdam[amsterdam['realSum'] < 1500]]
amsterdam = pd.concat(amsterdam_cleaned, ignore_index=True)
amsterdam.describe()

#plotting realSum (price) from the cleaned dataset
fig2, axs = plt.subplots(nrows=2, ncols=1, figsize=(13, 5))

sns.boxplot(y='realSum', data=amsterdam, ax=axs[0])
axs[0].set_yticks(np.arange(0,max(amsterdam['realSum']),350))
axs[0].set_xlabel('Price Distribution of Amsterdam')
axs[0].set_ylabel('realSum')

axs[1].hist(amsterdam['realSum'], bins=20, alpha=0.5, color='000000', density=True)
axs[1].set_xticklabels(np.arange(0,max(amsterdam['realSum']),350))
axs[1].set_xticks(np.arange(0,max(amsterdam['realSum']),350))
axs[1].set_xlabel('realSum')
axs[1].set_ylabel('Desnity')

plt.subplots_adjust(hspace=0.50)
plt.show()

#difference between realSum (price) on weekdays and weekends
plt.figure
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 3))
fig2, axs2 = plt.subplots(nrows=1, ncols=1, figsize=(15, 3))
sns.boxplot(y='realSum', data=amsterdam,x='week time',ax = axs[0])
axs[0].tick_params(axis='y', labelsize=15)
axs[0].tick_params(axis='x', labelsize=15)

amsterdam.groupby('week time')['realSum'].plot(kind='hist', alpha=0.15, bins=15,ax=axs[1])

sns.kdeplot(data=amsterdam[amsterdam['week time'] == 'weekdays']['realSum'], label='weekdays',ax=axs2)
sns.kdeplot(data=amsterdam[amsterdam['week time'] == 'weekends']['realSum'], label='weekends',ax=axs2)
plt.subplots_adjust(hspace=0.65)
plt.show()

#creating a list of the numeric features in the dataset
amsterdam_numeric_features = list(amsterdam.select_dtypes(include=['int64','float64']).columns[i] for i in [1,4,5,6,7,8,9,10,11,12,13,14])

#creating a funcion to plot the distribution of each numeric feature
def plotter(feature,color,row):
    sns.histplot(data=amsterdam[feature],ax=axes[row,0],kde=True,color=color,line_kws={'color': 'Yellow'})
    axes[row,0].set_title(str(feature)+" Frequency (HISTPLOT)")
    axes[row,1].boxplot(amsterdam[feature])
    axes[row,1].set_title(str(feature)+" Distribution (BOXPLOT)")
    
plt.figure
fig, axes = plt.subplots(nrows=12, ncols=2, figsize=(15, 50))
for i in range(12):
    plotter(amsterdam_numeric_features[i] , '#000000' , i)

plt.subplots_adjust(hspace=0.50)
plt.show()

##data pre-processing
amsterdam.head()

#replacing true and false values with 1,0
amsterdam.replace({False: 0, True: 1},inplace=True)
amsterdam.head()

#replacing categorical values with dummy variables
print(amsterdam['room_type'].value_counts())
print(amsterdam['week time'].value_counts())

amsterdam_dummy = pd.get_dummies(amsterdam[['room_type','week time']],drop_first=True)
amsterdam = pd.concat([amsterdam_dummy, amsterdam.drop(columns=['room_type','week time'])], axis=1)

#heatmap plot for checking for perfect corelation
plt.figure
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(25, 15))
sns.heatmap(amsterdam.corr(),cmap=sns.color_palette("Paired",20),annot=True,ax=axes)

#removing normalized data for future scaling
amsterdam.drop(columns = ['rest_index_norm','attr_index_norm','room_shared','room_private'],inplace=True)
plt.figure
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(25, 15))
sns.heatmap(amsterdam.corr(),cmap=sns.color_palette("Paired",20),annot=True,ax=axes)

#scaling features
Standard_Scaler = StandardScaler()
amsterdam.shape
amsterdam.columns

features_to_scale = ['person_capacity','cleanliness_rating','guest_satisfaction_overall','bedrooms','dist','metro_dist','attr_index','rest_index']
features_not_to_scale = ['room_type_Private room', 'room_type_Shared room', 'week time_weekends','realSum','host_is_superhost', 'multi', 'biz',]

scaled_features = pd.DataFrame(Standard_Scaler.fit_transform(amsterdam[features_to_scale]), columns=features_to_scale)
scaled_features.head()

amsterdam_final = pd.concat([scaled_features.reset_index(drop=True),  amsterdam[features_not_to_scale].reset_index(drop=True)], axis=1)
amsterdam_final.head()

#splitting data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(amsterdam_final.drop(columns=['realSum']) , amsterdam_final['realSum'], random_state=4, test_size=0.2, stratify=amsterdam_final[['week time_weekends']])
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

MSE = []
R2 = []

##linear regression
LINregressor = LinearRegression()
LINRegmodel = LINregressor.fit(X_train, Y_train)
LINRegpreds = LINRegmodel.predict(X_test)
MSE.append(mean_squared_error(Y_test, LINRegpreds))
print(LINRegmodel.coef_)
print(LINRegmodel.intercept_)
R2.append(LINRegmodel.score(X_train, Y_train))

#learning curve plot
plot_learning_curves(X_train, Y_train, X_test, Y_test, clf=LINregressor, scoring="mean_squared_error", print_model=False)
plt.title("Linear Regression")
plt.show()

#plot for residual error
plt.style.use('fivethirtyeight')
plt.scatter(LINregressor.predict(X_train), #plotting error in train data
            LINregressor.predict(X_train) - Y_train,
            color="green", s=10,
            label='Train data')
plt.scatter(LINregressor.predict(X_test), #plotting error in test data
            LINregressor.predict(X_test) - Y_test,
            color="blue", s=10,
            label='Test data')
plt.hlines(y=0, xmin=0, xmax=1200, linewidth=2, color="black") #zero error line
plt.legend(loc='upper right') #legend
plt.title("Residual errors - Linear Regression")
plt.show()

##ridge regression
RIDGEregressor = Ridge(alpha=100)
RIDGERegmodel = RIDGEregressor.fit(X_train, Y_train)
RIDGERegpreds = RIDGERegmodel.predict(X_test)
MSE.append(mean_squared_error(Y_test, RIDGERegpreds))
print(RIDGERegmodel.coef_)
print(RIDGERegmodel.intercept_)
R2.append(RIDGERegmodel.score(X_train, Y_train))
plot_learning_curves(X_train, Y_train, X_test, Y_test, clf=RIDGEregressor, scoring="mean_squared_error", print_model=False)
plt.title("Ridge Regression")
plt.show()

#plot for residual error
plt.style.use('fivethirtyeight')
plt.scatter(RIDGEregressor.predict(X_train), #plotting error in train data
            RIDGEregressor.predict(X_train) - Y_train,
            color="green", s=10,
            label='Train data')
plt.scatter(RIDGEregressor.predict(X_test), #plotting error in test data
            RIDGEregressor.predict(X_test) - Y_test,
            color="blue", s=10,
            label='Test data')
plt.hlines(y=0, xmin=0, xmax=1200, linewidth=2, color="black") #zero error line
plt.legend(loc='upper right') #legend
plt.title("Residual errors - Ridge Regression")
plt.show()

##lasso regression
LASSOregressor = Lasso(alpha=0.01) # Start with alpha=500, reduce to 50, 10, 3 and see the impact
LASSORegmodel = LASSOregressor.fit(X_train, Y_train)
LASSORegpreds = LASSORegmodel.predict(X_test)
MSE.append(mean_squared_error(Y_test, LASSORegpreds))
print(LASSORegmodel.coef_)
print(LASSORegmodel.intercept_)
R2.append(LASSORegmodel.score(X_train, Y_train))
plot_learning_curves(X_train, Y_train, X_test, Y_test, clf=LASSOregressor, scoring="mean_squared_error", print_model=False)
plt.title("Lasso Regression")
plt.show()

#plot for residual error
plt.style.use('fivethirtyeight')
plt.scatter(LASSOregressor.predict(X_train), #plotting error in train data
            LASSOregressor.predict(X_train) - Y_train,
            color="green", s=10,
            label='Train data')
plt.scatter(LASSOregressor.predict(X_test), #plotting error in test data
            LASSOregressor.predict(X_test) - Y_test,
            color="blue", s=10,
            label='Test data')
plt.hlines(y=0, xmin=0, xmax=1200, linewidth=2, color="black") #zero error line
plt.legend(loc='upper right') #legend
plt.title("Residual errors - Lasso Regression")
plt.show()

##decision tree
DTR = DecisionTreeRegressor()
DTR.fit(X_train,Y_train)
DTR_TrainSet_Prediction = DTR.predict(X_train)
DTR_TestSet_Prediction = DTR.predict(X_test)
DTR_predict_evaulation = {'Decision Tree Regression Predictions Evaluation (All Features)':
    {'Model fit - R_squared score (Train)':r2_score(Y_train, DTR_TrainSet_Prediction),
     'Model fit - R_squared score (Test)':r2_score(Y_test, DTR_TestSet_Prediction),
     'Model pred - Mean squared error (Train)':mean_squared_error(Y_train, DTR_TrainSet_Prediction),
     'Model pred - Mean squared error (Test)':mean_squared_error(Y_test, DTR_TestSet_Prediction)}}
DTR_predict_evaulation = pd.DataFrame(DTR_predict_evaulation)
DTR_predict_evaulation
MSE.append(mean_squared_error(Y_test, DTR_TestSet_Prediction))
R2.append(DTR.score(X_train, Y_train))

###GridSearchCV
##KNN
KNNparamgrid = {'n_neighbors': [2, 4, 6, 8, 10]}
KNNgrid = GridSearchCV(KNeighborsRegressor(), KNNparamgrid, cv=5)
KNNgrid.fit(X_train, Y_train)
bestKNNmodel = KNNgrid.best_estimator_
print("Mean squared error:", mean_squared_error(Y_test, bestKNNmodel.predict(X_test)))
MSE.append(mean_squared_error(Y_test, bestKNNmodel.predict(X_test)))
R2.append(bestKNNmodel.score(X_train, Y_train))

##SVM
SVRmodel = SVR(kernel='linear')
SVRmodel.fit(X_train, Y_train)
SVRpreds = SVRmodel.predict(X_test)
print("Mean squared error:", mean_squared_error(Y_test, SVRpreds))
MSE.append(mean_squared_error(Y_test, SVRpreds))
R2.append(SVRmodel.score(X_train, Y_train))

##Random Forest
#defining hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
}

RFregressor = RandomForestRegressor(random_state=23, oob_score=True)

grid_search = GridSearchCV(estimator=RFregressor, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, Y_train)

#best hyperparameter values
best_params = grid_search.best_params_

print("Best Hyperparameters:", best_params)

#best model with hyperparameters
best_RF_model = grid_search.best_estimator_

RFregpreds_train = best_RF_model.predict(X_train)
RFregpreds_test = best_RF_model.predict(X_test)

#model evaluation
RFR_prediction_evaluation = {
    'Random Forest Regression Fitting and Prediction Evaluation (All Features)': {
        'No. of Features': best_RF_model.n_features_in_,
        'Model fit - R_squared score (Train)': r2_score(Y_train, RFregpreds_train),
        'Model fit - R_squared score (Test)': r2_score(Y_test, RFregpreds_test),
        'Model pred - Mean squared error (Train)': mean_squared_error(Y_train, RFregpreds_train),
        'Model pred - Mean squared error (Test)': mean_squared_error(Y_test, RFregpreds_test)
    }
}
RFR_prediction_evaluation = pd.DataFrame(RFR_prediction_evaluation)
print(RFR_prediction_evaluation)
MSE.append(mean_squared_error(Y_test, RFregpreds_test))
R2.append(best_RF_model.score(X_train, Y_train))

#plot for residual error
plt.style.use('fivethirtyeight')
plt.scatter(best_RF_model.predict(X_train), #plotting error in train data
            best_RF_model.predict(X_train) - Y_train,
            color="green", s=10,
            label='Train data')
plt.scatter(best_RF_model.predict(X_test), #plotting error in test data
            best_RF_model.predict(X_test) - Y_test,
            color="blue", s=10,
            label='Test data')
plt.hlines(y=0, xmin=0, xmax=1200, linewidth=2, color="black") #zero error line
plt.legend(loc='upper right') #legend
plt.title("Residual errors - Random Forest")
plt.show()

    

##artificial neural networks
model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim = X_train.shape[1]))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(units=32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='linear'))

#compiling the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

#training
model.fit(X_train, Y_train, epochs=100, batch_size=16, validation_split=0.2)

#evaluating on test data
y_pred = model.predict(X_test)
mse = mean_squared_error(Y_test, y_pred)
print("Mean Squared Error on Test Set:", mse)
MSE.append(mean_squared_error(Y_test, y_pred))
R2.append(r2_score(Y_test, y_pred))

##output all model evaluations
models = ['Linear Regression', 'Ridge', 'Lasso', 'Decision Tree',
          'K-Nearest Neighbour', 'Support Vector Regression', 'Random Forest', 'Artificial Neural Network']

for item1, item2, item3, in zip(models, R2, MSE):
    print(f'Model: {item1} \nR2: {item2} \nMean Squared Error: {item3} \n')
