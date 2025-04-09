# airbnb-price
Dataset: https://www.kaggle.com/datasets/thedevastator/airbnb-prices-in-european-cities?select=amsterdam_weekends.csv

Predicting as well as analyzing variables that affect the prices of Airbnb's in Amsterdam 

# Airbnb Prices in Amsterdam
Importing datasets of weekend as well as weekdays.
Checking the distribution of prices in both to compare the difference.

![image](https://github.com/user-attachments/assets/96554a5c-d0d3-48e0-9875-11d9785fb113)

We can see that there is minimal difference in the ranges of realSum (price) on the weekdays and 
weekends.
Next we combine the datasets into a single dataframe. And check for the inter-quartile range of price 
(354.6) and remove any outliers.

# Plotting the distributions of the numeric features in the dataset:
![image](https://github.com/user-attachments/assets/fab7de21-d79d-4650-b4eb-8c6d5537411c)

We can derive that:
-	People capacity in descending order: 2,4,3,6,5
-	We can see the cleanliness rating is 8 and above, meaning that Airbnb in Amsterdam have a high cleanliness rating on average.
-	We can also see that guest satisfaction is much like cleanliness.
-	Bedroom listings in descending order: 1,2,3,4
-	Most listings are within 7km of the city centre. 
-	Most listings are also within 3km of the metro.

Converting Boolean values to 0,1 and making dummy variables for all the categorical features.

# Checking for corelation and removing normalized and categorical features for future scaling.
![image](https://github.com/user-attachments/assets/6ab77149-3fe5-4bd1-9554-f1e8aaaa33a9)

Scaling the numeric data.

Splitting data into train and test.

# Running many different regression models to see what gives us the best results:
![image](https://github.com/user-attachments/assets/1368f53c-fd4f-46a1-a279-adac2d355724)

# What are we looking for?
- A high r2 score: This shows us the proportion of the variance that is predicted by the model. 0 being the worst and 1 being the best. However, if we have a r2 value of 1 it can also be because of overfitting of the model.
- A low Mean Squared Error Score (MSE): The MSE is exactly as it sounds; it is the squared average of all the variances between the predicted values and the actual values of the test data. A higher value would indicate more error in the model’s prediction.

## Conclusion
- Here we can see that our Decision Tree model has a r2 of 1 which seems to be perfect, however we also have a MSE score of 22202 which indicates that we still have errors in our prediction. This is because of overfitting while training the model.
- We can also see that our worst performing models are the ANN and the SVR models.
- Linear Regression, Ridge, Lasso, and KNN seem to be performing average.
- We can see that our best model is the Random Forest Regressor, with a r2 score of 0.968 and a MSE score of 15947. Although we can see some overfitting in the model, it was made using GridSearch to find the best hyperparameters and to regularize the model further. This model also has the lowest MSE score from all the others and is hence still our best model to predict the prices of Airbnb stays in Amsterdam.

