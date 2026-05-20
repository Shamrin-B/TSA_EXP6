# Ex.No: 6               HOLT WINTERS METHOD
### Date: 16/05/2026



### AIM:
To analyze and forecast the **accuracy trend over years** using the **Holt-Winters Exponential Smoothing method** by performing data preprocessing, scaling, seasonal decomposition, model training, evaluation, and future prediction using Python.

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
```
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose

# Load the dataset and perform data exploration
data = pd.read_excel('/content/advanced_sales_report.xlsx')

# Convert Year column into datetime format
data['Year'] = pd.to_datetime(data['Year'], format='%Y')

# Set Year as index
data.set_index('Year', inplace=True)

# Plot original data
data.plot()
plt.title('Accuracy Over Years')
plt.xlabel('Year')
plt.ylabel('Accuracy')
plt.show()

# Scale the data
scaler = MinMaxScaler()

scaled_data = pd.Series(
    scaler.fit_transform(data['Accuracy'].values.reshape(-1, 1)).flatten(),
    index=data.index
)

# Plot scaled data
scaled_data.plot()

plt.title('Scaled Accuracy Data')
plt.xlabel('Year')
plt.ylabel('Scaled Accuracy')
plt.show()

# Check seasonality
decomposition = seasonal_decompose(data['Accuracy'], model="additive", period=2)

decomposition.plot()
plt.show()

# Split train and test data
scaled_data = scaled_data + 1

train_data = scaled_data[:int(len(scaled_data) * 0.8)]
test_data = scaled_data[int(len(scaled_data) * 0.8):]

# Create Holt-Winters model
model_add = ExponentialSmoothing(
    train_data,
    trend='add',
    seasonal='mul',
    seasonal_periods=2
).fit()

# Predict test data
test_predictions_add = model_add.forecast(steps=len(test_data))

# Plot train, test and predictions
ax = train_data.plot()

test_predictions_add.plot(ax=ax)
test_data.plot(ax=ax)

ax.legend(["train_data", "test_predictions", "test_data"])

ax.set_title('Visual Evaluation')

plt.xlabel('Year')
plt.ylabel('Accuracy')

plt.show()

# Evaluate model
print("RMSE:",
      np.sqrt(mean_squared_error(test_data, test_predictions_add)))

# Create final model
final_model = ExponentialSmoothing(
    scaled_data,
    trend='add',
    seasonal='mul',
    seasonal_periods=2
).fit()

# Predict future data
final_predictions = final_model.forecast(steps=3)

# Plot final predictions
ax = scaled_data.plot()

final_predictions.plot(ax=ax)

ax.legend(["Accuracy Data", "Future Predictions"])

ax.set_xlabel('Year')
ax.set_ylabel('Accuracy')

ax.set_title('Accuracy Prediction')

plt.show()
```
### OUTPUT: 


TEST_PREDICTION

<img width="352" height="266" alt="image" src="https://github.com/user-attachments/assets/a0da8e03-4377-4894-ad3e-8af8c7dc4884" />


FINAL_PREDICTION

<img width="717" height="314" alt="image" src="https://github.com/user-attachments/assets/e13f9dcb-766c-41f4-adb9-c0e6ee8186ce" />


### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
