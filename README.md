# Ex.No: 6               HOLT WINTERS METHOD
### Date: 20.05.2026



### AIM:

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
data = pd.read_csv('/content/gold_rate_history.csv')

# Convert Date column into datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Set Date as index
data.set_index('Date', inplace=True)

# Keep only required column
data = data[['Standard Gold (22 K)']]

# Convert yearly average
data = data.resample('YE').mean()

# Plot original data
data.plot()

plt.title('Gold Rate Over Years')

plt.xlabel('Year')

plt.ylabel('Gold Rate')

plt.show()

# Scale the data
scaler = MinMaxScaler()

scaled_data = pd.Series(
    scaler.fit_transform(
        data['Standard Gold (22 K)'].values.reshape(-1, 1)
    ).flatten(),
    index=data.index
)

# Plot scaled data
scaled_data.plot()

plt.title('Scaled Gold Rate Data')

plt.xlabel('Year')

plt.ylabel('Scaled Gold Rate')

plt.show()

# Check seasonality
decomposition = seasonal_decompose(
    data['Standard Gold (22 K)'],
    model="additive",
    period=2
)

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
test_predictions_add = model_add.forecast(
    steps=len(test_data)
)

# Plot train, test and predictions
ax = train_data.plot()

test_predictions_add.plot(ax=ax)

test_data.plot(ax=ax)

ax.legend([
    "train_data",
    "test_predictions",
    "test_data"
])

ax.set_title('Visual Evaluation')

plt.xlabel('Year')

plt.ylabel('Gold Rate')

plt.show()

# Evaluate model
print(
    "RMSE:",
    np.sqrt(
        mean_squared_error(
            test_data,
            test_predictions_add
        )
    )
)

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

ax.legend([
    "Gold Rate Data",
    "Future Predictions"
])

ax.set_xlabel('Year')

ax.set_ylabel('Gold Rate')

ax.set_title('Gold Rate Prediction')

plt.show()
```

### OUTPUT:


TEST_PREDICTION



<img width="584" height="455" alt="image" src="https://github.com/user-attachments/assets/6f018374-2d99-4e20-bf4e-7c7565ff4ac6" />



FINAL_PREDICTION


<img width="576" height="455" alt="image" src="https://github.com/user-attachments/assets/2c6d2437-b6dd-49de-8118-7aaf6580dc43" />


### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
