# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date: 

### AIM:
To implement the SARIMA model using Python on the website visitors dataset.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA model parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

data = pd.read_csv('/content/daily_website_visitors.csv')
data['Date'] = pd.to_datetime(data['Date'])
data['Page.Loads'] = data['Page.Loads'].str.replace(',', '').astype(int)

plt.plot(data['Date'], data['Page.Loads'])
plt.xlabel('Date')
plt.ylabel('Page Loads')
plt.title('Website Visitor Time Series')
plt.show()

def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))

check_stationarity(data['Page.Loads'])

plot_acf(data['Page.Loads'])
plt.show()
plot_pacf(data['Page.Loads'])
plt.show()

sarima_model = SARIMAX(data['Page.Loads'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()

train_size = int(len(data) * 0.8)
train, test = data['Page.Loads'][:train_size], data['Page.Loads'][train_size:]

sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()
predictions = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('RMSE:', rmse)

plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Page Loads')
plt.title('SARIMA Model Predictions')
plt.legend()
plt.show()
```
### OUTPUT:
![image](https://github.com/user-attachments/assets/b81091c6-e894-4463-aba1-2693a2a2ba5a)

![image](https://github.com/user-attachments/assets/6529d7fd-9e90-4703-99b9-c68cf7acf7de)

![image](https://github.com/user-attachments/assets/917de829-ea0f-43b6-a784-a48a719074f8)

![image](https://github.com/user-attachments/assets/9c9938a4-5b1e-44ee-a9e0-b4723bae0213)

![image](https://github.com/user-attachments/assets/cd277c2a-cefa-4f27-942a-06559e5ec2b5)

![image](https://github.com/user-attachments/assets/ad9d75a9-7754-4a47-acda-f363be9a880d)

### RESULT:
Thus, the program runs successfully based on the SARIMA model on the website visitors dataset.
