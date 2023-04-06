import requests
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# set up API connection and retrieve historical stock data
api_key = 'YOUR_API_KEY'
stock_symbol = 'AAPL' # replace with desired stock symbol
days_back = 365 # number of days to retrieve historical data
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={stock_symbol}&apikey={api_key}&outputsize=full'
response = requests.get(url)
data = json.loads(response.text)

# parse historical stock data into a Pandas DataFrame
df = pd.DataFrame(data['Time Series (Daily)']).T
df.index = pd.to_datetime(df.index)
df = df.sort_index()
df = df[(df.index >= start_date) & (df.index <= end_date)]
df['4. close'] = pd.to_numeric(df['4. close'], errors='coerce')

# calculate daily returns and create lagged variables for regression model
df['returns'] = df['4. close'].pct_change()
df['lag_1'] = df['returns'].shift(1)
df['lag_2'] = df['returns'].shift(2)
df['lag_3'] = df['returns'].shift(3)

# remove NaN values
df.dropna(inplace=True)

# split data into training and testing sets
train = df.iloc[:-30]
test = df.iloc[-30:]

# train linear regression model on training data
X_train = train[['lag_1', 'lag_2', 'lag_3']]
y_train = train['returns']
model = LinearRegression()
model.fit(X_train, y_train)

# make predictions on testing data
X_test = test[['lag_1', 'lag_2', 'lag_3']]
y_test = test['returns']
predictions = model.predict(X_test)

# calculate root mean squared error (RMSE)
mse = np.mean((predictions - y_test) ** 2)
rmse = np.sqrt(mse)

# print RMSE and predicted returns for the next 30 days
print('RMSE:', rmse)
print('Predicted returns for next 30 days:', predictions)
