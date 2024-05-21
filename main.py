import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
import math
from sklearn.metrics import mean_squared_error
import yfinance as yf

# Örnek olarak Apple'ın (AAPL) hisse senedi verilerini indiriyoruz
df = yf.download('AAPL', start='2010-01-01', end='2022-01-01')

# Kapanış fiyatlarını kullanacağız
data = df['Close'].values
data = data.reshape(-1, 1)

# Veriyi normalize edelim
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Eğitim ve test setlerini oluşturmak için veriyi bölüyoruz
train_data, test_data = train_test_split(scaled_data, test_size=0.2, shuffle=False)

# Veri setini LSTM girdi formatına dönüştürelim
def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        X.append(a)
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 60
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# LSTM girdi formatına dönüştürmek için yeniden şekillendirme
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Modeli eğitme
model.fit(X_train, y_train, batch_size=1, epochs=1)


# Tahmin yapma
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Tahminleri orijinal ölçeğe döndürme
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Modelin performansını değerlendirme


train_score = math.sqrt(mean_squared_error(y_train, train_predict))
test_score = math.sqrt(mean_squared_error(y_test, test_predict))

print(f'Train Score: {train_score:.2f} RMSE')
print(f'Test Score: {test_score:.2f} RMSE')

# Tahminleri ve gerçek verileri görselleştirme
plt.figure(figsize=(12,6))
plt.plot(scaler.inverse_transform(scaled_data), label='Gerçek Veriler')
plt.plot(np.arange(time_step, len(train_predict) + time_step), train_predict, label='Eğitim Tahminleri')
plt.plot(np.arange(len(train_predict) + (2 * time_step), len(train_predict) + (2 * time_step) + len(test_predict)), test_predict, label='Test Tahminleri')
plt.legend()
plt.show()