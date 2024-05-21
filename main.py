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
df = yf.download('AAPL', start='2012-01-01', end='2024-05-05')

# Verileri bir Excel dosyasına yazdırma
df.to_excel("AAPL_stock_data.xlsx", index=True)

# Kapanış fiyatlarını kullanacağız
data = df['Close'].values
data = data.reshape(-1, 1)

# Veriyi normalize edelim
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Eğitim ve test setlerini oluşturmak için veriyi bölüyoruz
train_data, test_data = scaled_data[:int(len(scaled_data)*0.8)], scaled_data[int(len(scaled_data)*0.8):]

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

# Modeli tanımlama
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
train_score = math.sqrt(mean_squared_error(scaler.inverse_transform(y_train.reshape(-1, 1)), train_predict))
test_score = math.sqrt(mean_squared_error(scaler.inverse_transform(y_test.reshape(-1, 1)), test_predict))

print(f'Train Score: {train_score:.2f} RMSE')
print(f'Test Score: {test_score:.2f} RMSE')

# Tahminleri ve gerçek verileri görselleştirme
plt.figure(figsize=(12,6))
plt.plot(scaler.inverse_transform(scaled_data), label='Gerçek Veriler')
plt.plot(np.arange(time_step, len(train_predict) + time_step), train_predict, label='Eğitim Tahminleri')
plt.plot(np.arange(len(train_predict) + (2 * time_step), len(train_predict) + (2 * time_step) + len(test_predict)), test_predict, label='Test Tahminleri')
plt.legend()
plt.show()

# Gelecek periyotlar için tahmin yapma
def predict_future(model, last_data, future_steps):
    predicted = []
    input_data = last_data[-time_step:].reshape(1, time_step, 1)
    for _ in range(future_steps):
        prediction = model.predict(input_data)
        predicted.append(prediction[0, 0])
        input_data = np.append(input_data[:, 1:, :], prediction.reshape(1, 1, 1), axis=1)
    return np.array(predicted).reshape(-1, 1)

# Son 60 günün verilerini kullanarak tahmin yapalım
last_data = scaled_data[-time_step:]

# Yarın, 1 hafta (7 gün) ve 1 ay (30 gün) sonrasını tahmin etme
future_days = [1, 7, 30]
future_predictions = {}

for days in future_days:
    future_predictions[days] = scaler.inverse_transform(predict_future(model, last_data, days))

# Tahmin sonuçlarını yazdırma
for days, prediction in future_predictions.items():
    print(f"{days} gün sonrası tahmini: {prediction[-1][0]:.2f} USD")

# Tahmin sonuçlarını görselleştirme
plt.figure(figsize=(12,6))
plt.plot(scaler.inverse_transform(scaled_data), label='Gerçek Veriler')
for days, prediction in future_predictions.items():
    future_dates = np.arange(len(scaled_data), len(scaled_data) + days)
    plt.plot(future_dates, prediction, label=f'{days} gün sonrası tahmini')
plt.legend()
plt.show()