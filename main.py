import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
import matplotlib.pyplot as plt

def search(stockName):
    try:
        Image = open('static\\stocks\\'+stockName+'\\'+stockName+'1.png', 'r')
        print("cache found")
        return 1
    except FileNotFoundError:
        print("file not found")
        return 0

# Function to process the data into slices with look-back period lb
def processData(data, lb, forecast_horizon):
    X, Y = [], []
    for i in range(len(data) - lb - forecast_horizon + 1):
        X.append(data[i:(i + lb), 0])
        Y.append(data[(i + lb):(i + lb + forecast_horizon), 0])
    return np.array(X), np.array(Y)

# Main Stock Prediction Function
def stockpredict(stockName):
    data = pd.read_csv('dataset\\all_stocks_5yr.csv')
    cl = data[data['Name']==stockName].close
    path = os.getcwd()
    os.makedirs(path + "\\static\\stocks\\" + stockName, exist_ok=True)

    # Scaling using MinMaxScaler
    scl = MinMaxScaler()
    cl = cl.values.reshape(cl.shape[0], 1)
    cl = scl.fit_transform(cl)

    # Adjust look-back period and forecast horizon
    look_back = 7
    forecast_horizon = 3
    X, Y = processData(cl, look_back, forecast_horizon)
    
    # Split data into training and testing sets
    split_ratio = 0.80
    split_index = int(X.shape[0] * split_ratio)
    X_train, X_test = X[:split_index], X[split_index:]
    Y_train, Y_test = Y[:split_index], Y[split_index:]

    # Building the RNN LSTM model
    model = Sequential()
    model.add(LSTM(32, input_shape=(look_back, 1)))
    model.add(Dropout(0.5))
    model.add(Dense(forecast_horizon))
    model.compile(optimizer='adam', loss='mse')

    # Reshape data for (Sample, Timestep, Features)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Train the model
    hist = model.fit(X_train, Y_train, epochs=50, validation_data=(X_test, Y_test), shuffle=False)

    # Plot and save training loss
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.legend(['Loss', 'Validation Loss'], loc='upper right')
    plt.savefig('static\\stocks\\' + stockName + '\\' + stockName + '2.png')
    plt.clf()
    plt.close()

    # Make predictions for the test set
    Xt = model.predict(X_test)
    Yt = scl.inverse_transform(Y_test)
    Yp = scl.inverse_transform(Xt)

    # Plot and save the results
    for i in range(forecast_horizon):
        plt.plot(Yt[:, i])
        plt.plot(Yp[:, i])
    plt.ylabel('Price')
    plt.xlabel('Days')
    plt.legend(['Real', 'Prediction'], loc='upper left')
    plt.savefig('static\\stocks\\' + stockName + '\\' + stockName + '1.png')
    plt.clf()
    plt.close()

    # Save the predicted prices and accuracy
    predicted_prices = scl.inverse_transform(model.predict(X_test))
    predicted_prices = predicted_prices[-1].tolist()  # Get the last prediction
    predicted_prices = [round(float(price), 2) for price in predicted_prices]

    errors = np.abs((Yt - Yp) / Yt) * 100
    mean_error = np.mean(errors)
    accuracy = 100 - mean_error

    filepathtosave = path + "\\static\\stocks\\" + stockName + "\\" + stockName + ".txt"
    with open(filepathtosave, 'w') as filehandle:
        for item in predicted_prices:
            filehandle.write(f'{item}\n')
        filehandle.write(f'Accuracy: {round(accuracy, 2)}%\n')

    return predicted_prices, round(accuracy, 2)
