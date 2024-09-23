from flask import Flask, render_template, request, redirect, url_for
from main import search, stockpredict, load_and_predict
import os
import yfinance as yf
import numpy as np
import pandas as pd

app = Flask(__name__)
app.debug = True
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route("/")
def index():
    return render_template("home.html")

@app.route('/news')
def news():
    return render_template('news.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/', methods=['POST'])
def requestStock():
    text = request.form['sname']
    stockName = text.upper()
    print(stockName)
    if search(stockName):
        return displayStock(stockName)
    else:
        return predictStock(stockName)

def displayStock(stockName):
    stockData = []
    try:
        with open(f'static\\stocks\\{stockName}\\{stockName}.txt', 'r') as filehandle:
            for line in filehandle:
                currentPlace = line.strip()
                stockData.append(currentPlace)
    except FileNotFoundError:
        print("Stock data file not found.")
    return render_template("stockdetail.html", stockName=stockName, stockData=stockData)

def predictStock(stockName):
    stockData, accuracy = stockpredict(stockName)
    return render_template("stockdetail.html", stockName=stockName, stockData=stockData)

# Route for fetching data from yfinance and predicting
@app.route('/predict_yfinance', methods=['POST'])
def predict_yfinance():
    stockName = request.form['sname'].upper()
    try:
        # Fetch data from yfinance for the last 30 days
        stock_data = yf.Ticker(stockName)
        hist = stock_data.history(period="30d")  # Get last 30 days
        if 'Close' not in hist.columns:
            return "Invalid data: 'Close' column missing."

        # Ensure data contains at least 30 days of closing prices
        if len(hist['Close']) > 30:
            return "Not enough data. Please choose a stock with more historical data."

        # Prepare the data for prediction
        new_data = hist['Close'].values.reshape(-1, 1)
        predictions = load_and_predict(stockName, new_data)
        predictions = predictions[-1].tolist()  # Use the last set of predictions
        predictions = [round(float(price), 2) for price in predictions]

        return render_template("prediction.html", stockName=stockName, predictions=predictions)
    except Exception as e:
        print(f"Error: {e}")
        return "Error fetching data from yfinance."

if __name__ == "__main__":
    app.run()
