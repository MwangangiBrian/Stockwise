import os
import pickle
import requests
import datetime as dt
import pandas_datareader.data as web
from bs4 import BeautifulSoup

def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text.strip()
        tickers.append(ticker)
    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)
    return tickers

def get_data_from_yahoo(reload_sp500=False):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)
    
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    start = dt.datetime(2014, 1, 1)
    end = dt.datetime.now()
    
    for ticker in tickers:
        print(f"Fetching data for {ticker}")
        if not os.path.exists(f'stock_dfs/{ticker}.csv'):
            try:
                df = web.DataReader(ticker, 'yahoo', start, end)
                df.reset_index(inplace=True)
                df.set_index("Date", inplace=True)
                df.to_csv(f'stock_dfs/{ticker}.csv')
            except Exception as e:
                print(f"Could not retrieve data for {ticker}: {e}")
        else:
            print(f'Already have {ticker}')

# Fetch and save S&P 500 tickers
# save_sp500_tickers()

# Get stock data from Yahoo Finance
# get_data_from_yahoo()
