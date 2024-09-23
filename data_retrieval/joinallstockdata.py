import pandas as pd

tickers = pd.read_csv('sp500StockList.csv')
tickers = tickers['Ticker'].to_list()

def addStockName():
    """
    Function to add a 'Name' column to each CSV file in the 'stock_dfs' directory.
    The 'Name' column contains the ticker symbol of the stock.
    """
    
    # Loop through each ticker
    for stock in tickers:
        # Construct the file path for the CSV
        path = f'stock_dfs/{stock}.csv'
        
        try:
            # Read the CSV file into a DataFrame
            df = pd.read_csv(path)
            
            # Add the 'Name' column with the ticker
            df['Name'] = stock
            
            # Optional Print(can be commented out in production)
            # print(df) # Uncomment to print the modified DataFrame
            
            # Save the modified DataFrame back to CSV (overwrites the original file)
            df.to_csv(path, index=False)
            
            # Print a message indicating that the file was processed
            print(f"Processed file: {path}")
            
        except FileNotFoundError:
            print(f"File not found: {path}")
        except pd.errors.EmptyDataError:
            print(f"File is empty: {path}")
        except Exception as e:
            print(f"An error occurred while processing the file {path}: {e}")


def joinAllStockData():
    """
    Function to join all stock data from individual CSV files into one CSV file.
    Each row in the output CSV file will have an additional column 'Name' which
    contains the ticker symbol of the stock.
    """
    
    # Loop through each ticker
    for stock in tickers:
        
        # Construct the file path for the CSV
        path = f'stock_dfs/{stock}.csv'
        try:
            df = pd.read_csv(path)
            # Add the 'Name' column with the ticker
            df['Name'] = stock
            
            # Append the modified DataFrame to the output CSV file
            df.to_csv('C:/Stockwise/data_retrieval/allstocks.csv',mode = 'a', header=False)
            
            # Print a message indicating that the file was processed
            print(f"Processed file: {path}")
            
        except FileNotFoundError:
            print(f"File not found: {path}")
        except pd.errors.EmptyDataError:
            print(f"File is empty: {path}")
        except Exception as e:
            print(f"An error occurred while processing the file {path}: {e}")
            print(f"An error occurred while processing the file {path}: {e}")

# Call the functions (Uncomment to run)
# addStockName()
# joinAllStockData()