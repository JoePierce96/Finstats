import MCclass as mc
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pandas_datareader import data as wb
ibm_data = pd.read_csv('IBMtimeseries.csv', dtype=str)

def import_stock_data(tickers, start = '2010-1-1'):
    """
    stock data import helper function
    imports n stock data
    """
    stock_data = pd.DataFrame()

    # returns a one column DF if only one ticker parsed elsewise returns a DF of n cols
    if len([tickers]) == 1:
        stock_data[tickers] = wb.DataReader(tickers, data_source='yahoo', start = start)['Adj Close']
        stock_data = pd.DataFrame(data)
    else:
        for ticker in tickers:
            stock_data[ticker] = wb.DataReader(ticker, data_source='yahoo', start = start)['Adj Close']

    return stock_data
     
def deannualise(annual_rate, periods=365):
    """
    Helper function to de-annualise yearly rates
    (May move into future bond class)
    """
    return (1 + annual_rate) ** (1/periods) - 1

def get_risk_free_rate():
    """
    Calcuates risk ree rate for 10 year bond
    """
    yesterday_date = datetime.today() - timedelta(days=1)
    annualised = import_stock_data(["TNX"], start = str(yesterday_date))

    daily = annualised.apply(deannualise)






def main():
    """
    Driver code
    """
    ticks = ["IBM"]
    monte = mc.MonteCarlo(ticks)
    monte.RunSims()
    

main()