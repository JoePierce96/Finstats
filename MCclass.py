"""
Created on Sat Nov 25 22:39:05 2023

@author: Joe
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from Finstats import import_stock_data



class MonteCarlo:
    """
    Monte Carlo sim class will use database

    """
    
    def __init__(self, tickers, market, sim_days, sim_runs):
        self.start = '2010-1-1'
        self.stock_data = import_stock_data(tickers, start = self.start)
        self.market_data = import_stock_data(market, start = self.start)
        self.sim_days = sim_days
        self.sim_runs = sim_runs
        self.riskfree_rate = 0.025

    def __str__(self):
        """
        Return human readable string
        """
        market = list(self.stock_data)
        headers = list(self.stock_data.values) 
        result = ""
        result += "tickers: " + str(headers)
        result += "Recent close price: " + str(self.stock_data.head())
        result += "Start date: " + str(self.start)
    
    def log_returns(self, data):
        """
        Calculate logarithmic daily returns for stocks
        """

        return (np.log(1 + data.pct_change()))

    def drift(self):
        """
        Calculates the drift of returns which is the change in expected value over a given time period
        """
        log_rets = self.log_return
        mean_log_rets = log_rets.mean()
        varience = log_rets.var()
        drift = mean_log_rets - (0.5 * varience)

        try:
            return drift.values
        except:
            return drift

    def daily_brownian_returns(self):
        """
        Daily returns of a stock(s) using brownian motion (deep)
        """
        dft = self.drift
        try:
            std = self.log_returns(self.stock_data).std().values
        except:
            std = self.log_returns(self.stock_data).std()

        daily_returns = np.exp(dft + std * norm.ppf(np.random.rand(self.sim_days, self.sim_runs)))
        return daily_returns

    def CAPM_Sharpe(self):
       """
       Compute the beta and risk_adjusted retuns for CAPM and Sharpe
       """

       """Beta"""
       m_rets, m_rets1 = self.market_data
       log_mrets = log_returns(m_rets)
       covariance = log_mrets.cov()* 252  # annualised .cov is an array
       covar_matrix = pd.DataFrame(covariance.iloc[:-1,-1]) # assigns covariance matrix to DF
       market_var = log_mrets.iloc[:,-1].var()*252
       beta = covar_matrix / market_var

       #standard deviation return array
       std_returns = pd.DataFrame(((log_mrets.std()*250**0.5) [:-1]), columns = ['Standard deviation'])
       beta.merge(std_returns, left_index=True)

    def probs_find(self, predicted, higherthan, on = 'value'):
        if on =='return':
            predicted0 = predicted.iloc[0, 0]
            predicted = predicted.iloc[-1]
            predList = list (predicted)
            over = [(i*100)/predicted0 for i in predList if ((i-predicted0)*100)/predicted0 >= higherthan]
            less = [(i*100)/predicted0 for i in predList if ((i-predicted0)*100)/predicted0 < higherthan]
            elif on == 'value': 
                predicted = predicted.iloc[-1]
                predList = list(predicted)
                over = [i for i in predList if i >= higherthan]
                less = [i for i in predList if i < higherthan]
            else:
                print("'on' must be either value or return")
            return (len(over)/(len(over)+len(less)))

    def simulate_mc(self, data, days, iterations, plot=True): 
        returns = daily_returns(data, days, iterations)
        # create empty matrix
        price_list = np.zeros_like(returns)
        # put the last actual price in the first row of matrix 
        price_list[0] = data.iloc[-1]
        # Calculate the price of each day
        for t in range(1, days):
            price_list[t] = price_list[t-1]*returns[t]
        
        if plot == True:
            x = pd.DataFrame(price_list).iloc[-1]
            fig, ax = plt.subplots(1,2, figsize=(14, 4))
            sns.distplot(x, ax=ax[0])
            sns.distplot(x, hist_kws={'cumulative':True}, kde_kws={'cumulartive':True},ax=ax[1])
            plt.xlabel("Stock Price")
            plt.show()

            #CAPM and Sharpe Ratio

            #Printing information about stock
            try:
                [print(nam) for nam in data.columns]
            except:
                print(data.name)
            print(data.name)
            print(f"Days: {days-1}")
            print(f"Expected Value: ${round(100*())}")
            print(f"print")

    


    def RunSims(self):
        for i in range(self.sims):
            rand_rets = np.random.normal(self.mu, self.vol, self.time_period) + 1
            forcasted_values = self.latest_price *(rand_rets).cumprod()
            plt.plot(range(self.time_period), forcasted_values)
        
        plt.show()



    




