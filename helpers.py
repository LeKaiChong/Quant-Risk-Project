import yfinance as yf
from datetime import date
import matplotlib.pyplot as plt
import numpy as np 
import yaml

class ConfigLoader():
    @staticmethod
    def configload(filepath="config.yaml"):
        with open(filepath, "r") as f:
            config = yaml.safe_load(f)
        return config

class data_extraction():
    @staticmethod
    def get_data(stocklist:list, startdate: date, enddate: date):
        stockData = yf.download(stocklist, start=startdate, end=enddate, auto_adjust=True)
        stockData = stockData['Close']
        returns = stockData.pct_change()
        meanReturns = returns.mean()
        covMatrix = returns.cov()
        return meanReturns, covMatrix
    
class montecarlosims():
    
    def monte_carlo(timeframe:int,covmatrix:np.ndarray,number_of_sims:int,meanreturn:object,weightofport:np.ndarray,initialportfoliovalue:int ):
        portfolios_sims = np.full(shape=(timeframe, number_of_sims), fill_value=0.0)
        meanreturn_matrix= np.full(shape=(timeframe, len(weightofport)), fill_value=meanreturn)
        meanreturn_matrix=meanreturn_matrix.T
        for m in range(number_of_sims):
        # MC Loops
            Z = np.random.normal(size=(timeframe, len(weightofport)))
            L = np.linalg.cholesky(covmatrix)# This translates independant shocks into correlated shocks
            dailyreturns = meanreturn_matrix + np.inner(L,Z) # compute the inner product of random shock L and correlated shock Z
            portfolios_sims[:,m] = np.cumprod(np.inner(weightofport, dailyreturns.T)+1)*initialportfoliovalue # cumulative increase in return -> convert to portfolio value 

        #Graph sims
        plt.plot(portfolios_sims)
        plt.ylabel('Portfolio value ($)')
        plt.xlabel('Date')
        plt.title(' MC simulation of a stock portfolio')
        plt.show()

        #mean return
        paths = portfolios_sims              # (T, mc_sims)
        V0 = float(initialportfoliovalue)
        terminal = paths[-1, :]             # (mc_sims,)
        exp_terminal = terminal.mean()
        print("Expected return:", exp_terminal)

        #medium return
        median_terminal = np.median(terminal)
        print("Median return:", median_terminal)

        # prob that VT < V0
        loss_prob = np.mean(terminal < V0)
        print("Loss probability:", loss_prob)


