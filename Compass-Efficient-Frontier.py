#%%
#importing libraries
import numpy as np
import pandas as pd
import pandas_datareader as pdr
import scipy.stats
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
import yfinance as yf
import random
from scipy.optimize import minimize
#%%
#Start and end Date
end =datetime.today()
start = end - timedelta(days = 260)#change days
tickers = ['NVDA','TSLA','AMD','RIVN','AAPL','NIO','INTC','PLTR','SIRI','BBD','F','SOFI','LCID','AAL','AMZN','MARA','T','KVUE','SNAP','HBAN','PFE','MU','GME','CCL','VALE']#Change Tickers

#%%
# Get data from excel File
def get_excel_data():
    portfolio = pd.read_excel('/Users/kaloyanpanov/Documents/bocconi/Quant/CODE/TSLA.xlsx', sheet_name='TSLA')

    # change path/put Excel file in same folder and just put the name in the first ''

    return portfolio

#%%
# Download data from Yahoo finance
def get_web_data(tickers):
    portfolio = pd.DataFrame()
    for stock in tickers:
        stock = yf.download(stock, start, end)
        stock = stock['Adj Close']
        portfolio = pd.concat([portfolio, stock], axis=1)

    return portfolio
#%%

# Calculate log daily returns of excel file
def get_excel_log_returns(portfolio):
    portfolio = portfolio.iloc[:, 1:]
    portfolio_log = np.log(portfolio / portfolio.shift(1))

    return portfolio_log.dropna()
#%%
# Calculate log daily returns of web downloaded data
def get_web_log_returns(portfolio):
    portfolio_log = np.log(portfolio / portfolio.shift(1))

    return portfolio_log.dropna()
#%%
#Initialize the portfolio data set
#Change functions depending on data
portfolio = get_web_data(tickers)
portfolio_log = get_web_log_returns(portfolio)
#%%
#Calculate covariance
portfolio_cov = portfolio_log.cov()
#%%
#Maybe a cov matrix
def ret_portfolio_cov(equity_cov,bonds_cov):
    equity_cov = np.array(equity_cov)
    bonds_cov = np.array(bonds_cov)
    portfolio_cov = block_diag(equity_cov,bonds_cov)
    portfolio_cov=pd.DataFrame(portfolio_cov)
    return portfolio_cov
#%%
#Find number of assets
#For excel data add -1
N = len(portfolio.columns)
#%%

# Find random weights with sum 1
def gen_weights(N):
    weights = np.random.random(N)
    weights = weights / sum(weights)

    return weights

#%%
# Find random weights with constraints
def generate_const_weights(N):
    initial_floats = np.random.uniform(0.00, 0.10, N)

    def objective(x):
        return np.sum((x - initial_floats) ** 2)

    # Countraint big positions with 40%
    def constraint_sum_range(x):
        return 0.40 - np.sum(x[(x >= 0.05)])

    # Contraint so no leverage
    def constraint_sum_at_most_one(x):
        return 1 - np.sum(x)

    # Constraint so at least 85% of capital is allocated
    def constraint_sum_at_least(x):
        return np.sum(x) - 0.85

    # Bounds for random weights
    bounds = [(0.00, 0.10) for _ in range(N)]

    constraints = [
        {'type': 'ineq', 'fun': constraint_sum_range},
        {'type': 'ineq', 'fun': constraint_sum_at_most_one},
        {'type': 'ineq', 'fun': constraint_sum_at_least}
    ]

    result = minimize(objective, initial_floats, constraints=constraints, bounds=bounds, options={'disp': False})

    floats = result.x

    return floats
#%%

# Calculates return of the portfolio
def calc_returns(portfolio_log_return, weights):
    portfolio_cuml_ret = np.sum(portfolio_log_return.mean() * weights) * 252

    return portfolio_cuml_ret
#%%

# Calculates volatility of the portfolio
def calc_vol(portfolio_cov, weights):
    annual_cov = np.dot(portfolio_cov * 252, weights)
    vol = np.dot(weights.transpose(), annual_cov)

    return np.sqrt(vol)
#%%

# Monte Carlo Simulation
# Change weight function depending on the portfolio
mc_return = []
mc_weights = []
mc_vol = []
for sim in range(100000):
    weights = gen_weights(N)
    mc_weights.append(weights)
    returns = calc_returns(portfolio_log, weights)
    mc_return.append(returns)
    vol = calc_vol(portfolio_cov, weights)
    mc_vol.append(vol)

#%%
#Calculates the Sharpe ration for each simulation
mc_sharpe = np.array(mc_return)/np.array(mc_vol)
#%%
#Return the position of the highest Sharpe ratio and the value
print(mc_sharpe.argmax(),mc_sharpe.max())
#%%
#Portfolio return for the highest Sharpe ratio
print(mc_return[82420])
#%%
#Weights of the best portfolio
print(mc_weights[82420])
#%%
#Plot the simulation results
plt.figure(dpi=200,figsize=(10,5))
plt.scatter(mc_vol,mc_return,c=mc_sharpe)
plt.ylabel('EXPECTDE RETS')
plt.xlabel('EXPECTED VOL')
plt.colorbar(label="SHARPE RATIO")
plt.show()