#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import requests
import pandas as pd
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from MCForecastTools import MCSimulation
import json
import warnings
import hvplot.pandas
import numpy as np
import seaborn as sns
#import scipy.stats as stats

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


get_ipython().system('python -m ensurepip --upgrade')
get_ipython().system('pip3 install streamlit')


# In[2]:


# Load .env enviroment variables
load_dotenv('.env')


# # Part 1 - Litecoin and Chainlink Cryptocurrency Analysis

# In[3]:


# Create a ticker for Litecoin and Chainlink
tickers = ["LTC", "LINK"]


# In[4]:


# Crypto API URLs
ltc_url = "https://api.alternative.me/v2/ticker/Litecoin/?convert=USD"
chl_url = "https://api.alternative.me/v2/ticker/Chainlink/?convert=USD"
btc_url = "https://api.alternative.me/v2/ticker/Bitcoin/?convert=USD"


# In[5]:


# Fetch current LTC price

response_data_ltc=requests.get(ltc_url)
print(response_data_ltc)
response_content_ltc = response_data_ltc.content
data_ltc = response_data_ltc.json()

#print(json.dumps(data_ltc, indent=4))

ltc_df = data_ltc['data']
#display(ltc_df)

my_ltc=ltc_df['2']['name']

my_ltc_value=ltc_df['2']['quotes']['USD']['price']

# Fetch current CHL price

response_data_chl=requests.get(chl_url)
print(response_data_chl)
response_content_chl = response_data_chl.content
data_chl = response_data_chl.json()

#print(json.dumps(data_chl, indent=4))

chl_df = data_chl['data']

my_chl=chl_df['1975']['name']
# Compute current value of my crpto
# YOUR CODE HERE!
my_chl_value=chl_df['1975']['quotes']['USD']['price']


# Print current crypto wallet balance
print(f"The current value of your {my_ltc} LTC is ${my_ltc_value:0.2f}")
print(f"The current value of your {my_chl} CHL is ${my_chl_value:0.2f}")


# In[6]:


# Set Alpaca API key and secret
alpaca_api_key = os.getenv ("ALPACA_API_KEY")
alpaca_secret_key = os.getenv("ALPACA_SECRET_KEY")
print(f"Alpaca key Type: {type(alpaca_api_key)}")
print(f"Alpaca Secret Key Type: {type(alpaca_secret_key)}")
# Create the Alpaca API object
alpaca = tradeapi.REST (
    alpaca_api_key,
    alpaca_secret_key,
    api_version = "v2" )


# ### 1.1 Create Dataframe for Litecoin and Chainlink

# In[7]:


# Set start and end dates of five years
start_date = pd.Timestamp('2017-09-28', tz='America/New_York').isoformat()
end_date = pd.Timestamp('2022-09-28', tz='America/New_York').isoformat()
timeframe = "1Day"

# Get current closing prices for Litcoin and Chainlink

df_portfolio = alpaca.get_bars(
    tickers,
    timeframe,
    start = start_date,
    end = end_date
).df

# Generating dataframe for Litcoin and Chainlink
LTC = df_portfolio [df_portfolio ['symbol'] == 'LTC'].drop('symbol', axis=1)
LINK = df_portfolio [df_portfolio ['symbol'] == 'LINK'].drop('symbol', axis=1)
df_portfolio.head()


# ### 1.2 Create Dataframe for Bitcoin 

# In[8]:


# Fetch current BTC price

response_data_btc=requests.get(btc_url)
print(response_data_btc)
response_content_btc = response_data_btc.content
data_btc = response_data_btc.json()

#print(json.dumps(data_btc, indent=4))

btc_df = data_btc['data']
#display(btc_df)


my_btc=btc_df['1']['name']

my_btc_value=btc_df['1']['quotes']['USD']['price']

# Print current crypto wallet balance
print(f"The current value of your {my_btc} BTC is ${my_btc_value:0.2f}")


# In[9]:


# Set start and end dates of five years
ticker_btc = ['BTCUSD']
start_date = pd.Timestamp('2017-09-28', tz='America/New_York').isoformat()
end_date = pd.Timestamp('2022-09-28', tz='America/New_York').isoformat()
timeframe = "1Day"

# Get dataframe for Bitcoin

df_btc = alpaca.get_crypto_bars(
    ticker_btc,
    timeframe,
    start = start_date,
    end = end_date
).df

# Reorganize and clean up the Bitcoin DataFrame
BTC = df_btc [df_btc['symbol'] == 'BTCUSD'].drop('symbol', axis=1)
df_btc = df_btc.drop(columns = ['open', 'high', 'low', 'volume', 'trade_count', 'vwap'])
df_btc.index = df_btc.index.date
df_btc.head()


# In[10]:


# Daily return for Bitcoin
daily_return_btc = df_btc['close'].pct_change()
daily_return_btc = daily_return_btc.dropna()
daily_return_btc.head()


# ### 1.3 Combine Litecoin and Chainlink as a portfolio

# In[11]:


# Concatenate the cryptocurrency portfolio DataFrames for Litecoin and Chainlink
df_portfolio_crypto = pd.concat([LTC, LINK],axis=1, keys=['LTC','LINK'])
df_portfolio_crypto.head()


# In[12]:


# Create and empty Dataframe for closing prices
df_crypto_price = pd.DataFrame()

# Fetch the closing prices of LTC and LINK
df_crypto_price["LTC"] = df_portfolio_crypto["LTC"]["close"]
df_crypto_price["LINK"] = df_portfolio_crypto["LINK"]["close"]


# Drop the time component of the date
df_crypto_price.index = df_crypto_price.index.date

# Drop the null value for cryptocurrency portfolio
df_crypto_price = df_crypto_price.dropna()
df_crypto_price.head()


# ### 1.4 Create line graph to show Litecoin and Chainlink closed price performance

# In[13]:


# 5 years closed price performance for Litecoin and Chainlink
df_crypto_price.hvplot(
    xlabel='Date',
    ylabel='Closed Price',
    label ='Five Years Performance for Litecoin vs Chainlink'
)


# ### 1.5 What are the daily return, standard deviation, covariance/variance, correlation and Beta? 

# In[14]:


#  Daily Return for both cryptocurrency
df_daily_return_crypto = df_crypto_price.pct_change()
df_daily_return_crypto = df_daily_return_crypto.dropna()
df_daily_return_crypto.head()


# In[15]:


# Mean of closed price for both Litecoin and Chainlink
mean_return_crypto = df_crypto_price.mean()
mean_return_crypto


# In[16]:


# Standard deviation of both Litecoin and Chainlink
standard_dev_crypto = df_daily_return_crypto.std()
standard_dev_crypto


# In[17]:


# Variance for cryptocurrency
variance_crypto = df_daily_return_crypto.var()
variance_crypto


# In[18]:


# Covariance for cryptocurrency
covariance_crypto = df_daily_return_crypto.cov()
covariance_crypto


# In[19]:


# Correlation for cryptocurrency
correlation_crypto = df_daily_return_crypto.corr()
correlation_crypto


# In[20]:


# # Visualize correlation between Litcoin and Chainlink
sns.heatmap(correlation_crypto, vmin=-1, vmax=1)


# In[21]:


# Beta for Litcoin
ltc_corvariance = df_daily_return_crypto['LTC'].cov(daily_return_btc)
btc_variance = daily_return_btc.var()
beta_ltc = ltc_corvariance / btc_variance

#beta_ltc
print(f"Beta for Litecoin is {beta_ltc: 2f}%")


# In[22]:


# Beta for Chainlink
link_corvariance = df_daily_return_crypto['LINK'].cov(daily_return_btc)
beta_link = link_corvariance / btc_variance

#beta_ltc
print(f"Beta for Chainlink is {beta_link: 2f}%")


# In[23]:


# Calculate cumulative return for crptocurrency
crypto_cumulative_returns = (1 + df_daily_return_crypto).cumprod() - 1

crypto_cumulative_returns['LTC'] = crypto_cumulative_returns['LTC'].round(5)
crypto_cumulative_returns['LINK'] = crypto_cumulative_returns['LINK'].round(5)
crypto_cumulative_returns.head()


# ### 1.6 Create line graph for cumulative return for both Litecoin and Chainlink

# In[24]:


# Ploting cummualative return for cryptocurrency
crypto_cumulative_returns.hvplot(xlabel='Year',ylabel='Cumulative Return Percentage', title='Cumulative Return Percentage for Litecoin vs Chainlink')


# In[25]:


# Annualized Sharpe Ratios for cryptocurrency portfolio
annual_sharpe_ratios = (crypto_cumulative_returns.mean()*252) / (crypto_cumulative_returns.std() * np.sqrt(252))
annual_sharpe_ratios


# ### 1.7 Investment performance on cryptocurrency portfolio

# In[26]:


# Set initial investment
initial_investment_crypto = 20000

# Set weights
weights_crypto = [0.5, 0.5]

# Calualte the investment on cryptocurrency by weighted daily return
crypto_returns = df_daily_return_crypto.dot(weights_crypto)

# Multiply the initial investment of $10,000 against the portfolio's series of cumulative returns
crypto_cumulative_profits = (initial_investment_crypto * crypto_cumulative_returns)

# Plot the cumulatives return
crypto_cumulative_profits.hvplot(xlabel='Year', ylabel='Investment Profit', title="Profit Performance for Investment on Litecoin and Chainlink")


# From the graph of investment on Litcoin and Chainlink, Chainlink is risky, Litcoin is less risky. We will earn money from Chainlink but we will lose money from Litcoin.

# # Part 2 - Apple and Tesla Stock Analysis

# ### 2.1 Import Data for Apple stock price

# In[27]:


# Read the Apple CSV into DataFrame and display a few rows
appl_df = pd.read_csv('AppleData.csv', index_col='Date', parse_dates=True, infer_datetime_format=True)
display(appl_df.head())
display(appl_df.tail())


# In[28]:


# Drop Volume, Open, High and Low, leave Closed price only
appl_df = appl_df.drop(columns=['Volume', 'Open', 'High', 'Low'])
appl_df = appl_df.sort_index()
appl_df.head()


# In[29]:


# Rename the column name to APPLE
columns = ['APPLE']
appl_df.columns = columns
appl_df['APPLE'] = appl_df['APPLE'].str.replace("$", "")
appl_df['APPLE'] = appl_df['APPLE'].astype("float")
appl_df.head()


# ### 2.2 Import data for Tesla stock price

# In[30]:


# Read the Tesla CSV into DataFrame and display a few rows
tesla_df = pd.read_csv('TeslaData.csv', index_col='Date', parse_dates=True, infer_datetime_format=True)
display(tesla_df.head())
display(tesla_df.tail())


# In[31]:


# Drop Volume, Open, High and Low, leave Closed price only
tesla_df = tesla_df.drop(columns=['Volume', 'Open', 'High', 'Low'])
tesla_df = tesla_df.sort_index()
tesla_df.head()


# In[32]:


# Rename the column name to TESLA
columns = ['TESLA']
tesla_df.columns = columns
tesla_df['TESLA'] = tesla_df['TESLA'].str.replace("$", "")
tesla_df['TESLA'] = tesla_df['TESLA'].astype("float")
tesla_df.head()


# ### 2.3 Create a stock portfolio for Apple and Tesla

# In[33]:


# Concatenate the stock portfolio DataFrames
df_stock_price = pd.concat([appl_df, tesla_df],axis="columns", join="inner")
df_stock_price.head()


# ### 2.4 Create line graph to see closed price performance for portfolio

# In[34]:


df_stock_price.hvplot(
    xlabel='Date',
    ylabel = "Closed Price",
    title="Five Years Stock price for APPLE vs TESLA"
)


# In[35]:


# check if there is null value
df_stock_price.isnull().sum()


# In[36]:


# Daily return for Apple and Tesla
daily_return_stock = df_stock_price.pct_change()
daily_return_stock = daily_return_stock.dropna()
daily_return_stock.head()


# In[37]:


# Average closed price for Apple and Tesla
mean_stock = df_stock_price.mean()
mean_stock


# In[38]:


# Standard deviation of daily return for Apple and Tesla
standard_dev_stock = daily_return_stock.std()
standard_dev_stock


# In[39]:


# Rename closed price for APPLE
appl_df.rename(columns={'APPLE' : 'close'}, inplace=True)
appl_df= pd.concat([appl_df], axis=1, keys='APPLE')
appl_df.head()


# ### 2.5 Monte Carlo simulation on Apple

# In[40]:


# Set number of simulations
num_sims = 500

# Configure a Monte Carlo simulation to forecast three years daily returns for APPLE
MC_APPLE = MCSimulation(
    portfolio_data = appl_df,
    num_simulation = num_sims,
    num_trading_days = 252 * 3
)


# In[41]:


# Run Morte Carlo Simulation 3 years daily return for APPLE
MC_APPLE.calc_cumulative_return()


# In[42]:


# Compute summary statistics from the simulated daily returns for APPLE
simulated_returns_appl = {
    "mean": list(MC_APPLE.simulated_return.mean(axis=1)),
    "median": list(MC_APPLE.simulated_return.median(axis=1)),
    "min": list(MC_APPLE.simulated_return.min(axis=1)),
    "max": list(MC_APPLE.simulated_return.max(axis=1))
}

# Create a DataFrame with the summary statistics for APPLE
appl_simulated_returns = pd.DataFrame(simulated_returns_appl)

# Display sample data
appl_simulated_returns.head()


# ### 2.6 Create line graph for Apple 3 years daily return simulation

# In[47]:


# Visualize the trajectory of AAPL stock daily returns on 3 years simulation
appl_simulated_returns.hvplot(xlabel = "Number of days cumulated", title="Next Three Years Daily Return Simulation for APPLE")


# In[45]:


# Set initial investment
initial_investment_appl = 10000

# Calulate cumulative statistics data with investment amount for APPLE
cumulative_appl = initial_investment_appl * appl_simulated_returns
cumulative_appl.head()


# ### 2.7 Create a line graph to forcast Profit performance for Apple

# In[48]:


# Create a chart of the simulated profits/losses for Apple
appl_plot = cumulative_appl.hvplot(
    xlabel="Number of days cumulated", 
    title="Simulated Outcomes Behavior of APPL Stock Over the Next Three Year",
).opts(
    yformatter="%.0f"
)
appl_plot


# In[49]:


# Summarize statistics from the Monte Carlo Simulation results
tbl_appl = MC_APPLE.summarize_cumulative_return()
print(tbl_appl)


# In[50]:


# Use the lower and upper `95%` confidence intervals to calculate the range of the possible outcomes with $10,000 investments in APPLE stocks
ci_lower = round(tbl_appl[8]*initial_investment_appl,2)
ci_upper = round(tbl_appl[9]*initial_investment_appl,2)
print(f"There is a 95% chance that an initial investment of $10,000 in APPLE stock"
      f" over the next year will end within in the range of"
      f" ${ci_lower} and ${ci_upper}.")


# In[51]:


# Rename closed price for TESLA
tesla_df.rename(columns={'TESLA' : 'close'}, inplace=True)
tesla_df= pd.concat([tesla_df], axis=1, keys='TESLA')
tesla_df.head()


# In[52]:


# Set number of simulations
num_sims = 500

# Configure a Monte Carlo simulation to forecast three years daily returns for TESLA
MC_TESLA = MCSimulation(
    portfolio_data = tesla_df,
    num_simulation = num_sims,
    num_trading_days = 252 * 3
)


# In[53]:


# Run Morte Carlo Simulation 3 years daily return for TESLA
MC_TESLA.calc_cumulative_return()


# In[54]:


# Compute summary statistics from the simulated daily returns for TESLA
simulated_returns_tesla = {
    "mean": list(MC_TESLA.simulated_return.mean(axis=1)),
    "median": list(MC_TESLA.simulated_return.median(axis=1)),
    "min": list(MC_TESLA.simulated_return.min(axis=1)),
    "max": list(MC_TESLA.simulated_return.max(axis=1))
}

# Create a DataFrame with the summary statistics for TESLA
tesla_simulated_returns = pd.DataFrame(simulated_returns_tesla)
tesla_simulated_returns.head()


# ### 2.8 Create line graph for Tesla 3 years daily return simulation

# In[60]:


# Create a chart of the simulated profits/losses for TESLA
tesla_simulated_returns.hvplot(
    xlabel = "Number of days cumulated", 
    label="Next Three Years Daily Return Simulation for TESLA"
)


# In[61]:


# Set initial investment
initial_investment_tesla = 10000

# Calulate cumulative statistics data with investment amount for TESLA
cumulative_tesla = initial_investment_tesla * tesla_simulated_returns
cumulative_tesla.head()


# ### 2.9 Create a line graph to forcast 3 years profit for Tesla

# In[62]:


# Create a chart of the simulated profits/losses for TESLA
tesla_plot = cumulative_tesla.hvplot(
    xlabel="Number of days cumulated", 
    label="Simulated Outcomes Behavior of TESLA Stock Over the Next Three Year",
).opts(
    yformatter="%.0f"
)
tesla_plot


# In[63]:


# Summarize statistics from the Monte Carlo Simulation results
tbl_tesla = MC_TESLA.summarize_cumulative_return()
print(tbl_tesla)


# In[64]:


# Use the lower and upper `95%` confidence intervals to calculate the range of the possible outcomes with $10,000 investments in APPLE stocks
ci_lower_tesla = round(tbl_tesla[8]*initial_investment_tesla,2)
ci_upper_tesla = round(tbl_tesla[9]*initial_investment_tesla,2)
print(f"There is a 95% chance that an initial investment of $10,000 in APPLE stock"
      f" over the next year will end within in the range of"
      f" ${ci_lower_tesla} and ${ci_upper_tesla}.")


# ### 2.10 Compare Tesla and Apple 3 years outcomes performace

# In[66]:


# Compare two plots to visualize both APPLE and TESLA simulated outcomes
appl_plot + tesla_plot


# According to the assumption of investment for Apple and Tesla, we can forcast that Tesla can bring more profit for us with same amount of invesment.

# ## Part 3 - Combined Portfolio for cryptocurrency and stock

# In[67]:


# Combine cryptocurrency and stock to a portfolio
df_portfolio_all = pd.concat([df_crypto_price, df_stock_price], axis="columns")
df_portfolio_all = df_portfolio_all.dropna()
df_portfolio_all.head()


# ### 3.1 Closed price performance for each single portfolio

# In[68]:


# Plot price movement for portfolio
df_portfolio_all.hvplot(
    xlabel='Date',
    ylabel='Closed Price',
    title='Closed Price Performance of Portfolio'
)


# In[70]:


# Daily return of portfolio
daily_return_all = df_portfolio_all.pct_change()
daily_return_all = daily_return_all.dropna()
daily_return_all.head()


# In[71]:


# set the weights of investment for portfolio
weights = [0.2,0.3, 0.4, 0.6]

portfolio_returns = daily_return_all.dot(weights)
portfolio_returns.head()


# In[72]:


# Calculate cumulative portfolio returns
cumulative_returns = (1 + portfolio_returns).cumprod() - 1
cumulative_returns.head()


# ### 3.2 Investment outcomes for our portfolio

# In[73]:


# Plot cumulative portfolio returns
cumulative_returns.hvplot(
    xlabel='Date',
    ylabel='Percentage cumulative returns',
    title='Portfolio Cumulative Returns Performance'
)


# In general, our portfolio is making profit. However, cumulative returns after 2021 has significant fluctutation
