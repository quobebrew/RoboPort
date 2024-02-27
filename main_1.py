
#import relevant libraries
import pandas as pd
import streamlit as st
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plts
from scipy.optimize import minimize
from pathlib import Path




st.title('Welcome to RoboPort')
st.subheader('Your ultimate robo portfolio optimization tool')


# funtions
# Data Retrieval from Yahoo Finance
def get_historical_prices(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return pd.DataFrame(data)

# Use the `pct_change` function to calculate daily returns of closing prices for each column
def get_daily_returns(price):
    returns = price.pct_change()
    return returns

# Use the `dot` function to multiply the weights by each stock's daily return to get the portfolio daily return
def get_portfolio_returns(weights,daily_returns):   
    portfolio_returns = daily_returns.dot(weights)
    return portfolio_returns

# Risk Parity: invest such a way that every asset we have in the portfolio has the same risk contribution
def calculate_risk_parity_weights(returns):

     # Calculate asset volatilities
    asset_volatility = returns.std(axis=0)

    # Calculate asset risk contributions
    asset_risk_contribution = asset_volatility / asset_volatility.sum()

    # Determine target risk allocation (e.g., equal risk)
    target_risk_allocation = 1 / len(asset_volatility)

    # Calculate weights based on risk contributions
    weights = target_risk_allocation / asset_risk_contribution

    # Normalize weights to sum to 1
    weights /= weights.sum()

    return weights


 #Calculate Beta coefficient for each ticker to benchmark, SP500
def calculate_beta(daily_returns, benchmark_returns):

    beta_list = {}
    
    # Iterate over each asset in the portfolio
    for ticker in daily_returns:
        # Calculate covariance between asset returns and benchmark returns
        covariance = daily_returns[ticker].cov(benchmark_returns)
        
        # Calculate variance of benchmark returns
        variance = benchmark_returns.var()
        
        # Calculate beta coefficient
        beta = covariance / variance
        
        # Store beta value in the dictionary
        beta_list[ticker] = beta
    
    # Create a Series from the dictionary
    beta_series = pd.Series(beta_list, name='Beta')
    
    return beta_series

# Calculate New Portfolio Weights Based on Stock Betas

def calculate_beta_weights(data):
    if isinstance(data, pd.Series):
        # If input is a Series, create a DataFrame with one column
        df = pd.DataFrame(data, columns=['Beta'])
    else:
        # If input is already a DataFrame, use it directly
        df = data

    beta_weights = {}
    target_market_beta = 1
    sum_of_all_stock_betas = df['Beta'].sum()
    
    for index, row in df.iterrows():
        numerator = target_market_beta - row['Beta']
        denominator = sum_of_all_stock_betas - row['Beta']
        stock_weight = numerator / denominator
        beta_weights[index] = stock_weight
        
    beta_weights_df = pd.DataFrame.from_dict(beta_weights, orient='index', columns=['Weight'])
    beta_weights_df_normalized = beta_weights_df['Weight'] / beta_weights_df['Weight'].sum()
    
    return beta_weights_df_normalized

#Calculate Markowitz Portfolio Optimization
def calculate_Markowitz(random_weight, data_returns):
    returns = data_returns/data_returns.shift(1)
    log_returns = np.log(returns)
    meanlog = log_returns.mean()
    sigma = log_returns.cov()
    random_weights = np.array(random_weight)
    R = np.sum(meanlog*random_weights)
    V = np.sqrt(np.dot(random_weight.T,np.dot(sigma,random_weights)))
    SR = R/V
    return -SR

# get today's date and calculate the start date as one year ago. 
end_date = datetime.today().date()
start_date = end_date - timedelta(days=365)

total_amount = st.number_input("Please enter the total dollar amount in your portfolio?:")
amount = float(total_amount)
#format amount entered
formatted_amount = "${:,.2f}".format(amount)
st.write('Total portfolio amount:', formatted_amount)
        


st.subheader('Please enter the tickers in your portfolio and their respective percentages')

# Initialize a dictionary to store ticker symbols and percentages
ticker_percentage = {}

# Define the number of ticker symbols to input
num_tickers = st.number_input('Number of tickers:', min_value=1, max_value=10, value=1, step=1)

# Input fields for ticker symbols and percentages
for i in range(1, num_tickers + 1):
    ticker = st.text_input(f'Ticker {i}:', key=f'ticker_{i}')
    percentage = st.number_input(f'Percentage {i} ({ticker}):', key=f'percentage_{i}')
    ticker_percentage[ticker] = percentage / 100

# Button to display entered data
if st.button('Submit'):
    for ticker, percentage in ticker_percentage.items():
        st.write(f'Ticker: {ticker}, weight: {percentage}')
   
    
    #Risk Level to be computed later.
   

# get tickers from dictionary
    tickers = list(ticker_percentage.keys())
    weights = list(ticker_percentage.values())

#plot weights as pie chart   
    st.subheader('Pie Chart of Portfolio Weights')
    fig, ax=plts.subplots()
    ax.pie(weights, labels=tickers, autopct='%0.1f%%', startangle=140)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Display the pie chart
    st.pyplot(fig)

    #get prices from get_historical_prices function
    prices = get_historical_prices(tickers, start_date, end_date)
    st.subheader("Historic Prices for the past year")
    prices

 # Display historic prices if available
    if prices is not None:
        plts.figure(figsize=(10, 6))
        for column in prices.columns:
            plts.plot(prices.index, prices[column], label=column)
        plts.xlabel('Date')
        plts.ylabel('Price')
        plts.title('Historic Prices for the past year')
        plts.legend()
        st.pyplot(plts)
        
    else:
        st.write("Historical prices not available.")
 # Plot using Matplotlib for Historic Prices for the past year


# Percentage change Daily Returns
    daily_returns = get_daily_returns(prices)
    st.subheader("Daily Returns (pct_change)")
    daily_returns

 # Plot using Matplotlib for Historic Prices for the past year
    plts.figure(figsize=(10, 6))
    for column in daily_returns.columns:
        plts.plot(daily_returns.index, daily_returns[column], label=column)
    plts.xlabel('Date')
    plts.ylabel('Percentage Change')
    plts.title('Percentage Change of Daily Returns')
    plts.legend()
    st.pyplot(plts)


#use calculate_portfolio_return function to get the portfolio daily retunr
    port_daily_return = get_portfolio_returns(weights, daily_returns)

# Display the portfolio daily return
    st.subheader("Portfolio Daily Returns")
    port_daily_return

#display graph
    plts.figure(figsize=(10, 6))
    plts.plot(port_daily_return.index, port_daily_return, label='Portfolio Returns')
    plts.xlabel('Date')
    plts.ylabel('Portfolio Returns')
    plts.title('Portfolio Returns OVer Time')
    plts.legend()
    st.pyplot(plts)


#Beta
    benchmark_ticker = "^GSPC"
    benchmark_prices = get_historical_prices(benchmark_ticker, start_date, end_date)
    benchmark_daily_returns = get_daily_returns(benchmark_prices)
    #benchmark_daily_returns.set_index(benchmark_daily_returns.columns[0], inplace=True)
    benchmark_daily_returns =benchmark_daily_returns.iloc[:, 0]
    betas = calculate_beta(daily_returns,benchmark_daily_returns)
    st.subheader(f"Beta Coefficient By Tickers benchmarked with {benchmark_ticker}")
    betas

# calculate beta weights 
    beta_weight = calculate_beta_weights(betas)
    st.subheader("Beta weight")
    beta_weight
#use calculate_risk_parity_weights to get Risk Parity
    risk_parity_weights= calculate_risk_parity_weights(daily_returns)
    st.subheader("Risk Parity Weights")
    risk_parity_weights

# Pie Graph for Risk Parity
    st.subheader('Risk Parity (Equally weighted portfolio)')
    fig, ax=plts.subplots()
    ax.pie(risk_parity_weights, labels=tickers, autopct='%0.1f%%', startangle=140)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Display the pie chart
    st.pyplot(fig)

# Sharpe Ratio
    # get rf rate from csv
    risk_free_rate_path = Path("yield-curve-rates-1990-2023.csv")
    risk_free_rate = pd.read_csv(risk_free_rate_path, index_col='Date', parse_dates=True, infer_datetime_format=True)
    
    # use the 10yr data and rename as rf_rate
    rf_rate = risk_free_rate.iloc[:, 10].rename('rf_rate')
    annualized_risk_free_rate = (1 + rf_rate)**252 - 1
    annualized_risk_free_rate
    all_portfolios_returns = pd.concat([daily_returns,annualized_risk_free_rate], axis='columns', join='inner')
    
    #calculate Sharpe Ratio
    sharpe_ratios = ((all_portfolios_returns.mean()-all_portfolios_returns['rf_rate'].mean()) * 252) / (all_portfolios_returns.std() * np.sqrt(252))
    st.subheader("Sharpe Ratio")
    sharpe_ratios
    
    











