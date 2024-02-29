
#import relevant libraries
import pandas as pd
import streamlit as st
import numpy as np
import yfinance as yf
import hvplot.pandas 
from datetime import datetime, timedelta



st.title('Welcome to RoboPort')
st.subheader('Your ultimate robo portfolio optimization tool')

#st.write("What is the total dollar amount in your portfolio?")
total_amount = st.text_input("What is the total dollar amount in your portfolio?")
total_amount

# set an empty list to store user ticker inputs
st.write('Please enter the tickers from your portfolio and the weight:')
ticker_percentage ={}


num_tickers = st.number_input(f'Number of tickers:',min_value=1,max_value=4, value=1, step=1)


# create input fields for up to 10 tickers
for i  in range(1,num_tickers+1):
    ticker = st.text_input(f'Ticker {i}', '')  
    percentage = st.number_input(f'percentage {ticker}')
    ticker_percentage[ticker] = percentage/100
    ticker_percentage

   
# select risk level, aggressive, passive, medium
risk_level = st.selectbox('Select your risk level:', ['','high', 'medium', 'low'])

# display risk level if not empty
if risk_level.strip() !="":
    st.write(f'You select a {risk_level}: risk level')
else:
    st.write("Please select a risk level to proceed to analysis")


# Data Retrieval from Yahoo Finance
def get_historical_prices(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data

# Use the `pct_change` function to calculate daily returns of closing prices for each column
def get_daily_returns(price):
    returns = price.pct_change()
    return returns

# Use the `dot` function to multiply the weights by each stock's daily return to get the portfolio daily return
def get_portfolio_returns(weights,daily_returns):   
    portfolio_returns = daily_returns.dot(weights)
    return portfolio_returns


# get today's date and calculate the start date as one year ago. 
end_date = datetime.today().date()
start_date = end_date - timedelta(days=365)

# get tickers from dictionary
tickers = list(ticker_percentage.keys())
weights = list(ticker_percentage.values())


prices = get_historical_prices(tickers, start_date, end_date)
prices


daily_returns = get_daily_returns(prices)
daily_returns

#use calculate_portfolio_return function to get the portfolio daily retunr
port_daily_return = get_portfolio_returns(weights, daily_returns)

# Display the portfolio daily return
port_daily_return

#display graph
#st.title("Graph of Daily Portfolio Returns")
#st.write(port_daily_return.hvplot.line(x='Date', y='0', xlabel='X',ylabel='Y', title="test"))
#testplot = port_daily_return.hvplot.line(x='Date', y='0', xlabel='X',ylabel='Y', title="test")
#st.bokeh_chart(testplot)

# Risk Parity
def calculate_risk_parity_weights(returns):
    """
    Calculates risk parity weights for a DataFrame of returns.
    
    Parameters:
        returns (DataFrame): DataFrame containing historical returns for all assets.
    
    Returns:
        numpy.array: Array of risk parity weights.
    """
    # Calculate the covariance matrix of returns
    cov_matrix = returns.cov()
    
    # Calculate the inverse of the covariance matrix
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    
    # Calculate the risk contribution of each asset
    risk_contributions = np.dot(inv_cov_matrix, returns.mean())
    
    # Calculate the risk parity weights
    risk_parity_weights = risk_contributions / np.sum(risk_contributions)
    
    return risk_parity_weights

st.write("Risk Parity Weights")
risk_parity_weights= calculate_risk_parity_weights(daily_returns)
risk_parity_weights

#portfolio returns



# Define educational articles section
def educational_articles():
    st.title("Educational Articles")
    st.markdown("## Introduction to Portfolio Optimization")
    st.write("Lorem ipsum dolor sit amet, consectetur adipiscing elit...")

# Display educational articles section
educational_articles()


# Define video content section
def video_content():
    st.title("Video Content")
    st.write("Watch these videos to learn more about portfolio optimization:")
    st.video("https://www.youtube.com/watch?v=VIDEO_ID")
    # Add more videos here

# Display video content section
video_content()




# Define FAQs section
def faqs():
    st.title("FAQs")
    st.write("Explore frequently asked questions about portfolio optimization:")
    with st.expander("What is Portfolio Optimization?"):
        st.write("Portfolio optimization is...")
    # Add more FAQs here

# Display FAQs section
faqs()
