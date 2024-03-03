
#import relevant libraries
import pandas as pd
import streamlit as st
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plts
from scipy.optimize import minimize
from pathlib import Path
import matplotlib.pyplot as plt




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
def portfolio_value_evoluvation(tickers, value_test_weight, years):
        if np.sum(value_test_weight) != 1:
            print("Sum of weights must be 1.")
            return None

        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=years * 365)

        stock_data = get_historical_prices(tickers, start_date, end_date)
        weighted_stock_price = stock_data * value_test_weight
        stock_data.loc[:, "Profit Close"] = weighted_stock_price.sum(axis=1)
        return stock_data


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


    import datetime
    import matplotlib.pyplot as plt
    import streamlit as st
    return_value = {}
    

    orginal_portfolio_value = portfolio_value_evoluvation(tickers, weights, 3)
    if orginal_portfolio_value is not None:
        st.subheader('Total return on intial allocation')
        plt.figure(figsize=(40, 12))
        plt.title(f"Portfolio Value Evolution (10 years) using user allocation",fontsize=40)
        plt.plot(orginal_portfolio_value["Profit Close"], color='blue')  # Use 'Profit Close' instead of 'Close'
        plt.xlabel('Date',fontsize=40)
        plt.ylabel('Value in $',fontsize=40)
        st.pyplot(plt)

        Total_return_o = (orginal_portfolio_value['Profit Close'][-1]/orginal_portfolio_value['Profit Close'][0])-1
        st.write(f"Total portfolito return on user allocation: {Total_return_o:.2%}")
        return_value['User']=Total_return_o
    else: 
        return_value['User']=0
    # risk parity weights
    rp_portfolio_value = portfolio_value_evoluvation(tickers, risk_parity_weights, 3)
    if rp_portfolio_value is not None:
        st.subheader('Total return on risk parity allocation')
        

        plt.figure(figsize=(40, 12))
        plt.title(f"Portfolio Value Evolution (10 years) using Risk parity",fontsize=40)
        plt.plot(rp_portfolio_value["Profit Close"], color='blue')  # Use 'Profit Close' instead of 'Close'
        plt.xlabel('Date',fontsize=40)
        plt.ylabel('Value in $',fontsize=40)
        st.pyplot(plt)

        Total_return_rp = (rp_portfolio_value['Profit Close'][-1]/rp_portfolio_value['Profit Close'][0])-1
        st.write(f"Total portfolito return on risk parity allocation: {Total_return_rp:.2%}")
        return_value['Risk Parity']=Total_return_rp
    else: 
        return_value['Risk Parity']=0
    # beta retturn
    beta_portfolio_value = portfolio_value_evoluvation(tickers, beta_weight, 3)
    if beta_portfolio_value is not None:
        st.subheader('Total return on Beta Weight allocation')
    

        plt.figure(figsize=(40, 12))
        plt.title(f"Portfolio Value Evolution (10 years) using Beta Weights",fontsize=40)
        plt.plot(beta_portfolio_value["Profit Close"], color='blue')  # Use 'Profit Close' instead of 'Close'
        plt.xlabel('Date',fontsize=40)
        plt.ylabel('Value in $',fontsize=40)
        st.pyplot(plt)

        Total_return_beta = (beta_portfolio_value['Profit Close'][-1]/beta_portfolio_value['Profit Close'][0])-1
        st.write(f"Total portfolito return bata allocation: {Total_return_beta:.2%}")
        return_value['Beta']=Total_return_beta
    else:
        return_value['Beta']=0
# Marcovic analysis and sharp ratio
    
    st.subheader('Sharp Ratio')
    from scipy.optimize import minimize
    returns_marco = prices/(prices.shift(1))
    returns_marco.dropna(inplace=True)
    logreturns=np.log(returns_marco)
    meanlog = logreturns.mean()
    no_porfolio = 10000
    test_weight = np.zeros((no_porfolio,num_tickers))
    sigma = logreturns.cov()
    test_return = np.zeros(no_porfolio)
    test_volatility = np.zeros(no_porfolio)
    sharpratio = np.zeros(no_porfolio)
    for k in range(no_porfolio):
        random_weight = np.array(np.random.random(num_tickers))
        random_weight = random_weight/sum(random_weight)
        test_weight[k,:] = random_weight
        #log returns
        test_return[k] = np.sum(meanlog * random_weight) 
        #volatility
        test_volatility[k] = np.sqrt(np.dot(random_weight.T,np.dot(sigma,random_weight)))
        #sharp ratio
        sharpratio[k] = test_return[k]/test_volatility[k]
    max_sharpratio = sharpratio.argmax()
    sharpratio_weight=test_weight[max_sharpratio,:]
    st.subheader('Sharpe ratio of 10000 random weights')
    # plolt of random weights sharp ratio
    plt.figure(figsize=(40, 12))
    plt.scatter(test_volatility, test_return, c=sharpratio)
    plt.xlabel('Volatility')
    plt.ylabel('Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.scatter(test_volatility[max_sharpratio], test_return[max_sharpratio], c='black')
    st.pyplot(plt)
    # total returns if the weights sharp ratio
    sr_portfolio_value = portfolio_value_evoluvation(tickers,sharpratio_weight, 3)
    if sr_portfolio_value is not None:
        plt.figure(figsize=(40, 12))
        plt.title(f"Portfolio Value Evolution (10 years) using Sharp Ratio",fontsize=40)
        plt.plot(sr_portfolio_value["Profit Close"], color='blue')  # Use 'Profit Close' instead of 'Close'
        plt.xlabel('Date',fontsize=40)
        plt.ylabel('Value in $',fontsize=40)
        st.pyplot(plt)

        Total_return_sr = (sr_portfolio_value['Profit Close'][-1]/sr_portfolio_value['Profit Close'][0])-1
        st.write(f"Total portfolito return with Sharp ratio: {Total_return_sr:.2%}")
        return_value['Sharp Ratio']=Total_return_sr
    else:
        return_value['Sharp Ratio']=0
    
    def negative_sharpratio(random_weight):
        random_weights = np.array(random_weight)
        R = np.sum(meanlog*random_weights)
        V = np.sqrt(np.dot(random_weight.T,np.dot(sigma,random_weights)))
        SR = R/V
        return -SR
    def checksumtoone(random_weight):
        return np.sum(random_weight)-1
    w_0 = [1/num_tickers for _ in range(num_tickers)]
    bounds = [(0,1) for _ in range(num_tickers)]
    constraints = ({'type':'eq','fun':checksumtoone})
    optimal_weight = minimize(negative_sharpratio,w_0,method='SLSQP',bounds=bounds,constraints=constraints)
    marcovic_weight = (optimal_weight.x)
    returns=np.linspace(0,max(test_return),50)
    optimal_volatility = []  
    def minmizevolatility(random_weight):
        random_weights = np.array(random_weight)
        V = np.sqrt(np.dot(random_weight.T,np.dot(sigma,random_weights)))
        return V
    def getreturn(w):
        w = np.array(w)
        R = np.sum(meanlog*w)
        return R

    for r in returns:
        #find best volatility
        constraints = ({'type':'eq','fun':checksumtoone},{'type':'eq','fun': lambda random_weight: getreturn(random_weight)- r})
        optimal = minimize(minmizevolatility,w_0,method='SLSQP',bounds=bounds,constraints=constraints)
        optimal_volatility.append(optimal['fun'])
    plt.figure(figsize=(40,12))
    plt.scatter(test_volatility,test_return,c=sharpratio)
    plt.xlabel('volatility',fontsize=40)
    plt.ylabel('return',fontsize=40)
    plt.colorbar(label='Sharpe Ratio')
    plt.scatter(test_volatility[max_sharpratio],test_return[max_sharpratio],c='black')
    plt.plot(optimal_volatility,returns,'--')
    st.pyplot(plt)   
    
# portfolio value evoluvation
    

# Display plot using Streamlit
        
    
# Call the function and store the result if needed
    marcovic_portfolio_value = portfolio_value_evoluvation(tickers, optimal_weight.x, 3)
    if marcovic_portfolio_value is not None:
        st.subheader('Markowitz portfolio solver')
        plt.figure(figsize=(40, 12))
        plt.title(f"Portfolio Value Evolution (10 years) using Markowitz",fontsize=40)
        plt.plot(marcovic_portfolio_value["Profit Close"], color='blue')  
        plt.xlabel('Date',fontsize=40)
        plt.ylabel('Value in $',fontsize=40)
        st.pyplot(plt)

        Total_return = (marcovic_portfolio_value['Profit Close'][-1]/marcovic_portfolio_value['Profit Close'][0])-1
        st.write(f"Total portfolito return using marcovic: {Total_return:.2%}")
        return_value['Markowitz']=Total_return

        
    else:
        return_value['Markowitz']=0
    max_return = max(return_value, key=lambda k: return_value[k])
    st.write(f"As per analysis using varies methods the best return is got from using : {max_return}. Below is the recomended distribution to get the best return:")
    st.write("Corresponding weights")
    import os
    import requests
    from dotenv import load_dotenv
    import alpaca_trade_api as tradeapi
    from MCForecastTools import MCSimulation
    if max_return == 'Risk Parity':
        fig, ax=plts.subplots()
        ax.pie(risk_parity_weights, labels=tickers, autopct='%0.1f%%', startangle=140)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig)
        df = pd.DataFrame({'Key': tickers, 'Value': risk_parity_weights})
        st.write(df)

    elif max_return == 'Markowitz':
        fig, ax=plts.subplots()
        ax.pie(optimal_weight.x, labels=tickers, autopct='%0.1f%%', startangle=140)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig)
        df = pd.DataFrame({'Key': tickers, 'Value': optimal_weight.x})
        st.write(df)
    elif max_return == 'User':
        fig, ax=plts.subplots()
        ax.pie(weights, labels=tickers, autopct='%0.1f%%', startangle=140)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig)
        st.write("Do not make changes to your allocation")
        df = pd.DataFrame({'Key': tickers, 'Value': weights})
        st.write(df)

    elif max_return == 'Beta':
        fig, ax=plts.subplots()
        ax.pie(beta_weight, labels=tickers, autopct='%0.1f%%', startangle=140)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig)
        df = pd.DataFrame({'Key': tickers, 'Value': beta_weight})
        st.write(df)
    elif max_return == 'Sharp Ratio':
        fig, ax=plts.subplots()
        ax.pie(sharpratio_weight, labels=tickers, autopct='%0.1f%%', startangle=140)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig)
        df = pd.DataFrame({'Key': tickers, 'Value': sharpratio_weight})
        st.write(df)
    







