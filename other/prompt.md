# Question

Using Python, I want to create a Support Vector Model that uses technical indicators to predict future price action for Bitcoin.

The API we will be using to get the OHLCV data is https://min-api.cryptocompare.com/data/histominute

Usually in the parameters we are asked to specify a number of minutes used to fetch data for that period of time. Let's do 1 year worth of minutes.

Here is the list of technical indicators:

- 7-day simple moving average
- 21-day simple moving average
- Exponential moving average with a decay rate of 0.67
- 12-day exponential moving average
- 26-day exponential moving average
- Moving Average Convergence Divergence (MACD) with the 12-day and 26-day exponential moving averages
- 20-day Standard Deviation of BTC Closing price
- 2 Bollinger Bands ± two 20-day standard deviation of the price from the 21-day simple moving average
- High-Low Spread
- Ethereum Price
- Gold spot price
- Moving average indicator

I also want to have specific training, evaluation, and testing score numbers to know what aspect of the model to improve.