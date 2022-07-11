# Currency-return-prediction
This MDN model uses return in the last 10 minutes, S&P500, Euro Stoxx 50, implied volatility and interest rate as features to predict the EURUSD currency return. 

Before the model training, I used the min-max scaler to scale all the data. And finally, calculate the approximate confidence interval for every data in the testing set to see if the results fall within this interval. 

And I got the prediction accuracy at 50% when using 5000 data, and 75% when using 13000 data.
