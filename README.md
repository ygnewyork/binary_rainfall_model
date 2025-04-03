# binary_rainfall_model

Here's a model dataset to predict rain based on weather factors, placed in top 25% of Kagglers for the competition - https://www.kaggle.com/competitions/playground-series-s5e3/leaderboard, with an AUC score of 0.896
Python file implemented a combination of models, used sklearn and tensorflow.
First created a variety of features to predict rainfall such as relative humidity, vapor poressure, etc.
Then appplied models to all of the features, and ran neural networks with various hyperparameters to identify best performing ones as per AUC score. Cross validated data using real submission data as well.
