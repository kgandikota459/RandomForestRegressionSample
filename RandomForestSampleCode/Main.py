import pandas as pd
from sklearn.model_selection import train_test_split
import scipy as sp
import sklearn.ensemble as sk
df = pd.read_csv('/workspaces/RandomForestRegressionSample/RandomForestSampleCode/apl.csv')
df.dropna()
df = df.drop('Date', axis=1)
df = df.replace({'\$': '', ',': ''}, regex=True).astype(float)

x = df.drop('Open', axis=1)
y = df['Open']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
# END: 8f3c5d7b7e7a
g1 = sk.RandomForestRegressor(n_estimators = 100, random_state = 42)
g1.fit(x_train, y_train)
pred = g1.predict(x_test)

results_df = pd.DataFrame({'Actual Open': y_test, 'Predicted Open': pred})
results_df = results_df.reset_index(drop=True)
print(results_df.head(10))


    
  