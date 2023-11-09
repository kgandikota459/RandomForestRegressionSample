import pandas as pd
from sklearn.model_selection import train_test_split
import scipy as sp
import sklearn.ensemble as sk

df = pd.read_csv('mbs.csv')



from sklearn.impute import SimpleImputer

x = df.drop(['seqn', 'Sex', 'Marital', 'Race', 'Triglycerides'], axis=1)
y = df['Triglycerides']


# Replace missing values with the mean of the non-missing values in the same column
imputer = SimpleImputer()
x = imputer.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .2, random_state = 78)

g2 = sk.GradientBoostingRegressor(n_estimators = 10000, random_state = 42)
g2.fit(x_train, y_train)

pred = g2.predict(x)

results_df = pd.DataFrame({'Triglycerides': y, 'Prediction of Trigs': pred})

results_df = results_df.reset_index(drop=True)
print(results_df)


    
  