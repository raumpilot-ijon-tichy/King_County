import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np



#read data
df_kc = pd.read_csv("King_County_House_prices_dataset.csv")

#set correct format
df_kc["id"] = df_kc["id"].astype("str")
df_kc["date"] = pd.to_datetime(df_kc["date"]).dt.date
df_kc["price"] = df_kc["price"].astype("int")
df_kc["waterfront"] = df_kc["waterfront"].astype("category")
df_kc["view"] = df_kc["view"].fillna(0).astype("int")
df_kc["condition"] = df_kc["condition"].astype("category")
df_kc["grade"] = df_kc["grade"].astype("category")
df_kc["sqft_basement"] = pd.to_numeric(df_kc["sqft_basement"], errors='coerce')
df_kc["zipcode"] = df_kc["zipcode"].astype("category")
df_kc["yr_renovated"] = df_kc["yr_renovated"].fillna(0).astype("int")

#fill NaNs
df_kc["waterfront"] = df_kc["waterfront"].fillna(0)
for i in range(df_kc.shape[0]):
    if df_kc["yr_renovated"][i] == 0:
        df_kc["yr_renovated"][i] = df_kc["yr_built"][i]

#feature engineering
df_kc["pp_sqft_living"] = df_kc["price"] / df_kc["sqft_living"]
df_kc["pp_sqft_lot"] = df_kc["price"] / df_kc["sqft_lot"]

#feature selection
X = df_kc[["grade", "sqft_living", "sqft_living15", "pp_sqft_living", "bathrooms"]]
Y = df_kc["price"]

#splitting data
print("-----  Splitting the data in train and test ----")
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

#adding the constant
X_train = sm.add_constant(X_train) # adding a constant
X_test = sm.add_constant(X_test) # adding a constant

#training the model
print("-----  Training the model ----")
model = sm.OLS(y_train, X_train).fit()
print_model = model.summary()


#predictions to check the model
print("-----  Evaluating the model ----")
predictions = model.predict(X_train)
err_train = np.sqrt(mean_squared_error(y_train, predictions))
predictions_test = model.predict(X_test)
err_test = np.sqrt(mean_squared_error(y_test, predictions_test))


print(print_model)
print ("-------------")
print (f"RMSE on train data: {err_train}")
print (f"RMSE on test data: {err_test}")
