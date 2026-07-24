import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("train.csv")

features = ["GrLivArea", "BedroomAbvGr", "FullBath"]
target = "SalePrice"

df = df[features + [target]].dropna()

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("Coefficients:", dict(zip(features, model.coef_)))
print("Intercept:", model.intercept_)
print("RMSE:", rmse)
print("R2 Score:", r2)
