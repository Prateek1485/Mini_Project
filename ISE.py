# Bike Rental Prediction using Random Forest + Feature Importance Graph

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("Sales.csv")


df_bikes = df[df["Product_Category"] == "Bikes"].copy()


features = [
    "Year", "Month", "Customer_Age", "Customer_Gender",
    "Country", "State", "Unit_Cost", "Unit_Price",
    "Profit", "Revenue"
]
target = "Order_Quantity"


cat_cols = df_bikes[features].select_dtypes(include=["object"]).columns
for col in cat_cols:
    df_bikes[col] = LabelEncoder().fit_transform(df_bikes[col])


X = df_bikes[features]
y = df_bikes[target]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n🚴 Bike Rental Prediction (Bikes Category)")
print("------------------------------------------")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R² Score: {r2:.3f}")


importances = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)


print("\n🔥 Top 5 Features Influencing Bike Rentals:")
for i, row in importances.head(5).iterrows():
    print(f"{row['Feature']}: {row['Importance']:.3f}")


plt.figure(figsize=(10, 6))
plt.barh(importances["Feature"], importances["Importance"], color="skyblue")
plt.gca().invert_yaxis()  
plt.title("Feature Importance in Bike Rental Prediction", fontsize=14)
plt.xlabel("Importance", fontsize=12)
plt.ylabel("Features", fontsize=12)
plt.tight_layout()
plt.show()
