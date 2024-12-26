import os
from math import floor

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Прогнозирование стоимости недвижимости",
)

data=pd.read_csv(".venv/realty_data.csv")
print(data.head())

data=data[["price","rooms","floor"]]
data=data.dropna()
print(data.info())

from sklearn.model_selection import train_test_split
X= data[["rooms","floor"]]
y=data["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2024, test_size=0.25)
print(f"Обущающая выборка:{X_train.shape}, Тестова выборка: {X_test.shape}")

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error

dummy = DummyRegressor(strategy="mean")
dummy.fit(X_train, y_train);
y_pred_dummy = dummy.predict(X_test)

print("DummyRegressor:")
print(f"MAE: {mean_absolute_error(y_test,y_pred_dummy)}")
print(f"MSE:{mean_squared_error(y_test,y_pred_dummy)}")


model=LinearRegression()
model.fit(X_train,y_train)
y_pred_lr = model.predict(X_test)

print("LinearRegression:")
print(f"MAE: {mean_absolute_error(y_test,y_pred_lr)}")
print(f"MSE:{mean_squared_error(y_test,y_pred_lr)}")

rooms=st.sidebar.number_input("Количество комнат",1,6,1)
floor=st.sidebar.number_input("Этаж",1,43,1)

if st.button("Спрогнозировать"):
    new_data= pd.DataFrame({"rooms":[rooms], "floor": [floor]})
    predicted_price = model.predict(new_data)
    st.write(f"Предсказанная стоимость недвижимости: {predicted_price[0]:.2f} руб.")

