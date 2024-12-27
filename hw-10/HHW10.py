import uvicorn

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

app=FastAPI()

data=pd.read_csv("realty_data.csv")
data=data[["price","rooms","floor"]]
data=data.dropna()

X= data[["rooms","floor"]]
y=data["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2024, test_size=0.25)

dummy = DummyRegressor(strategy="mean")
dummy.fit(X_train, y_train);
y_pred_dummy = dummy.predict(X_test)

model=LinearRegression()
model.fit(X_train,y_train)
y_pred_lr = model.predict(X_test)

class ModelRequestData(BaseModel):
    rooms: int
    floor: int

#class Result(BaseModel):
    #result: float

@app.get("/health")
def health():
    return JSONResponse(content={"message": "alive"}, status_code=200)

@app.get("/predict_get")
def predict_get(rooms:int, floor: int):
        prediction = model.predict([[rooms,floor]])[0]
        return {"prediction":prediction}

@app.post("/predict_post")
def predict_post(request: ModelRequestData):
        prediction = model.predict([[request.rooms,request.floor]])[0]
        return {"prediction":prediction}

if __name__ == "__hw10__":
    uvicorn.run(app,host="0.0.0.0",port=8000)


