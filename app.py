from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel,Field
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib

model=joblib.load('hgb_model.pkl')
y_scaler=joblib.load('scaler_y.pkl')
X_scaler=joblib.load('scaler_X.pkl')

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    MedInc:float
    HouseAge:float
    AveRooms:float
    AveBedrms:float
    Population:float
    AveOccup:float


app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


@app.post('/predict')
def prediction(query:Query):
    try:
        input_data = pd.DataFrame([query.model_dump()])
        input_data=X_scaler.transform(input_data)
        pred=model.predict(input_data)
        pred.reshape(-1,1)
        pr={"price":pred}
        pr=pd.DataFrame([pr])
        pred=y_scaler.inverse_transform(pr)
        return {"prediction":pred[0][0]}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail='failure at our side'
        )


