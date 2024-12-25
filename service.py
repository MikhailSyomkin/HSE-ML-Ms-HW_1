from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd
from io import StringIO

app = FastAPI()

model = joblib.load("ridge_model.pkl")

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float

class Items(BaseModel):
    objects: List[Item]

@app.post("/predict_item")
def predict_item(item: Item) -> float:
    values = [[item.name, item.year, item.selling_price, item.km_driven, item.fuel, item.seller_type,
               item.transmission, item.owner, item.mileage, item.engine, item.max_power, item.torque, item.seats]]
    return model.predict(values)[0]

@app.post("/predict_items")
def predict_items(items: Items) -> List[float]:
    values = [[item.name, item.year, item.selling_price, item.km_driven, item.fuel, item.seller_type,
               item.transmission, item.owner, item.mileage, item.engine, item.max_power,
               item.torque, item.seats] for item in items.objects]
    predictions = model.predict(values)
    return predictions.tolist()

@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(StringIO(content.decode("utf-8")))
    features = df[["name", "year", "selling_price", "km_driven", "fuel", "seller_type", "transmission", 
                   "owner", "mileage", "engine", "max_power", "torque", "seats"]].values
    predictions = model.predict(features)
    df['predicted_price'] = predictions
    output_file = "predicted_prices.csv"
    df.to_csv(output_file, index=False)
    return {"file_path": output_file}
