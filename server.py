from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import io
import uvicorn
import pickle
import re

app = FastAPI()

# Загрузка модели и вспомогательных объектов
with open('model_best.pkl', 'rb') as f:
    model = pickle.load(f)

with open('encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)

# Классы для валидации данных
class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: float
    engine: int
    max_power: float
    torque: float
    seats: float
    max_torque_rpm: float

class Items(BaseModel):
    objects: List[Item]

# Функции для обработки данных
def extract_first_two_words_regex(text):
    if not isinstance(text, str):
        return ""
    match = re.match(r'(\S+\s*\S*)', text.strip())
    return match.group(1) if match else ""

def extract_first_word_regex(text):
    if not isinstance(text, str):
        return ""
    match = re.match(r'(\S+\s*)', text.strip())
    return match.group(1) if match else ""

def process_dataframe(df, encoder):
    df['seats'] = df['seats'].astype('object')
    categorical_columns = ['name', 'fuel', 'seller_type', 'transmission', 'owner', 'seats']
    df['first_two_words'] = df['name'].apply(extract_first_two_words_regex)
    df['first_words'] = df['name'].apply(extract_first_word_regex)
    word_counts = df['first_two_words'].value_counts()
    df['choice'] = df['first_two_words'].apply(lambda x: word_counts[x] >= 50)
    df['name'] = df.apply(
        lambda row: row['first_two_words'] if row['choice'] else row['first_words'], axis=1
    )
    df = df.drop(columns=["first_two_words", "first_words", "choice"])
    X = encoder.transform(df[categorical_columns])
    encoded_columns = encoder.get_feature_names_out(categorical_columns)
    X_enc = pd.DataFrame(X, columns=encoded_columns, index=df.index)
    df_processed = pd.concat([df.drop(columns=categorical_columns), X_enc], axis=1)
    return df_processed

def model_predict(data):
    return model.predict(data)

# Методы API
@app.post("/predict_item")
def predict_item(item: Item):
    try:
        data = pd.DataFrame([item.dict()])
        X = process_dataframe(data, encoder)
        predictions = model_predict(X)
        return {"predicted_price": float(predictions[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in prediction: {e}")

@app.post("/predict_items")
def predict_items(items: Items):
    try:
        # Преобразование списка объектов Item в DataFrame
        data = pd.DataFrame([item.dict() for item in items.objects])
        X = process_dataframe(data, encoder)
        predictions = model_predict(X)
        # Возвращаем предсказания вместе с исходными данными
        data['predicted_price'] = predictions
        return data.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in batch prediction: {e}")

@app.post("/predict_batch")
def predict_batch(file: UploadFile = File(...)):
    try:
        # Чтение CSV-файла
        contents = file.file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        if 'selling_price' in df.columns:
            df = df.drop(columns=['selling_price'])
        
        X = process_dataframe(df.copy(), encoder)
        predictions = model_predict(X)
        
        # Добавление предсказаний в DataFrame
        df['predicted_price'] = predictions
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        return {"file": output.getvalue()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {e}")

if __name__ == "__main__":
    uvicorn.run("server:app", host="localhost", port=8000, reload=True)
