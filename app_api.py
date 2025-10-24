from fastapi import FastAPI, Request, HTTPException
import pickle
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

# Загрузка модели из файла pickle
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Счетчик запросов
request_count = 0

# Модель для валидации входных данных
class PredictionInput(BaseModel):
    Pclass: int
    Age: float
    Fare: float

@app.get("/stats")
def stats():
    return {"request_count": request_count}

@app.get("/health")
def health():
    return {"status": "OK"}

@app.post("/predict_model")
def predict_model(input_data: PredictionInput):
    global request_count
    request_count += 1

    # Создание DataFrame из данных
    new_data = pd.DataFrame({
        'Pclass': [input_data.Pclass],
        'Age': [input_data.Age],
        'Fare': [input_data.Fare]
    })

    # Предсказание
    predictions = model.predict(new_data)



    return {"prediction": predictions}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)