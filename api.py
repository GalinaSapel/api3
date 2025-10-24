from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel, create_model
import os

app = FastAPI()

# === Загрузка артефактов ===
MODEL_PATH = "best_lgb.pkl"
FEATURES_PATH = "feature_names.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
    raise FileNotFoundError("Модель или список признаков не найдены. Убедитесь, что файлы лежат в корне проекта.")

model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)  # должен быть список строк

# === Динамическое создание Pydantic-модели на основе feature_names ===
# Все признаки — float (можно адаптировать под int/bool при необходимости)
PredictionInput = create_model(
    "PredictionInput",
    **{feat: (float, ...) for feat in feature_names}
)

# === Счётчик запросов ===
request_count = 0

# === Эндпоинты ===
@app.get("/health")
def health():
    return {"status": "OK", "model_loaded": True, "n_features": len(feature_names)}

@app.get("/stats")
def stats():
    return {"request_count": request_count}

@app.post("/predict")
def predict(input_data: PredictionInput):
    global request_count
    request_count += 1

    try:
        # Преобразуем входные данные в DataFrame с правильным порядком признаков
        input_dict = input_data.model_dump()  # или .dict() в старых версиях Pydantic
        df = pd.DataFrame([input_dict])
        df = df[feature_names]  # гарантирует правильный порядок

        # Предсказание
        pred = model.predict(df)[0]
        proba = model.predict_proba(df)[0].max() if hasattr(model, "predict_proba") else None

        result = "Default" if pred == 1 else "No Default"

        return {
            "prediction": result,
            "probability": float(proba) if proba is not None else None,
            "request_id": request_count
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка предсказания: {str(e)}")

# === Запуск (для локального теста) ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)