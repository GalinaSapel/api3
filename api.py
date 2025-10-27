from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel, create_model
import os

app = FastAPI()

# === Загрузка артефактов ===
MODEL_PATH = "best_lgb.pkl"
FEATURES_PATH = "feature_names.pkl"
EXAMPLE_DATA_PATH = "X_inf_example.parquet"

if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
    raise FileNotFoundError("Модель или список признаков не найдены. Убедитесь, что файлы лежат в корне проекта.")

model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)  # должен быть список строк

# === Динамическое создание Pydantic-модели на основе feature_names ===
PredictionInput = create_model(
    "PredictionInput",
    **{feat: (float, ...) for feat in feature_names}
)

# === Счётчик запросов ===
request_count = 0

def make_prediction(input_dict: dict) -> int:
    df = pd.DataFrame([input_dict])[feature_names]
    pred = model.predict(df)
    return int(pred[0])

# === Эндпоинты ===
@app.get("/health")
def health():
    return {"status": "OK", "model_loaded": True, "n_features": len(feature_names)}

@app.get("/stats")
def stats():
    return {"request_count": request_count}

@app.get("/predict_random")
def predict_random():
    global request_count
    request_count += 1

    try:
        df_example = pd.read_parquet(EXAMPLE_DATA_PATH)
        missing = set(feature_names) - set(df_example.columns)
        if missing:
            raise ValueError(f"В parquet-файле отсутствуют признаки: {missing}")

        random_row = df_example.sample(n=1).iloc[0].to_dict()
        input_for_model = {feat: random_row[feat] for feat in feature_names}
        prediction = make_prediction(input_for_model)

        return {
            "input": input_for_model,
            "prediction": prediction
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка в predict_random: {str(e)}")

# === Запуск (для локального теста) ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000)