import pandas as pd
import joblib

# Загрузка артефактов
model = joblib.load(r'best_lgb.pkl')
feature_names = joblib.load(r'feature_names.pkl')
X_inf = pd.read_parquet(r'X_inf_example.parquet')

# Возьмём первую строку как пример нового запроса
single_row = X_inf.iloc[[0]]  
X_input = single_row[feature_names]

# Предсказание
prediction = model.predict(X_input)
probability = model.predict_proba(X_input)[:, 1]

print(f"Предсказание: {prediction[0]}, Вероятность: {probability[0]:.4f}")