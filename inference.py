import pickle
import pandas as pd
import numpy as np 

# Загрузка модели из файла pickle
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

import pandas as pd

# Новые данные
new_data = pd.DataFrame({
    'time_since_last_liquidated': [np.random.randint(0, 1000)],      
    'time_since_first_deposit': [np.random.randint(100, 5000)],
    'total_available_borrows_eth': [np.random.uniform(0.1, 50.0)],   
    'borrow_timestamp': [np.random.randint(1600000000, 1700000000)], 
    'wallet_age': [np.random.randint(1000, 10000)],                  
    'min_eth_ever': [np.random.uniform(0.01, 5.0)],
    'max_eth_ever': [np.random.uniform(10.0, 100.0)],
    'total_collateral_eth': [np.random.uniform(1.0, 80.0)]
})

# Предсказание
predictions = model.predict(new_data)

# Вывод результатов
print(predictions)