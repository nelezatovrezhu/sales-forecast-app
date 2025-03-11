import requests
import os
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
url = 'http://127.0.0.1:5000/predict'

# Загрузка обучающего датасета для генерации даты и признаков
train_df = pd.read_csv(os.path.join(project_root, 'data', 'sales_forecasting_dataset.csv'))
train_df = train_df[train_df['product_id'] == 'Aluminium'].drop(columns=['product_id'])
train_df['date'] = pd.to_datetime(train_df['date'])
last_date = train_df['date'].max()

# Генерация даты для прогноза
next_month_date = last_date + pd.DateOffset(months=1)

# Используем последние значения признаков из train_df
sample_row = train_df.iloc[-1][[
    'raw_material_price', 'fuel_price', 'exchange_rate', 
    'market_demand_index', 'production_capacity'
]].to_dict()

# Добавляем дату следующего месяца
sample_row['date'] = next_month_date.strftime("%Y-%m-%d")

# Тестовые данные
data = [sample_row]

response = requests.post(url, json=data)

if response.status_code == 200:
    print("Прогноз:", response.json()['predictions'][0])
    print("Последнее значение в данных:", train_df['sales_volume'].iloc[-1])
else:
    print("Ошибка:", response.json())