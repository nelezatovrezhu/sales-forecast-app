import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Функция для генерации случайных данных
def generate_realistic_dataset(start_date='2000-01-01', end_date='2025-03-01', products=['Aluminium'], seed=42):
    np.random.seed(seed)
    
    # Создание диапазона дат
    dates = pd.date_range(start=start_date, end=end_date, freq='M')  # Ежемесячные данные
    
    # Генерация данных для каждого продукта
    data = []
    for product in products:
        sales_volume_prev = None  # Для создания трендов в продажах
        for date in dates:
            # Определение базовых значений с учетом экономических факторов
            year = date.year
            if year == 2020:  # Пандемия: снижение спроса
                sales_base = np.random.randint(80, 200)
                raw_material_price = np.random.uniform(60, 120)
                fuel_price = np.random.uniform(30, 50)
                exchange_rate = np.random.uniform(70, 80)
            elif year in [2022, 2023]:  # Санкции: высокая волатильность
                sales_base = np.random.randint(150, 300)
                raw_material_price = np.random.uniform(80, 200)
                fuel_price = np.random.uniform(40, 80)
                exchange_rate = np.random.uniform(80, 120)
            else:  # Другие годы: стабильный рост
                sales_base = np.random.randint(200, 400)
                raw_material_price = np.random.uniform(70, 150)
                fuel_price = np.random.uniform(35, 70)
                exchange_rate = np.random.uniform(60, 90)
            
            # Генерация объема продаж с трендом
            if sales_volume_prev is not None:
                sales_volume = max(sales_volume_prev + np.random.randint(-50, 50), 0)  # Легкий тренд
            else:
                sales_volume = sales_base
            sales_volume_prev = sales_volume
            
            # Генерация дополнительных признаков
            market_demand_index = np.random.randint(50, 100)
            production_capacity = np.random.randint(1500, 2000)
            
            # Сезонные факторы
            #month = date.month
            #is_holiday = 1 if month in [1, 5, 12] else 0  # Праздники: январь, май, декабрь
            #season = ['winter', 'spring', 'summer', 'fall'][(month - 1) // 3]
            
            # Лаговые переменные
            #sales_lag_1 = np.nan if len(data) < 1 else data[-1]['sales_volume'] if data[-1]['product_id'] == product else np.nan
            #sales_lag_3 = np.nan if len(data) < 3 else data[-3]['sales_volume'] if data[-3]['product_id'] == product else np.nan
            #sales_lag_6 = np.nan if len(data) < 6 else data[-6]['sales_volume'] if data[-6]['product_id'] == product else np.nan
            
            # Добавление случайных пропусков
            if np.random.rand() < 0.05:  # 5% вероятность пропуска данных
                raw_material_price = np.nan
            if np.random.rand() < 0.03:  # 3% вероятность пропуска данных
                fuel_price = np.nan
            if np.random.rand() < 0.02:  # 2% вероятность пропуска данных
                exchange_rate = np.nan
            
            # Добавление записи в датасет
            data.append({
                'date': date,
                'product_id': product,
                'sales_volume': sales_volume,
                'raw_material_price': raw_material_price,
                'fuel_price': fuel_price,
                'exchange_rate': exchange_rate,
                'market_demand_index': market_demand_index,
                'production_capacity': production_capacity,
                #'month': month,
                #'is_holiday': is_holiday,
                #'season': season,
                #'sales_lag_1': sales_lag_1,
                #'sales_lag_3': sales_lag_3,
                #'sales_lag_6': sales_lag_6
            })
    
    # Создание DataFrame
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])

    return df

# Генерация датасета
df = generate_realistic_dataset()


# Сохранение в CSV
df.to_csv('sales_forecasting_dataset2.csv', index=False)
    
# Вывод первых строк датасета
print(df.head())