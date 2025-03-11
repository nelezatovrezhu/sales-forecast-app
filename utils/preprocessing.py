import pandas as pd
import numpy as np

def preprocess_data(df, train_df=None, drop_sales_volume=True):
    try:
        print("Предобработка данных начата")
        
        # Преобразование даты
        if 'date' not in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Столбец 'date' отсутствует или индекс не является датой!")
        
        # Если 'date' в столбцах — преобразуем и устанавливаем как индекс
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
            df = df.dropna(subset=['date']).reset_index(drop=True)
            df.set_index('date', inplace=True)
        
        # Проверяем, что индекс является DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        print("Дата преобразована и обработана")
        
        # Проверка обязательных столбцов
        required_num_cols = ['raw_material_price', 'fuel_price', 'exchange_rate', 
                            'market_demand_index', 'production_capacity']
        missing_cols = [col for col in required_num_cols if col not in df.columns]
        if missing_cols:
            raise KeyError(f"Отсутствуют обязательные признаки: {', '.join(missing_cols)}")
        
        # Обработка NaN в числовых признаках
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        print("NaN значения обработаны")
        
        # Лаговые переменные и скользящие средние
        has_sales_volume = 'sales_volume' in df.columns
        
        if has_sales_volume:
            for lag in [1, 3, 6]:
                df[f'sales_lag_{lag}'] = df['sales_volume'].shift(lag).fillna(0)
            
            df['rolling_mean_3'] = df['sales_volume'].rolling(window=3).mean().ffill().fillna(0)
            df['rolling_mean_6'] = df['sales_volume'].rolling(window=6).mean().ffill().fillna(0)
            
            df['sales_diff_1'] = df['sales_volume'].diff(1).fillna(0)
            df['sales_diff_3'] = df['sales_volume'].diff(3).fillna(0)
            df['sales_diff_6'] = df['sales_volume'].diff(6).fillna(0)
            print("Лаговые переменные и скользящие средние добавлены")
        else:
            if train_df is not None and not train_df.empty:
                for lag in [1, 3, 6]:
                    last_sales = train_df['sales_volume'].iloc[-lag] if len(train_df) >= lag else 0
                    df[f'sales_lag_{lag}'] = last_sales
                
                df['rolling_mean_3'] = train_df['sales_volume'].iloc[-3:].mean() if len(train_df) >=3 else 0
                df['rolling_mean_6'] = train_df['sales_volume'].iloc[-6:].mean() if len(train_df) >=6 else 0
                
                df['sales_diff_1'] = train_df['sales_volume'].iloc[-1] - train_df['sales_volume'].iloc[-2] if len(train_df) >=2 else 0
                df['sales_diff_3'] = train_df['sales_volume'].iloc[-1] - train_df['sales_volume'].iloc[-4] if len(train_df) >=4 else 0
                df['sales_diff_6'] = train_df['sales_volume'].iloc[-1] - train_df['sales_volume'].iloc[-7] if len(train_df) >=7 else 0
            else:
                df['sales_lag_1'] = 0
                df['sales_lag_3'] = 0
                df['sales_lag_6'] = 0
                df['rolling_mean_3'] = 0
                df['rolling_mean_6'] = 0
                df['sales_diff_1'] = 0
                df['sales_diff_3'] = 0
                df['sales_diff_6'] = 0
            print("Лаговые переменные добавлены для прогноза")
        
        # Сезонность
        df['season'] = df.index.month.map({
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'fall', 10: 'fall', 11: 'fall'
        })
        # Явное указание всех категорий сезонов
        all_seasons = ['winter', 'spring', 'summer', 'fall']
        df = pd.get_dummies(df, columns=['season'], drop_first=True, dtype=int)
        # Добавление отсутствующих категорий
        for season in all_seasons:
            season_col = f'season_{season}'
            if season_col not in df.columns:
                df[season_col] = 0
        
        # Если это прогноз, добавляем все сезоны из train_df
        if train_df is not None and not train_df.empty:
            train_seasons = [col for col in train_df.columns if col.startswith('season_')]
            for season in train_seasons:
                if season not in df.columns:
                    df[season] = 0
        
        print("Сезонность добавлена")
        
        # Создаем признак month
        df['month'] = df.index.month
        
        # Признак праздника
        holidays = [1, 5, 12]
        df['is_holiday'] = df.index.month.isin(holidays).astype(int)
        print("Признак праздника добавлен")
        
        # Interaction features
        df['price_demand_ratio'] = df['raw_material_price'] / df['market_demand_index'].replace(0, 1)
        df['fuel_production_ratio'] = df['fuel_price'] / df['production_capacity'].replace(0, 1)
        print("Интерактивные признаки добавлены")
        
        # Удаление ненужных столбцов
        if 'product_id' in df.columns:
            df = df.drop(columns=['product_id'], errors='ignore')
        print("Ненужные столбцы удалены")
        
        # Удаление 'sales_volume' только при необходимости
        if drop_sales_volume and 'sales_volume' in df.columns:
            df = df.drop(columns=['sales_volume'], errors='ignore')
        print("Столбец 'sales_volume' удален при необходимости")
        
        # Проверка на NaN
        if df.isnull().sum().sum() > 0:
            df.fillna(0, inplace=True)
        print("NaN значения заполнены")
        
        return df
    except Exception as e:
        print(f"Ошибка предобработки: {e}")
        return None