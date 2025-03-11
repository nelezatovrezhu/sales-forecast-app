import os
import sys
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from utils.preprocessing import preprocess_data
from utils.model_utils import train_model, save_model_and_scaler

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Обязательно для работы сессий

model = None
scaler = None
required_features = None
train_df = None
training_message = None  # Сообщение о результате обучения
original_df = None  # Для сохранения исходных данных

# Инициализация сессии
@app.before_request
def initialize_session():
    if 'predictions' not in session:
        session['predictions'] = []

@app.route('/', methods=['GET', 'POST'])
def index():
    global model, scaler, required_features, train_df, training_message, original_df

    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            return "No file uploaded", 400
        
        try:
            df = pd.read_csv(file, parse_dates=['date'], date_format='%Y-%m-%d')
            
            if 'date' not in df.columns:
                return "Столбец 'date' отсутствует в файле", 400
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                return "Столбец 'date' должен быть в формате YYYY-MM-DD", 400
            
            required_columns = ['date', 'raw_material_price', 'fuel_price', 
                               'exchange_rate', 'market_demand_index', 'production_capacity', 'sales_volume']
            missing = [col for col in required_columns if col not in df.columns]
            if missing:
                return f"Отсутствуют столбцы: {', '.join(missing)}", 400
            
            df_processed = preprocess_data(df, drop_sales_volume=False)
            if df_processed is None or df_processed.empty:
                return "Ошибка предобработки данных", 400
            
            X = df_processed.drop('sales_volume', axis=1)
            y = df_processed['sales_volume']
            if X.empty or y.empty:
                return "Ошибка при создании X/y", 400

            model, scaler = train_model(X, y)
            required_features = X.columns.tolist()
            train_df = df_processed
            original_df = df.copy()

            save_model_and_scaler(model, scaler, project_root)

            # Генерация графика
            plt.figure(figsize=(12, 6))
            plt.plot(original_df['date'], original_df['sales_volume'], label='Исторические данные', color='blue')
            
            # Добавляем прогнозы из сессии
            session_predictions = session.get('predictions', [])
            if session_predictions:
                forecast_dates = [pd.to_datetime(pred['date']) for pred in session_predictions]
                forecast_values = [pred['prediction'] for pred in session_predictions]
                plt.plot(forecast_dates, forecast_values, 'r--', marker='o', label='Прогноз', color='red')
            
            plt.title('Продажи и прогноз')
            plt.xlabel('Дата')
            plt.ylabel('Объем продаж')
            plt.legend()
            plt.grid(True)
            
            img = BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()
            plt.close()

            training_message = "Модель успешно обучена"
            
            # Очищаем прогнозы после отображения
            predictions_to_display = session.pop('predictions', [])
            
            return render_template('index.html', 
                                  plot_url=plot_url,
                                  model_ready=True,
                                  training_message=training_message,
                                  predictions=predictions_to_display)
        
        except Exception as e:
            training_message = f"Ошибка при обучении модели: {str(e)}"
            return render_template('index.html', 
                                  model_ready=False,
                                  training_message=training_message,
                                  predictions=session.get('predictions', [])), 500

    # GET-запрос
    plot_url = None
    if original_df is not None and not original_df.empty:
        plt.figure(figsize=(12, 6))
        plt.plot(original_df['date'], original_df['sales_volume'], label='Исторические данные', color='blue')
        
        session_predictions = session.get('predictions', [])
        if session_predictions:
            forecast_dates = [pd.to_datetime(pred['date']) for pred in session_predictions]
            forecast_values = [pred['prediction'] for pred in session_predictions]
            plt.plot(forecast_dates, forecast_values, 'r--', marker='o', label='Прогноз', color='red')
        
        plt.title('Продажи и прогноз')
        plt.xlabel('Дата')
        plt.ylabel('Объем продаж')
        plt.legend()
        plt.grid(True)
        
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

    # Очищаем прогнозы после отображения
    predictions_to_display = session.pop('predictions', [])

    return render_template('index.html', 
                          plot_url=plot_url,
                          model_ready=(model is not None),
                          training_message=training_message,
                          predictions=predictions_to_display)

@app.route('/predict-scenario', methods=['POST'])
def predict_scenario():
    global model, scaler, train_df, required_features

    if not model or train_df is None or train_df.empty:
        return jsonify({'error': 'Модель не обучена или данные не загружены'}), 400

    scenario = request.form.get('scenario', 'average')

    try:

        latest_date = pd.Timestamp(train_df.index.max())
        next_date = (latest_date.to_period('M') + 1).to_timestamp(how='end')


        base_values = {
            'raw_material_price': train_df['raw_material_price'].median(),
            'fuel_price': train_df['fuel_price'].median(),
            'exchange_rate': train_df['exchange_rate'].median(),
            'market_demand_index': train_df['market_demand_index'].median(),
            'production_capacity': train_df['production_capacity'].median(),
        }

        if scenario == 'optimistic':
            base_values['raw_material_price'] *= 0.8
            base_values['production_capacity'] *= 1.2
        elif scenario == 'pessimistic':
            base_values['raw_material_price'] *= 1.2
            base_values['exchange_rate'] *= 1.2

        scenario_data = pd.DataFrame([{
            'date': next_date,
            **base_values
        }])
        scenario_data.set_index('date', inplace=True)

        input_data = preprocess_data(scenario_data, train_df=train_df, drop_sales_volume=True)
        if input_data is None or input_data.empty:
            return jsonify({'error': 'Ошибка предобработки данных'}), 500

        missing_features = [f for f in required_features if f not in input_data.columns]
        if missing_features:
            return jsonify({'error': f"Отсутствуют признаки: {', '.join(missing_features)}"}), 400

        X = input_data[required_features]
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)[0]

        # Сохраняем прогноз в сессии
        session_predictions = session.get('predictions', [])
        session_predictions.append({
            'scenario': scenario,
            'date': next_date.strftime('%Y-%m-%d'),
            'prediction': round(prediction, 2)
        })
        session['predictions'] = session_predictions

        return redirect(url_for('index'))

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict_custom():
    global model, scaler, required_features, train_df

    if not model:
        return jsonify({'error': 'Модель не обучена'}), 400

    try:
        data = {
            'raw_material_price': float(request.form['raw_material_price']),
            'fuel_price': float(request.form['fuel_price']),
            'exchange_rate': float(request.form['exchange_rate']),
            'market_demand_index': float(request.form['market_demand_index']),
            'production_capacity': float(request.form['production_capacity'])
        }
        
        # Автоматически определяем следующую дату
        latest_date = train_df.index.max()
        next_date = latest_date + pd.DateOffset(months=1)
        data['date'] = next_date.strftime('%Y-%m-%d')  # Добавляем дату в данные
        
        input_data = pd.DataFrame([data])
        
        # Предобработка
        input_data = preprocess_data(input_data, train_df=train_df, drop_sales_volume=True)
        input_data = input_data[required_features]
        
        # Прогноз
        X_scaled = scaler.transform(input_data)
        prediction = model.predict(X_scaled)[0]

        # Сохраняем прогноз в сессии
        session_predictions = session.get('predictions', [])
        session_predictions.append({
            'scenario': 'custom',
            'date': next_date.strftime('%Y-%m-%d'),
            'prediction': round(prediction, 2)
        })
        session['predictions'] = session_predictions

        return redirect(url_for('index'))
    
    except Exception as e:
        return jsonify({'error': f'Ошибка прогноза: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)