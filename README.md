<<<<<<< HEAD
# Sales Forecast App

Это веб-приложение для прогнозирования объема продаж с использованием машинного обучения.

## Требования

- Python 3.9+
- Docker (опционально)

## Установка

### Локальная установка

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/your-username/sales-forecast-app.git
   cd sales-forecast-app

2. Установите зависимости:
    pip install -r requirements.txt

3. Запустите приложение:
    python api/app.py

Приложение будет доступно по адресу: http://localhost:5000

#### Использование Docker

1. Соберите Docker-образ:

    docker build -t sales-forecast-app .
2. Запустите контейнер:

    docker run -d -p 5000:5000 --name sales-forecast-container sales-forecast-app

Приложение будет доступно по адресу: http://localhost:5000

Как использовать
Загрузите исторические данные в формате CSV (пример файла: example.csv).
Нажмите кнопку "Обучить"
Выберите сценарий прогноза или введите собственные данные.
Нажмите кнопку "Предсказать" для получения прогноза.
Лицензия
