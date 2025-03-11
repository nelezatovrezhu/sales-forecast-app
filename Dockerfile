# Используем официальный образ Python 3.9
FROM python:3.9-slim

# Обновляем системные зависимости и утилиты
RUN apt-get update && apt-get install -y \
    libgl1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Обновляем pip до последней версии
RUN python -m pip install --upgrade pip

# Создаем рабочую директорию
WORKDIR /app

# Устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Создаем необходимые папки в контейнере
RUN mkdir -p /app/api/data /app/api/models /app/api/static /app/api/templates

# Копируем проект в контейнер
COPY . /app

# Настройка среды
ENV FLASK_APP=api/app.py
ENV FLASK_ENV=development

# Опубликованный порт
EXPOSE 5000

# Запуск приложения
CMD ["python", "./api/app.py"]