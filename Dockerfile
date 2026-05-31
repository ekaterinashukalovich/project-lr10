FROM python:3.11-slim

WORKDIR /app

# Устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код проекта
COPY . .

# Открываем порт
EXPOSE 8000

# Запуск
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]