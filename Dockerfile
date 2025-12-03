FROM cr.yandex/crp2q2b12lka2f8enigt/pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime

WORKDIR /app

# Установка зависимостей
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Копируем проект
COPY . /app

# Создаём папки для моделей и выходных логов
RUN mkdir -p /app/models /app/outputs

# CMD запускает основной файл
CMD ["python", "solution.py"]
