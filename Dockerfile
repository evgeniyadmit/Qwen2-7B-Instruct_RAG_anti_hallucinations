FROM cr.yandex/crp2q2b12lka2f8enigt/pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime


# ===== Рабочая директория =====
WORKDIR /workspace

# ===== Копируем весь репозиторий (код + веса в LFS) =====
COPY . /workspace

# ===== Оффлайн и стабильность =====
ENV TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1

# ===== Устанавливаем зависимости =====
# (requirements.txt должен лежать в корне репо)
RUN pip install --no-cache-dir -r requirements.txt

# ===== Запуск =====
CMD ["python", "solution.py"]
