# One-Class CT Chest Normality Detector

ИИ-сервис для выявления **нормы** на компьютерной томографии органов грудной клетки (КТ ОГК).  
Модель реализована в виде **one-class автоэнкодера** (TensorFlow / Keras), обучаемого только на нормальных исследованиях.  
Валидация используется для построения эмпирической CDF ошибки восстановления, что позволяет преобразовать её в вероятность нормы `p_norm`.  
Вероятность патологии:  
```
p_pathology = 1 - p_norm
```

---

## Возможности
- Поддержка входных данных в формате **DICOM** (ZIP-архивов).
- Проверка метаданных: **Modality == "CT"** и **BodyPartExamined == "CHEST"**.
- Классификация на два класса: «норма» (0) и «патология» (1).
- Формирование Excel-отчёта (.xlsx) с результатами:
  - `path_to_study`  
  - `study_uid`  
  - `series_uid`  
  - `probability_of_pathology`  
  - `pathology`  
  - `processing_status` (Success / Failure с причиной)  
  - `time_of_processing`
- Возможность локального запуска в контейнере Docker (без ручной установки зависимостей).
- Простой веб-интерфейс (Flask) для загрузки ZIP, анализа и скачивания отчёта.

---

## Ограничения
- Модель обучалась только на **исследованиях без патологии**. Всё иное относится к классу «патология».
- Поддерживаются **КТ ОГК без контрастного усиления**.

---

## Системные требования
- **Docker** ≥ 20.10 и **Docker Compose** ≥ 2.0  
- RAM: ≥ 8 ГБ  
- ОС: Linux / macOS / Windows (WSL2)  

---

## Структура проекта
```
project/
│
├── app.py               # Flask веб-приложение
├── train_oneclass.py    # обучение автоэнкодера
├── eval_model.py        # оценка качества
├── infer_zip.py         # инференс ZIP и Excel-отчёт
│
├── data_utils.py        # работа с DICOM
├── metrics_utils.py     # метрики, калибровка ECDF
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Быстрый старт (без Docker)

### Обучение
```bash
python train_oneclass.py   --train_dir data/train/normal   --val_normal_dir data/val/normal   --val_pathology_dir data/val/pathology   --out_dir artifacts   --img_size 256   --batch_size 16   --epochs 30   --tune_threshold --tune_mode youden
```

### Оценка
```bash
python eval_model.py   --val_normal_dir data/val/normal   --val_pathology_dir data/val/pathology   --test_normal_dir data/test/normal   --test_pathology_dir data/test/pathology   --artifacts_dir artifacts   --img_size 256
```

### Инференс ZIP-архивов
```bash
python infer_zip.py   --zip_paths data/test.zip   --artifacts_dir artifacts   --xlsx_out results.xlsx   --img_size 256
```

---

## Запуск через Docker

### 1. Сборка контейнера
В корне проекта (где лежит `Dockerfile` и `docker-compose.yml`):
```bash
docker compose build
```

### 2. Запуск контейнера
```bash
docker compose up -d
```

По умолчанию запустится Flask-приложение на [http://localhost:8000](http://localhost:8000).

### 3. Использование
- Загрузите ZIP-архив через веб-интерфейс.  
- После анализа отобразится краткий результат:
  - число обработанных архивов
  - число файлов/слайсов
  - количество норм / патологий
  - количество ошибок и причины
- Скачайте полный Excel-отчёт (`results.xlsx`).

### 4. Остановка контейнера
```bash
docker compose down
```

---

## Руководство пользователя
1. Запустите контейнер (см. выше).  
2. Перейдите в браузере по адресу `http://localhost:8000`.  
3. Загрузите архив с DICOM-файлами.  
4. Дождитесь окончания анализа.  
5. Скачайте Excel-отчёт с результатами.
