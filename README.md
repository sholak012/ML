# ML: стилизация / продолжение текста на казахском языке

Проект по работе с корпусом «Сөздері Абая» (`abai-sozderi.json`): анализ данных, подготовка выборок, базовые модели (в т.ч. LSTM), трансформеры (ruGPT-3, mT5, ByT5) и оценка качества.

## Структура репозитория

| Файл / папка | Описание |
|--------------|----------|
| `abai-sozderi.json` | Исходный JSON-корпус |
| `data/` | Train/val/test CSV, признаки для LSTM (`lstm_X.npy`, `lstm_y.npy`), `tokenizer.pkl` |
| `lstm_baseline.keras` | Сохранённая baseline-модель Keras |
| `Project_Report.docx` | Отчёт по проекту |
| `1_data_analysis_preprocessing.ipynb` | Анализ данных, очистка, токенизация (BPE, Stanza и др.) |
| `2_data_augmentation.ipynb` | Аугментация данных |
| `3_baseline_models.ipynb` | Базовые модели |
| `4_transformer_models.ipynb` | Эксперименты с трансформерами |
| `5_evaluation_error_analysis.ipynb` | Оценка и разбор ошибок |
| `5_mT5_ByT5_experiments.ipynb` | Seq2Seq: mT5 / ByT5 (prefix → продолжение) |

## Окружение

Рекомендуется Python 3.10+ и виртуальное окружение:

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
```

Зависимости устанавливаются из первых ячеек ноутбуков (`pip install ...`) или вручную: `pandas`, `numpy`, `scikit-learn`, `torch`, `transformers`, `datasets`, `nltk`, `stanza`, `matplotlib`, `seaborn`, `jupyter`, при необходимости TensorFlow/Keras для baseline.

## Порядок работы

Запускайте ноутбуки по номерам `1` → `5` (модули 5 можно смотреть в удобном порядке: оценка и mT5/ByT5).

## Что не в Git

В `.gitignore` исключены:

- `.venv/` — локальное окружение
- крупные каталоги с чекпоинтами: `results_google-mt5-small/`, `results_rugpt3/`, `abai_rugpt3_final/`

Их можно хранить отдельно (диск, [Git LFS](https://git-lfs.com/), [Hugging Face Hub](https://huggingface.co/)) и при необходимости воспроизвести обучение по ноутбукам.

## Лицензия и данные

Уточните условия использования корпуса «Сөздері Абая» у правообладателя, если планируете публичное распространение производных работ.
