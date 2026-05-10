## Запуск сайта

```bash
uvicorn app.main:app --reload
```

## Запуск desktop-приложения

```bash
python -m app.desktop_watcher
```

## Сборка OPG_Plugin.exe

```bash
pip install pyinstaller
pyinstaller --onefile --noconsole --name OPG_Plugin --paths . --hidden-import app.model --hidden-import app.preprocessing --hidden-import app.predictor --hidden-import app.local_dataset --hidden-import app.local_training app\desktop_watcher.py
```

## Локальное обучение

Врачебная разметка сохраняется в папку `local_dataset`.

Файл `local_dataset/labels.csv` имеет такой же формат, как `data/label.csv` в обучающем проекте:

```text
image,caries,pulpitis,wisdom_teeth_count,impacted_wisdom_teeth_count
```

Заголовок в файл не записывается. Значения `Да/Нет` из формы сохраняются как `1/0`.

Локально дообученная модель сохраняется отдельно:

```text
models/opg_model_local.pth
```

Исходная модель `models/opg_model.pth` не изменяется. Если локальная модель существует и успешно загружается, сайт и desktop-приложение используют ее автоматически.
