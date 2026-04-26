сделать .exe

pip install pyinstaller
pyinstaller --onefile --noconsole --name OPG_Plugin --paths . --hidden-import app.model --hidden-import app.preprocessing --hidden-import app.predictor app\desktop_watcher.py      
запуск через терминал активация папки
python -m app.desktop_watcher

запуск сайта
uvicorn app.main:app --reload