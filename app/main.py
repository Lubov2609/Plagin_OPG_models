import json
import shutil
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.model import DenseNetModel
from app.preprocessing import OPGPreprocessor
from app.predictor import OPGPredictor


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "opg_model.pth"
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="OPG Analysis Plugin",
    description="Плагин для анализа ортопантомограмм",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

preprocessor = OPGPreprocessor()

predictor = OPGPredictor(
    model_path=MODEL_PATH,
    model_class=DenseNetModel,
    preprocessor=preprocessor,
    threshold=0.5
)


def save_result(filename, result):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_name = Path(filename).stem.replace(" ", "_")
    result_file = RESULTS_DIR / f"{safe_name}_{timestamp}.json"

    data = {
        "filename": filename,
        "created_at": timestamp,
        "result": result
    }

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    return result_file.name


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "result": None,
            "filename": None,
            "image_url": None,
            "result_file": None
        }
    )

@app.post("/analyze-page", response_class=HTMLResponse)
async def analyze_page(request: Request, file: UploadFile = File(...)):
    try:
        file_path = UPLOAD_DIR / file.filename

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = predictor.predict(str(file_path))
        result_file = save_result(file.filename, result)

        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "filename": file.filename,
                "image_url": f"/uploads/{file.filename}",
                "result": result,
                "result_file": result_file
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "error": str(e),
                "result": None,
                "filename": None,
                "image_url": None,
                "result_file": None
            }
        )


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        file_path = UPLOAD_DIR / file.filename

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = predictor.predict(str(file_path))
        result_file = save_result(file.filename, result)

        return {
            "filename": file.filename,
            "result_file": result_file,
            "result": result
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )