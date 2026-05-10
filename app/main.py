import json
import shutil
from datetime import datetime
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.local_dataset import (
    count_labels,
    load_learning_settings,
    pending_label_count,
    save_doctor_label,
    should_start_training,
    update_train_trigger_count,
)
from app.local_training import train_local_model
from app.model import DenseNetModel
from app.preprocessing import OPGPreprocessor
from app.predictor import OPGPredictor


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "opg_model.pth"
LOCAL_MODEL_PATH = BASE_DIR / "models" / "opg_model_local.pth"
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
    version="1.1.0",
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

preprocessor = OPGPreprocessor()

predictor = OPGPredictor(
    model_path=MODEL_PATH,
    local_model_path=LOCAL_MODEL_PATH,
    model_class=DenseNetModel,
    preprocessor=preprocessor,
    threshold=0.5,
)


def save_result(filename, result):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_name = Path(filename).stem.replace(" ", "_")
    result_file = RESULTS_DIR / f"{safe_name}_{timestamp}.json"

    data = {
        "filename": filename,
        "created_at": timestamp,
        "result": result,
        "model_source": predictor.model_source,
        "model_path": str(predictor.active_model_path),
    }

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    return result_file.name


def run_local_training_job():
    result = train_local_model(
        base_dir=BASE_DIR,
        model_class=DenseNetModel,
        base_model_path=MODEL_PATH,
        local_model_path=LOCAL_MODEL_PATH,
        epochs=3,
        batch_size=8,
    )

    if result.get("status") == "success":
        predictor.reload_model()


def learning_context():
    settings = load_learning_settings(BASE_DIR)
    return {
        "learning_settings": settings,
        "dataset_count": count_labels(BASE_DIR),
        "pending_count": pending_label_count(BASE_DIR),
        "model_status": predictor.get_model_status(),
        "active_model_path": predictor.active_model_path,
        "local_model_exists": LOCAL_MODEL_PATH.exists(),
    }


def page_context(**kwargs):
    context = {
        "result": None,
        "filename": None,
        "image_url": None,
        "result_file": None,
        "error": None,
        "label_message": None,
        "training_message": None,
    }
    context.update(learning_context())
    context.update(kwargs)
    return context


def safe_upload_path(filename):
    return UPLOAD_DIR / Path(filename).name


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        page_context(),
    )


@app.post("/analyze-page", response_class=HTMLResponse)
async def analyze_page(request: Request, file: UploadFile = File(...)):
    try:
        file_path = safe_upload_path(file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = predictor.predict(str(file_path))
        result_file = save_result(file.filename, result)

        return templates.TemplateResponse(
            request,
            "index.html",
            page_context(
                filename=file_path.name,
                image_url=f"/uploads/{file_path.name}",
                result=result,
                result_file=result_file,
            ),
        )
    except Exception as e:
        return templates.TemplateResponse(
            request,
            "index.html",
            page_context(error=str(e)),
        )


@app.post("/doctor-label", response_class=HTMLResponse)
async def doctor_label(
    request: Request,
    background_tasks: BackgroundTasks,
    image_name: str = Form(...),
    caries: int = Form(...),
    pulpitis: int = Form(...),
    wisdom_teeth_count: int = Form(...),
    impacted_wisdom_teeth_count: int = Form(...),
    doctor_comment: str = Form(""),
    train_trigger_count: int = Form(8),
):
    try:
        image_path = safe_upload_path(image_name)
        update_train_trigger_count(BASE_DIR, train_trigger_count)

        saved = save_doctor_label(
            image_path=image_path,
            base_dir=BASE_DIR,
            caries=caries,
            pulpitis=pulpitis,
            wisdom_teeth_count=wisdom_teeth_count,
            impacted_wisdom_teeth_count=impacted_wisdom_teeth_count,
            doctor_comment=doctor_comment,
        )

        training_message = (
            f"Разметка сохранена. В локальном датасете: {saved['label_count']} снимков. "
            f"До следующего запуска дообучения: "
            f"{max(0, train_trigger_count - saved['pending_count'])}."
        )

        if should_start_training(BASE_DIR):
            background_tasks.add_task(run_local_training_job)
            training_message = "Разметка сохранена. Локальное дообучение запущено в фоне."

        result = predictor.predict(str(image_path))

        return templates.TemplateResponse(
            request,
            "index.html",
            page_context(
                filename=image_path.name,
                image_url=f"/uploads/{image_path.name}",
                result=result,
                label_message="Врачебная разметка сохранена в локальный датасет.",
                training_message=training_message,
            ),
        )
    except Exception as e:
        return templates.TemplateResponse(
            request,
            "index.html",
            page_context(error=str(e)),
        )


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        file_path = safe_upload_path(file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = predictor.predict(str(file_path))
        result_file = save_result(file.filename, result)

        return {
            "filename": file_path.name,
            "result_file": result_file,
            "model_source": predictor.model_source,
            "result": result,
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )
