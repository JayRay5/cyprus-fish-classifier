import io
import torch
import uvicorn
import asyncio
import gradio as gr
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
from .config import settings
from .utils import predict_image
from .ui import create_ui


constants = {}
model_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Init & Load model/processor
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        constants["processor"] = AutoImageProcessor.from_pretrained(
            settings.huggingface_repo_id
        )  # nosec B615

        constants["model"] = AutoModelForImageClassification.from_pretrained(
            settings.huggingface_repo_id
        ).to(device)  # nosec B615

        constants["device"] = constants["model"].device

        print("[INFO] Model & Processor loaded. [INFO]")

    except Exception as e:
        print(f"Error Message : {e}")
        raise e

    yield  # API running

    # when API is stopped
    constants.clear()
    print("[INFO] API stopped. [INFO]")


# --- Start the API ---
app = FastAPI(
    title="Cyprus Fish Recognition API",
    description="API for classifying fish species using ConvNext Tiny",
    version="1.0.0",
    lifespan=lifespan,
)

# --- ENDPOINTS ---


@app.get("/health_check")
def health_check():
    return {"status": "ok", "model_loaded": "model" in constants}


@app.get("/model_device")
def model_device_check():
    if "device" not in constants:
        if "model" not in constants:
            raise HTTPException(status_code=503, detail="Model not found.")
        model = constants["model"]
        try:
            device = model.device.type
        except Exception:
            device = "unknown"
    else:
        device = constants["device"].type

    return {"status": "ready", "device": device}


@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    if "model" not in constants:
        raise HTTPException(status_code=503, detail="Model not found.")

    if "processor" not in constants:
        raise HTTPException(status_code=503, detail="Processor not found.")

    if "device" not in constants:
        raise HTTPException(status_code=503, detail="Device not found.")

    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(
            status_code=400, detail="Unsupported format, require JPG or PNG."
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        model = constants["model"]
        processor = constants["processor"]

        # authorize one thread at a time
        async with model_lock:
            results = await run_in_threadpool(predict_image, image, processor, model)

        return results

    except Exception as e:
        print(f"Error during model prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


interface = create_ui(model_context=constants, model_lock=model_lock)

app = gr.mount_gradio_app(app, interface, path="/")


def start():
    uvicorn.run("src.app.api:app", host="0.0.0.0", port=8000, reload=True)  # nosec B104 (to ignore the bandit alert)


if __name__ == "__main__":
    start()
