"""
FastAPI web application for animal statement verification.

Endpoints:
    GET  /          → serve index.html
    POST /verify    → multipart: image file + text field → JSON verdict
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

import onnxruntime as ort
import spacy
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.classifier.infer_classifier_onnx import classify_image
from src.ner.infer_ner import extract_from_nlp

logger = logging.getLogger(__name__)

CLASSIFIER_MODEL = "models/classifier/classifier_model.onnx"
NER_MODEL = "models/ner/custom_ner_model"
STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once at startup; release on shutdown."""
    logger.info("Loading classifier ONNX session...")
    app.state.ort_session = ort.InferenceSession(CLASSIFIER_MODEL)
    logger.info("Loading spaCy NER model...")
    app.state.nlp = spacy.load(NER_MODEL)
    logger.info("Models ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(title="Animal Verifier", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/verify")
async def verify(image: UploadFile = File(...), text: str = Form(...)):
    """Verify whether the image matches the animal described in the text.

    Args:
        image: Image file (JPEG, PNG, …).
        text:  Natural-language sentence(s) describing the image content.

    Returns:
        JSON with predicted_class, confidence, extracted_animals, verdict (bool).
    """
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    # Write upload to a temp file so classify_image can read it by path
    with NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(await image.read())
        tmp_path = tmp.name

    try:
        predicted, confidence = classify_image(
            tmp_path,
            CLASSIFIER_MODEL,
            session=app.state.ort_session,
        )
        animals = extract_from_nlp(text, app.state.nlp)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    predicted_lower = predicted.lower()
    animals_lower = [a.lower() for a in animals]
    verdict = bool(animals_lower and predicted_lower in animals_lower)

    return {
        "predicted_class": predicted,
        "confidence": round(confidence, 4),
        "extracted_animals": animals,
        "verdict": verdict,
    }
