"""
AgriVisionAI — FastAPI Backend
Main application entry point. Serves API + static frontend.
All configuration from .env — nothing hardcoded.
"""

import os
import io
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

load_dotenv()

from backend.inference import ModelInference
from backend.guardrails import GuardrailEngine
from backend.llm_service import LLMService
from backend.schemas import (
    PredictionResponse, PredictionItem, GuardrailResult,
    RecommendRequest, RecommendResponse, LLMRecommendation,
    HealthResponse, ClassesResponse, ClassInfo
)

# === Globals (initialized on startup) ===
model_engine: Optional[ModelInference] = None
guardrail_engine: Optional[GuardrailEngine] = None
llm_service: Optional[LLMService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup."""
    global model_engine, guardrail_engine, llm_service

    print("\n🌱 AgriVisionAI — Starting up...")
    model_engine = ModelInference()
    guardrail_engine = GuardrailEngine()
    llm_service = LLMService()
    print("✅ All services initialized\n")

    yield

    print("🛑 AgriVisionAI — Shutting down")


app = FastAPI(
    title="AgriVisionAI",
    description="Crop Disease Intelligence System — Matrix Fusion 4.0",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === API Endpoints ===

@app.post("/api/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    crop: Optional[str] = Form(None),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
):
    """
    Upload a leaf image for disease prediction.
    Returns prediction + guardrail validation results.
    """
    # Read image bytes
    contents = await file.read()
    file_size = len(contents)

    # Run file-level guardrails
    guardrail_results = guardrail_engine.check_file_validity(
        file.filename, file_size, file.content_type
    )

    # Check for hard errors in file validation
    hard_errors = [r for r in guardrail_results if not r["passed"] and r["severity"] == "error"]
    if hard_errors:
        return PredictionResponse(
            success=False,
            guardrails=[GuardrailResult(**r) for r in guardrail_results],
            warnings=[r["message"] for r in hard_errors],
            needs_recapture=True,
        )

    # Open image
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        return PredictionResponse(
            success=False,
            guardrails=[GuardrailResult(**r) for r in guardrail_results],
            warnings=[f"Could not open image: {str(e)}"],
            needs_recapture=True,
        )

    # Run image-level guardrails
    image_checks = [
        guardrail_engine.check_image_dimensions(image),
        guardrail_engine.check_blur(image),
        guardrail_engine.check_leaf_likelihood(image),
        guardrail_engine.check_gps(latitude, longitude),
    ]
    guardrail_results.extend(image_checks)

    # Collect warnings
    warnings = [r["message"] for r in guardrail_results if not r["passed"]]
    has_hard_error = any(
        not r["passed"] and r["severity"] == "error"
        for r in guardrail_results
    )

    if has_hard_error:
        return PredictionResponse(
            success=False,
            guardrails=[GuardrailResult(**r) for r in guardrail_results],
            warnings=warnings,
            needs_recapture=True,
            image_info={"width": image.size[0], "height": image.size[1]},
        )

    # Run model inference
    result = model_engine.predict(image, top_k=5)

    if not result["success"]:
        return PredictionResponse(
            success=False,
            warnings=[result.get("error", "Prediction failed")],
            guardrails=[GuardrailResult(**r) for r in guardrail_results],
        )

    # Confidence check
    top_prediction = result["prediction"]
    confidence_check = guardrail_engine.check_confidence(top_prediction["confidence"])
    guardrail_results.append(confidence_check)

    if not confidence_check["passed"]:
        warnings.append(confidence_check["message"])

    # If crop was specified, validate it matches prediction
    if crop and top_prediction["crop"].lower() != crop.lower():
        warnings.append(
            f"Selected crop '{crop}' doesn't match predicted crop '{top_prediction['crop']}'. "
            f"The model predicts this is a {top_prediction['crop']} leaf."
        )

    return PredictionResponse(
        success=True,
        prediction=PredictionItem(**top_prediction),
        top_predictions=[PredictionItem(**p) for p in result["top_predictions"]],
        guardrails=[GuardrailResult(**r) for r in guardrail_results],
        warnings=warnings,
        needs_recapture=not confidence_check["passed"],
        image_info={"width": image.size[0], "height": image.size[1]},
    )


@app.post("/api/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    """Get LLM-powered recovery recommendations for a detected disease."""
    result = await llm_service.get_recommendation(
        crop=request.crop,
        disease=request.disease,
        confidence=request.confidence,
        latitude=request.latitude,
        longitude=request.longitude,
    )

    if result["success"]:
        return RecommendResponse(
            success=True,
            recommendation=LLMRecommendation(**result["recommendation"]),
        )
    else:
        return RecommendResponse(
            success=False,
            error=result.get("error", "Unknown error"),
        )


@app.get("/api/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model_engine.model_loaded if model_engine else False,
        model_arch=model_engine.execution_mode if model_engine else "not loaded",
        num_classes=model_engine.num_classes if model_engine else 0,
        groq_configured=llm_service.configured if llm_service else False,
    )


@app.get("/api/classes", response_model=ClassesResponse)
async def get_classes():
    """Return supported crops and disease classes."""
    if not model_engine or not model_engine.class_names:
        return ClassesResponse(crops=[], classes=[], total_classes=0)

    classes = []
    crops_set = set()
    for idx, class_name in model_engine.class_names.items():
        crop, disease = ModelInference.parse_class_name(class_name)
        crops_set.add(crop)
        classes.append(ClassInfo(
            index=idx,
            class_name=class_name,
            crop=crop,
            disease=disease,
        ))

    return ClassesResponse(
        crops=sorted(list(crops_set)),
        classes=classes,
        total_classes=len(classes),
    )


# === Serve Frontend ===
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR / "assets")), name="assets") if (FRONTEND_DIR / "assets").exists() else None
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    async def serve_frontend():
        """Serve the frontend SPA."""
        return FileResponse(str(FRONTEND_DIR / "index.html"))
else:
    @app.get("/")
    async def root():
        return {"message": "AgriVisionAI API is running. Frontend not found at ./frontend/"}


# === Run ===
if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("backend.main:app", host=host, port=port, reload=True)
