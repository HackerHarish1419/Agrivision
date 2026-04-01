"""
AgriVisionAI — Pydantic Schemas
Request/Response models for the API.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


# === Guardrail Results ===
class GuardrailResult(BaseModel):
    check: str = Field(..., description="Name of the guardrail check")
    passed: bool = Field(..., description="Whether the check passed")
    score: Optional[float] = Field(None, description="Numeric score if applicable")
    message: str = Field(..., description="Human-readable message")
    severity: str = Field("info", description="info | warning | error")


# === Prediction ===
class PredictionItem(BaseModel):
    class_name: str
    crop: str
    disease: str
    confidence: float


class PredictionResponse(BaseModel):
    success: bool
    prediction: Optional[PredictionItem] = None
    top_predictions: List[PredictionItem] = []
    guardrails: List[GuardrailResult] = []
    warnings: List[str] = []
    image_info: Dict[str, Any] = {}
    needs_recapture: bool = False


# === LLM Recommendation ===
class RecommendRequest(BaseModel):
    crop: str = Field(..., description="Crop type: Apple, Grape, or Tomato")
    disease: str = Field(..., description="Detected disease name")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence 0-1")
    latitude: Optional[float] = Field(None, description="GPS latitude for time context")
    longitude: Optional[float] = Field(None, description="GPS longitude for time context")


class LLMRecommendation(BaseModel):
    severity: str
    location_context: str
    time_context: str
    recommendations: List[str]
    recovery_time: str
    preventive_note: str = "unknown"


class RecommendResponse(BaseModel):
    success: bool
    recommendation: Optional[LLMRecommendation] = None
    error: Optional[str] = None


# === Health & Info ===
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_arch: str
    num_classes: int
    groq_configured: bool


class ClassInfo(BaseModel):
    index: int
    class_name: str
    crop: str
    disease: str


class ClassesResponse(BaseModel):
    crops: List[str]
    classes: List[ClassInfo]
    total_classes: int
