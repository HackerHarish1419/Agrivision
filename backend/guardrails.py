"""
AgriVisionAI — Input Validation & Guardrails
Blur detection, confidence gating, GPS validation, file validation.
All thresholds loaded from environment.
"""

import os
import io
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from typing import Optional, Tuple

load_dotenv()


class GuardrailEngine:
    """Configurable input validation engine."""

    def __init__(self):
        self.blur_threshold = float(os.getenv("BLUR_THRESHOLD", 100.0))
        self.confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", 0.60))
        self.max_file_size_mb = float(os.getenv("MAX_FILE_SIZE_MB", 10))
        self.min_image_dim = int(os.getenv("MIN_IMAGE_DIMENSION", 64))

        # GPS farm region — all from env
        self.farm_lat_min = float(os.getenv("FARM_LAT_MIN", -90))
        self.farm_lat_max = float(os.getenv("FARM_LAT_MAX", 90))
        self.farm_lng_min = float(os.getenv("FARM_LNG_MIN", -180))
        self.farm_lng_max = float(os.getenv("FARM_LNG_MAX", 180))
        self.farm_region_name = os.getenv("FARM_REGION_NAME", "Configured Farm Region")

        self.allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.allowed_mimes = {
            "image/jpeg", "image/png", "image/bmp", "image/webp"
        }

    def check_file_validity(self, filename: str, file_size: int, content_type: Optional[str] = None) -> dict:
        """Validate file format and size."""
        ext = os.path.splitext(filename)[1].lower() if filename else ""
        results = []

        # Extension check
        if ext and ext not in self.allowed_extensions:
            results.append({
                "check": "file_format",
                "passed": False,
                "score": None,
                "message": f"Invalid file format '{ext}'. Allowed: {', '.join(self.allowed_extensions)}",
                "severity": "error"
            })
        else:
            results.append({
                "check": "file_format",
                "passed": True,
                "score": None,
                "message": f"File format '{ext}' is valid",
                "severity": "info"
            })

        # Size check
        size_mb = file_size / (1024 * 1024)
        if size_mb > self.max_file_size_mb:
            results.append({
                "check": "file_size",
                "passed": False,
                "score": round(size_mb, 2),
                "message": f"File too large ({size_mb:.1f}MB). Maximum: {self.max_file_size_mb}MB",
                "severity": "error"
            })
        else:
            results.append({
                "check": "file_size",
                "passed": True,
                "score": round(size_mb, 2),
                "message": f"File size OK ({size_mb:.1f}MB)",
                "severity": "info"
            })

        return results

    def check_image_dimensions(self, image: Image.Image) -> dict:
        """Check minimum image dimensions."""
        w, h = image.size
        passed = w >= self.min_image_dim and h >= self.min_image_dim
        return {
            "check": "image_dimensions",
            "passed": passed,
            "score": min(w, h),
            "message": (
                f"Image dimensions OK ({w}×{h})" if passed
                else f"Image too small ({w}×{h}). Minimum: {self.min_image_dim}×{self.min_image_dim}"
            ),
            "severity": "info" if passed else "error"
        }

    def check_blur(self, image: Image.Image) -> dict:
        """Detect blurry images using Laplacian variance."""
        try:
            import cv2
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array

            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            is_blurry = laplacian_var < self.blur_threshold

            return {
                "check": "blur_detection",
                "passed": not is_blurry,
                "score": round(float(laplacian_var), 2),
                "message": (
                    f"Image is sharp (blur score: {laplacian_var:.1f}, threshold: {self.blur_threshold})"
                    if not is_blurry
                    else f"⚠️ Image appears blurry (blur score: {laplacian_var:.1f}, threshold: {self.blur_threshold}). Please recapture a clearer image."
                ),
                "severity": "info" if not is_blurry else "error"
            }
        except ImportError:
            return {
                "check": "blur_detection",
                "passed": True,
                "score": None,
                "message": "Blur detection unavailable (OpenCV not installed)",
                "severity": "info"
            }

    def check_confidence(self, confidence: float) -> dict:
        """Check if prediction confidence meets threshold."""
        passed = confidence >= self.confidence_threshold
        return {
            "check": "confidence_threshold",
            "passed": passed,
            "score": round(confidence, 4),
            "message": (
                f"Confidence OK ({confidence:.1%} ≥ {self.confidence_threshold:.0%})"
                if passed
                else f"⚠️ Low confidence ({confidence:.1%} < {self.confidence_threshold:.0%}). Consider recapturing with better lighting/angle."
            ),
            "severity": "info" if passed else "warning"
        }

    def check_gps(self, latitude: Optional[float], longitude: Optional[float]) -> dict:
        """Validate GPS coordinates against configured farm region."""
        if latitude is None or longitude is None:
            return {
                "check": "gps_validation",
                "passed": True,
                "score": None,
                "message": "GPS coordinates not provided (optional)",
                "severity": "info"
            }

        in_region = (
            self.farm_lat_min <= latitude <= self.farm_lat_max and
            self.farm_lng_min <= longitude <= self.farm_lng_max
        )

        return {
            "check": "gps_validation",
            "passed": in_region,
            "score": None,
            "message": (
                f"GPS location ({latitude:.4f}, {longitude:.4f}) is within {self.farm_region_name}"
                if in_region
                else f"⚠️ GPS location ({latitude:.4f}, {longitude:.4f}) is outside {self.farm_region_name}. "
                     f"Expected region: lat [{self.farm_lat_min}, {self.farm_lat_max}], "
                     f"lng [{self.farm_lng_min}, {self.farm_lng_max}]"
            ),
            "severity": "info" if in_region else "warning"
        }

    def check_leaf_likelihood(self, image: Image.Image) -> dict:
        """Heuristic check — does the image likely contain a leaf? (green channel dominance)."""
        img_array = np.array(image.resize((64, 64)))
        if len(img_array.shape) < 3:
            return {
                "check": "leaf_detection",
                "passed": True,
                "score": None,
                "message": "Grayscale image — leaf detection skipped",
                "severity": "info"
            }

        r_mean = img_array[:, :, 0].mean()
        g_mean = img_array[:, :, 1].mean()
        b_mean = img_array[:, :, 2].mean()

        # Leaves tend to have higher green channel, or be brownish (diseased)
        # Very permissive check — just flags clearly non-plant images
        green_ratio = g_mean / (r_mean + g_mean + b_mean + 1e-6)
        is_likely_leaf = green_ratio > 0.25  # Very permissive

        return {
            "check": "leaf_detection",
            "passed": is_likely_leaf,
            "score": round(float(green_ratio), 3),
            "message": (
                f"Image appears to contain plant material (green ratio: {green_ratio:.2f})"
                if is_likely_leaf
                else f"⚠️ Image may not contain a leaf (green ratio: {green_ratio:.2f}). Please upload a leaf image."
            ),
            "severity": "info" if is_likely_leaf else "warning"
        }

    def run_all_checks(
        self,
        image: Image.Image,
        filename: str,
        file_size: int,
        content_type: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
    ) -> list:
        """Run all guardrail checks and return results."""
        results = []

        # File checks
        results.extend(self.check_file_validity(filename, file_size, content_type))

        # Image checks
        results.append(self.check_image_dimensions(image))
        results.append(self.check_blur(image))
        results.append(self.check_leaf_likelihood(image))

        # GPS check
        results.append(self.check_gps(latitude, longitude))

        return results
