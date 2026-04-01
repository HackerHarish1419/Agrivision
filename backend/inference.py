"""
AgriVisionAI — Model Inference Engine
Loads trained model and runs prediction on images.
Supports PyTorch (.pt) and Ryzen AI NPU ONNX Runtime execution (.onnx).
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Optional

load_dotenv()

class ModelInference:
    """Handles model loading and prediction — fully configurable."""

    def __init__(self):
        self.class_names: Dict[int, str] = {}
        self.num_classes = 38
        self.image_size = int(os.getenv("IMAGE_SIZE", 224))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_loaded = False
        
        # Core model attributes
        self.pytorch_model = None
        self.onnx_session = None
        self.execution_mode = "unknown"

        # ImageNet normalization (matches user's training code)
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        self._load_class_names()
        self._load_model()

    def _load_class_names(self):
        class_names_path = Path(os.getenv("CLASS_NAMES_PATH", "models/class_names.json"))
        if class_names_path.exists():
            with open(class_names_path, "r") as f:
                raw = json.load(f)
                self.class_names = {int(k): v for k, v in raw.items()}
                self.num_classes = len(self.class_names)
            print(f"✅ Loaded {self.num_classes} class names")
        else:
            print("⚠️ class_names.json not found, proceeding with raw class indexing.")

    def _build_pytorch_model(self):
        """Build exact architecture from user's training script."""
        model = models.efficientnet_b0(weights=None)
        # The user's script specifically replaces classifier[1]
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, self.num_classes)
        return model

    def _load_model(self):
        """Load either ONNX (preferred for Ryzen AI) or PyTorch (.pt) model."""
        pt_path = Path("d:/agri_Harish_M/training/agrivision_efficientnet_b0.pt")
        onnx_path = Path("d:/agri_Harish_M/models/agrivision_efficientnet_b0.onnx")
        
        print("\n--- Initializing Vision Core ---")
        
        # 1. Check for ONNX Model (Ryzen AI NPU Pipeline)
        if onnx_path.exists():
            try:
                import onnxruntime as ort
                # Attempt to use Ryzen AI NPU
                providers = []
                if 'VitisAIExecutionProvider' in ort.get_available_providers():
                    providers.append('VitisAIExecutionProvider')
                elif 'DmlExecutionProvider' in ort.get_available_providers():
                    providers.append('DmlExecutionProvider')
                    
                providers.extend(['CPUExecutionProvider'])
                
                self.onnx_session = ort.InferenceSession(str(onnx_path), providers=providers)
                self.execution_mode = f"ONNX Runtime ({self.onnx_session.get_providers()[0]})"
                self.model_loaded = True
                print(f"✅ Loaded ONNX model on {self.execution_mode}")
                if 'VitisAI' in self.execution_mode:
                    print("🚀 Ryzen AI NPU Acceleration Active!")
                return
            except ImportError:
                print("⚠️ onnxruntime not installed, falling back to PyTorch .pt")
        
        # 2. Check for PyTorch Model
        if pt_path.exists():
            try:
                self.pytorch_model = self._build_pytorch_model()
                # Handle both raw full-model saves and state_dict saves
                state_dict = torch.load(str(pt_path), map_location=self.device)
                
                if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                    state_dict = state_dict["model_state_dict"]
                elif isinstance(state_dict, nn.Module):
                    state_dict = state_dict.state_dict()
                
                # Catch mismatched classes (18 vs 38)
                try:
                    self.pytorch_model.load_state_dict(state_dict)
                except RuntimeError as e:
                    if "size mismatch" in str(e):
                        print("🔄 Size mismatch detected (expected 38, got 18). Adjusting layer...")
                        self.num_classes = 18
                        self.pytorch_model = self._build_pytorch_model()
                        self.pytorch_model.load_state_dict(state_dict)
                    else:
                        raise e
                    
                self.pytorch_model.to(self.device)
                self.pytorch_model.eval()
                self.model_loaded = True
                self.execution_mode = f"PyTorch Native ({self.device})"
                print(f"✅ Loaded raw PyTorch model on {self.execution_mode}")
            except Exception as e:
                print(f"❌ Failed to load PyTorch model: {e}")
        else:
            print(f"⚠️ Neither ONNX nor PyTorch model found. Looking for: {pt_path}")

    @staticmethod
    def parse_class_name(class_name: str) -> Tuple[str, str]:
        parts = class_name.split("___")
        crop = parts[0] if parts else "Unknown"
        disease = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
        return crop, disease

    def _predict_onnx(self, tensor: torch.Tensor) -> torch.Tensor:
        import numpy as np
        np_input = tensor.cpu().numpy()
        input_name = self.onnx_session.get_inputs()[0].name
        outputs = self.onnx_session.run(None, {input_name: np_input})
        return torch.tensor(outputs[0])

    def _predict_pytorch(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.to(self.device)
        with torch.no_grad():
            outputs = self.pytorch_model(tensor)
        return outputs

    def predict(self, image: Image.Image, top_k: int = 3) -> Dict:
        if not self.model_loaded:
            return {"success": False, "error": "Model not loaded. Ensure your .pt file is in the right place."}

        if image.mode != "RGB":
            image = image.convert("RGB")

        # Prep tensor
        tensor = self.transform(image).unsqueeze(0)

        # Execute on dynamically determined backend
        if self.onnx_session is not None:
            outputs = self._predict_onnx(tensor)
        else:
            outputs = self._predict_pytorch(tensor)

        probabilities = F.softmax(outputs, dim=1)[0]
        top_probs, top_indices = torch.topk(probabilities, min(top_k, self.num_classes))

        predictions = []
        for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
            class_name = self.class_names.get(int(idx), f"Class {idx}")
            crop, disease = self.parse_class_name(class_name)
            predictions.append({
                "class_name": class_name,
                "crop": crop,
                "disease": disease,
                "confidence": float(prob)
            })

        return {
            "success": True,
            "prediction": predictions[0] if predictions else None,
            "top_predictions": predictions,
            "execution_mode": self.execution_mode
        }
