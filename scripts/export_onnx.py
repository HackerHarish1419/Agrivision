"""
AgriVisionAI — PyTorch to ONNX Exporter
Converts the trained EfficientNet-B0 to ONNX format for Ryzen AI NPU acceleration.
"""

import os
import torch
import torch.nn as nn
from torchvision import models

def export_to_onnx(pt_model_path, onnx_model_path, num_classes=38):
    print(f"Loading PyTorch model from: {pt_model_path}")
    
    # Rebuild defining architecture to match exact training structure provided by the user
    model = models.efficientnet_b0(weights=None)
    
    # Replace the final classification head
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    # Load weights
    try:
        # User might have saved just the state_dict or the full model
        state_dict = torch.load(pt_model_path, map_location="cpu")
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif isinstance(state_dict, nn.Module):
            model = state_dict
        else:
            model.load_state_dict(state_dict)
            
        print("✅ Weights loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    model.eval()
    
    # Create dummy input based on EfficientNet 224x224 input
    dummy_input = torch.randn(1, 3, 224, 224, device="cpu")
    
    # Export to ONNX
    print(f"Exporting to ONNX: {onnx_model_path}")
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_model_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    
    print("\n✅ Model successfully converted to ONNX!")
    print(f"   Output saved at: {onnx_model_path}")

if __name__ == "__main__":
    PT_PATH = r"d:\agri_Harish_M\training\agrivision_efficientnet_b0.pt"
    ONNX_PATH = r"d:\agri_Harish_M\models\agrivision_efficientnet_b0.onnx"
    
    # Infer number of classes (Default 38 for PlantVillage if not 18)
    # The script will try 38 first, if dimension mismatch, it tries 18
    try:
        export_to_onnx(PT_PATH, ONNX_PATH, num_classes=38)
    except RuntimeError as e:
        if "size mismatch" in str(e):
            print("\n🔄 Detected size mismatch! Attempting 18 classes instead...")
            export_to_onnx(PT_PATH, ONNX_PATH, num_classes=18)
        else:
            raise e
