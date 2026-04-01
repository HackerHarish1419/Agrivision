import torch
import torch.nn as nn
from torchvision import models

pt_path = r'd:\agri_Harish_M\training\agrivision_efficientnet_b0.pt'
onnx_path = r'd:\agri_Harish_M\models\agrivision_efficientnet_b0.onnx'

try:
    print(f"Loading {pt_path}...")
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 18)

    state_dict = torch.load(pt_path, map_location='cpu')
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    elif isinstance(state_dict, nn.Module):
        state_dict = state_dict.state_dict()

    model.load_state_dict(state_dict)
    model.eval()

    print("Exporting ONNX...")
    dummy_input = torch.randn(1, 3, 224, 224, device='cpu')
    torch.onnx.export(model, dummy_input, onnx_path, export_params=True, opset_version=14, input_names=['input'], output_names=['output'])
    print('✅ ONNX file created successfully at:', onnx_path)
except Exception as e:
    print("❌ Failed to export:", e)
