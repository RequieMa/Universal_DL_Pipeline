import torch.onnx
from torchvision import models

categories = [
    "Anime", "People", "Scenes", 
    "Foods", "Animals", "Vehicles", 
    # "Documents"
]
num_classes = len(categories)

model_loaded = models.efficientnet_v2_s()
model_loaded.classifier[1] = torch.nn.Linear(model_loaded.classifier[1].in_features, num_classes)
# model_loaded = models.efficientnet_v2_s()
# model_loaded.fc = torch.nn.Linear(model_loaded.fc.in_features, num_classes)
model_loaded.load_state_dict(torch.load('fine_tune_model_15.pth', weights_only=True))
model_loaded.eval()

dummy_input = torch.randn(1, 3, 224, 224)  # Example input
torch.onnx.export(model_loaded, dummy_input, "tuned_regnet.onnx")