import torch.nn as nn
from torchvision import models

def get_model(model_type, num_classes):
    num_ftrs = model_type.fc.in_features
    model_type.fc = nn.Linear(num_ftrs, num_classes)  # Replace num_classes with the actual number of classes
    model_type.fc.weight = nn.init.kaiming_normal_(model_type.fc.weight, mode='fan_out', nonlinearity='relu')
    return model_type

def get_model_idx(model_type, num_classes, idx):
    num_ftrs = model_type.classifier[idx].in_features
    model_type.classifier[idx] = nn.Linear(num_ftrs, num_classes)  # Replace num_classes with the actual number of classes
    model_type.classifier[idx].weight = nn.init.kaiming_normal_(model_type.classifier[idx].weight, mode='fan_out', nonlinearity='relu')
    return model_type

def get_model_dict(num_classes):
    resnet18 = get_model(models.resnet18(weights=models.ResNet18_Weights.DEFAULT), num_classes)
    resnet34 = get_model(models.resnet34(weights=models.ResNet34_Weights.DEFAULT), num_classes)
    resnet50 = get_model(models.resnet50(weights=models.ResNet50_Weights.DEFAULT), num_classes)
    resnet101 = get_model(models.resnet101(weights=models.ResNet101_Weights.DEFAULT), num_classes)
    resnet152 = get_model(models.resnet152(weights=models.ResNet152_Weights.DEFAULT), num_classes)
    vgg16 = get_model_idx(models.vgg16(weights=models.VGG16_Weights.DEFAULT), num_classes, 6)
    vgg19 = get_model_idx(models.vgg19(weights=models.VGG19_Weights.DEFAULT), num_classes, 6)
    efficientnet_b0 = get_model_idx(models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT), num_classes, 1)
    model_dict = {
        "resnet18": resnet18,
        "resnet34": resnet34,
        "resnet50": resnet50,
        "resnet101": resnet101,
        "resnet152": resnet152,
        "vgg16": vgg16,
        "vgg19": vgg19,
        "efficientnet_b0": efficientnet_b0
    }
    return model_dict