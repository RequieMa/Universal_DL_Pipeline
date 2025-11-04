import torch
import torch.nn as nn
from torchvision import models
from abc import ABC, abstractmethod
from monai.networks.nets import DenseNet121, EfficientNetBN

class Model(ABC):
    def __init__(self, model_name, num_classes):
        self.model_name = model_name
        self.model = None
        self.num_classes = num_classes
        self.head_module = None
        
    def set_model(self, model):
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False   
    
    @abstractmethod
    def replace_feature(self):
        pass
    
    @abstractmethod
    def replace_head(self):
        pass
        
class EfficientNet(Model):
    def __init__(self, model_name, num_classes):
        super().__init__(model_name, num_classes)
        self.set_model(models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT))
        
    def replace_feature(self):
        new_feature = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        nn.init.kaiming_normal_(new_feature.weight, mode='fan_out', nonlinearity='relu')
        self.model.features[0][0] = new_feature
        
    def replace_head(self):
        self.head_module = nn.Linear(1280, self.num_classes)
        nn.init.kaiming_normal_(self.head_module.weight, mode='fan_out', nonlinearity='relu')
        self.model.classifier[1] = self.head_module
      
class EfficientNet_MonAI(Model):
    def __init__(self, model_name, num_classes):
        super().__init__(model_name, num_classes)
        self.set_model(EfficientNetBN(
            "efficientnet-b4",
            spatial_dims=2,
            in_channels=1,
            num_classes=num_classes,
            pretrained=True
        ))
        
    def replace_feature(self):
        pass
        
    def replace_head(self):
        self.head_module = self.model._fc
        
class RegNet(Model):
    def __init__(self, model_name, num_classes):
        super().__init__(model_name, num_classes)
        self.set_model(models.regnet_y_32gf(weights=models.RegNet_Y_32GF_Weights.IMAGENET1K_SWAG_E2E_V1.DEFAULT))
        
    def replace_feature(self):
        new_feature = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        nn.init.kaiming_normal_(new_feature.weight, mode='fan_out', nonlinearity='relu')
        self.model.stem[0] = new_feature
        
    def replace_head(self):
        self.head_module = nn.Linear(3712, self.num_classes)
        nn.init.kaiming_normal_(self.head_module.weight, mode='fan_out', nonlinearity='relu')
        self.model.fc = self.head_module
        
class SqueezeNet(Model):
    def __init__(self, model_name, num_classes):
        super().__init__(model_name, num_classes)
    
    def replace_feature(self):
        pass
        
    def replace_head(self):
        pass
        
class ShuffleNet(Model):
    def __init__(self, model_name, num_classes):
        super().__init__(model_name, num_classes)
        
    def replace_feature(self):
        pass
        
    def replace_head(self):
        pass
        
class ResNet18(Model):
    def __init__(self, model_name, num_classes):
        super().__init__(model_name, num_classes)
        
    def replace_feature(self):
        pass
        
    def replace_head(self):
        pass
        
class ResNet101(Model):
    pass

class ResNet101_MonAI(Model):
    def __init__(self, model_name, num_classes):
        super().__init__(model_name, num_classes)
        self.set_model(ResNet101(
            block='basic',  # 'basic', 'bottleneck'
            layers=[3, 4, 6, 3],  # 各层块数
            block_inplanes=[64, 128, 256, 512],
            n_input_channels=3,
            num_classes=2,
            spatial_dims=2
        ))
        
    def replace_feature(self):
        pass
        
    def replace_head(self):
        self.head_module = self.model.class_layers.out
        
class ConvnextLarge(Model):
    def __init__(self, model_name, num_classes):
        super().__init__(model_name, num_classes)
        self.set_model(models.convnext_large(weights=models.ConvNeXt_Large_Weights.DEFAULT))
        
    def replace_feature(self):
        new_feature = nn.Conv2d(1, 192, kernel_size=(4, 4), stride=(4, 4))
        nn.init.kaiming_normal_(new_feature.weight, mode='fan_out', nonlinearity='relu')
        self.model.features[0][0] = new_feature
        
    def replace_head(self):
        self.head_module = nn.Linear(1536, self.num_classes)
        nn.init.kaiming_normal_(self.head_module.weight, mode='fan_out', nonlinearity='relu')
        self.model.classifier[2] = self.head_module
        
class Vgg19(Model):
    def __init__(self, model_name, num_classes):
        super().__init__(model_name, num_classes)
        self.set_model(models.vgg19_bn(weights=models.VGG19_BN_Weights.DEFAULT))
        
    def replace_feature(self):
        new_feature = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        nn.init.kaiming_normal_(new_feature.weight, mode='fan_out', nonlinearity='relu')
        self.model.features[0] = new_feature
        
    def replace_head(self):
        for param in self.model.classifier[0].parameters():
            param.requires_grad = True
        for param in self.model.classifier[3].parameters():
            param.requires_grad = True
        self.head_module = nn.Linear(4096, self.num_classes)
        nn.init.kaiming_normal_(self.head_module.weight, mode='fan_out', nonlinearity='relu')
        self.model.classifier[6] = self.head_module
        
class WideResNet101(Model):
    def __init__(self, model_name, num_classes):
        super().__init__(model_name, num_classes)
        
    def replace_feature(self):
        pass
        
    def replace_head(self):
        pass
    
class Vit32(Model):
    def __init__(self, model_name, num_classes):
        super().__init__(model_name, num_classes)
        self.set_model(models.vit_l_32(weights=models.ViT_L_32_Weights.DEFAULT))
        
    def replace_feature(self):
        new_feature = nn.Conv2d(1, 1024, kernel_size=(32, 32), stride=(32, 32))
        nn.init.kaiming_normal_(new_feature.weight, mode='fan_out', nonlinearity='relu')
        self.model.conv_proj = new_feature
        
    def replace_head(self):
        self.head_module = nn.Linear(1024, self.num_classes)
        nn.init.kaiming_normal_(self.head_module.weight, mode='fan_out', nonlinearity='relu')
        self.model.heads.head = self.head_module
        
class DenseNet161(Model):
    def __init__(self, model_name, num_classes):
        super().__init__(model_name, num_classes)
        self.set_model(models.densenet161(weights=models.DenseNet161_Weights.DEFAULT))
        
    def replace_feature(self):
        new_feature = nn.Conv2d(1, 96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        nn.init.kaiming_normal_(new_feature.weight, mode='fan_out', nonlinearity='relu')
        self.model.features.conv0 = new_feature
        
    def replace_head(self):
        self.head_module = nn.Linear(2208, self.num_classes)
        nn.init.kaiming_normal_(self.head_module.weight, mode='fan_out', nonlinearity='relu')
        self.model.classifier = self.head_module
        
class DenseNet121_MonAI(Model):
    def __init__(self, model_name, num_classes):
        super().__init__(model_name, num_classes)
        self.set_model(DenseNet121(
            spatial_dims=2,
            in_channels=1,
            out_channels=num_classes,
            pretrained=True
        ))
        
    def replace_feature(self):
        pass
        
    def replace_head(self):
        self.head_module = self.model.class_layers.out
        
class DenseNet201(Model):
    def __init__(self, model_name, num_classes):
        super().__init__(model_name, num_classes)
        self.set_model(models.densenet201(weights=models.DenseNet201_Weights.DEFAULT))
        
    def replace_feature(self):
        new_feature = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        nn.init.kaiming_normal_(new_feature.weight, mode='fan_out', nonlinearity='relu')
        self.model.features.conv0 = new_feature
        
    def replace_head(self):
        self.head_module = nn.Linear(1920, self.num_classes)
        nn.init.kaiming_normal_(self.head_module.weight, mode='fan_out', nonlinearity='relu')
        self.model.classifier = self.head_module
        
class SwinV2(Model):
    def __init__(self, model_name, num_classes):
        super().__init__(model_name, num_classes)
        self.set_model(models.swin_v2_b(weights=models.Swin_V2_B_Weights.DEFAULT))
        
    def replace_feature(self):
        new_feature = nn.Conv2d(1, 128, kernel_size=(4, 4), stride=(4, 4))
        nn.init.kaiming_normal_(new_feature.weight, mode='fan_out', nonlinearity='relu')
        self.model.features[0][0] = new_feature
        
    def replace_head(self):
        self.head_module = nn.Linear(1024, self.num_classes)
        nn.init.kaiming_normal_(self.head_module.weight, mode='fan_out', nonlinearity='relu')
        self.model.head = self.head_module