import os
import torch
import torch.nn as nn
import torch.cuda.amp as amp
from torch.utils.data import DataLoader, Subset
from torchvision import models
from dataset_generation import load_datasets, categories
from collections import Counter
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
import time
from const import SEED
from msgspec import Struct, Meta, yaml

torch.manual_seed(SEED)
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = False

class DLConfig(Struct):
    data_root : str
    model_name : str
    image_size : int = 224
    merge_train_val : bool = False
    k_fold : int | None = None
    unfreeze_layers : int = 0
    lr: float = 0.1
    weight_decay: float = 0.01

class Classifier:
    CHECKPOINT_PATH = "training_checkpoint.pth"
    KFOLD_CHECKPOINT_PATH = "kfold_checkpoint_{fold}.pth"
    
    def __init__(self, config, categories):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        self.config = config
        
        if self.config.k_fold:
            self.full_dataset, _, self.test_dataset = load_datasets(
                self.config.data_root, 
                image_size=self.config.image_size, 
                merge_train_val=self.config.merge_train_val, 
                k_fold=self.config.k_fold
            )
            self.kf = KFold(
                n_splits=self.config.k_fold, 
                shuffle=True, 
                random_state=SEED
            )
        else:
            self.train_dataset, self.val_dataset, self.test_dataset = load_datasets(
                self.config.data_root, 
                image_size=self.config.image_size, 
                merge_train_val=self.config.merge_train_val,
            )
        
        self.num_classes = len(categories)
        self.model = None
        self.head_module = None
        self.criterion = None
        self.optimizer = None

    def build_model(self):  # 默认改为单通道
        net = None
        model_name = self.config.model_name
        unfreeze_layers = self.config.unfreeze_layers
        match model_name:
            case 'efficientnet_v2_s': 
                from model_generation import EfficientNet
                net = EfficientNet(model_name, self.num_classes)
            case 'efficientnet_med': 
                from model_generation import EfficientNet_MonAI
                net = EfficientNet_MonAI(model_name, self.num_classes)
            case 'regnet_y_32gf':
                from model_generation import RegNet
                net = RegNet(model_name, self.num_classes)
            case 'squeezenet1_1':
                self.model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
            case 'shufflenet_v2_x0_5':
                self.model = models.shufflenet_v2_x0_5(weights=models.ShuffleNet_V2_X0_5_Weights.DEFAULT)
            case 'resnet18':
                self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            case 'resnet101':
                self.model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
            case 'convnext_large':
                from model_generation import ConvnextLarge
                net = ConvnextLarge(model_name, self.num_classes)
            case 'vgg19_bn':
                from model_generation import Vgg19
                net = Vgg19(model_name, self.num_classes)
            case 'wide_resnet101_2':
                self.model = models.wide_resnet101_2(weights=models.Wide_ResNet101_2_Weights.DEFAULT)
            case 'vit_l_32':
                from model_generation import Vit32
                net = Vit32(model_name, self.num_classes)
            case 'densenet161':
                from model_generation import DenseNet161
                net = DenseNet161(model_name, self.num_classes)
            case 'densenet121':
                from model_generation import DenseNet121_MonAI
                net = DenseNet121_MonAI(model_name, self.num_classes)
            case 'densenet201':
                from model_generation import DenseNet201
                net = DenseNet201(model_name, self.num_classes)
            case 'swin':
                from model_generation import SwinV2
                net = SwinV2(model_name, self.num_classes)
            case _:
                raise ValueError(f"不支持的模型: {model_name}")
        
        assert net is not None, "No Net..."
        net.replace_feature()
        net.replace_head()

        self.model = net.model
        self.head_module = net.head_module
        if unfreeze_layers > 0:
            children = list(self.model.children())
            total_layers = len(children)
            layers_to_unfreeze = min(unfreeze_layers, total_layers)
            for child in children[-layers_to_unfreeze:]:
                for param in child.parameters():
                    param.requires_grad = True
            print(f"解冻最后 {layers_to_unfreeze} 层参数")
        self.model.to(self.device)

    def prepare_train(self, beta1=0.9, beta2=0.999):
        # class_weights = self.get_class_weights()
        # self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(self.device), label_smoothing=0.1)
        self.criterion = torch.nn.CrossEntropyLoss()
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.AdamW(
            trainable_params, 
            lr=self.config.lr, 
            weight_decay=self.config.weight_decay,
            betas=(beta1, beta2)  # 调整beta参数
        )
        # self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     self.optimizer, max_lr=0.005, epochs=20, 
        #     pct_start=0.2, steps_per_epoch=len(train_loader)
        # )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer, 
        #     mode='max', 
        #     factor=0.5, 
        #     patience=3, 
        #     verbose=True
        # )
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     self.optimizer,
        #     T_0=10,  # 初始周期
        #     T_mult=1,  # 周期倍增因子
        #     eta_min=1e-6  # 最小学习率
        # )
        # self.warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        #     self.optimizer,
        #     lr_lambda=lambda epoch: min(1.0, (epoch + 1) / self.warmup_epochs)
        # )

    def create_data_loaders(self, train_idx=None, val_idx=None):
        if self.k_fold and train_idx is not None and val_idx is not None:
            train_subset = Subset(self.full_dataset, train_idx)
            val_subset = Subset(self.full_dataset, val_idx)
        else:
            train_subset = self.train_dataset
            val_subset = self.val_dataset

        batch_size = 32
        train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True, persistent_workers=True,
        )
        val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True, persistent_workers=True,
        )
        return train_loader, val_loader

    def get_class_weights(self):
        labels = self.train_dataset.labels
        class_counts = Counter(labels)
        total_samples = len(labels)
        class_weights = {}
        
        for class_idx, count in class_counts.items():
            # 使用逆频率加权，给少数类更高权重
            class_weights[class_idx] = total_samples / (len(class_counts) * count)
        return torch.tensor([class_weights[i] for i in sorted(class_weights.keys())])

    
       
    def load_kfold_checkpoint(self, fold):
        checkpoint_path = self.KFOLD_CHECKPOINT_PATH.format(fold=fold)
        if os.path.exists(checkpoint_path):
            print(f"发现第 {fold} 折检查点，恢复训练...")
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            return (
                checkpoint['epoch'] + 1,
                checkpoint['train_scores'],
                checkpoint['val_scores']
            )
        return 0, [], []  # 没有检查点时从头开始
        
    def save_kfold_checkpoint(self, fold, epoch, train_scores, val_scores):
        checkpoint_path = self.KFOLD_CHECKPOINT_PATH.format(fold=fold)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_scores': train_scores,
            'val_scores': val_scores,
        }, checkpoint_path)
        print(f"第 {fold} 折检查点已保存: epoch {epoch + 1}")
        
    def load_checkpoint(self):
        if os.path.exists(self.CHECKPOINT_PATH):
            print("发现检查点，恢复训练...")
            checkpoint = torch.load(self.CHECKPOINT_PATH, weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            return (
                checkpoint['epoch'] + 1,  # 从下一轮开始
                checkpoint['train_scores'],
                checkpoint['validation_scores']
            )
        return 0, [], []  # 没有检查点时从头开始  
        
    def save_checkpoint(self, epoch, train_scores, validation_scores):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_scores': train_scores,
            'validation_scores': validation_scores,
        }, self.CHECKPOINT_PATH)
        print(f"检查点已保存: epoch {epoch + 1}")
           
    def train_epoch(self, train_loader, epoch, num_epochs):
        if torch.cuda.memory_allocated() > 0.8 * torch.cuda.max_memory_allocated():
            torch.cuda.empty_cache()
            
        total_samples = len(train_loader.dataset)
        self.model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            self.optimizer.zero_grad()
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
        train_loss = running_loss / total_samples
        train_acc = 100 * train_correct / total_samples if total_samples > 0 else 0
        return train_loss, train_acc
            
    def validate_epoch(self, val_loader):
        self.model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Verification..."):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        return val_loss, val_acc
            
    def train_process(self, num_epochs=10, start_epoch=0, train_scores=[], validation_scores=[]):
        if self.k_fold:
            fold_results = []
            for fold, (train_idx, val_idx) in enumerate(self.kf.split(range(len(self.full_dataset)))):
                print(f"\n--- 开始第 {fold + 1}/{self.k_fold} 折交叉验证 ---")
                train_loader, val_loader = self.create_data_loaders(train_idx, val_idx)
                
                # 重置模型和优化器
                self.build_model(model_name, unfreeze_layers=3)
                # self.warmup_epochs = min(5, num_epochs//4)  # 预热epoch数
                self.prepare_train(
                    lr=0.005,
                    weight_decay=0.001,
                    beta1=0.9 - fold*0.01,  # 每折微调beta1
                    beta2=0.999 - fold*0.0001  # 每折微调beta2
                )
                start_epoch, train_scores, val_scores = self.load_kfold_checkpoint(fold)
                
                for epoch in range(start_epoch, num_epochs):
                    # if epoch < self.warmup_epochs:
                    #     self.warmup_scheduler.step()
                    train_loss, train_acc = self.train_epoch(train_loader, epoch, num_epochs)
                    val_loss, val_acc = self.validate_epoch(val_loader)
                    
                    train_scores.append(train_acc)
                    val_scores.append(val_acc)
                    
                    print(  f"Epoch [{epoch+1}/{num_epochs}]: "
                            f"Train Loss: {train_loss:.4f}; Train Acc: {train_acc:.2f}% - "
                            f"Val Loss: {val_loss:.4f}; Val Acc: {val_acc:.2f}%")
                    # if epoch >= self.warmup_epochs:
                    self.scheduler.step()
                    self.save_kfold_checkpoint(fold, epoch, train_scores, val_scores)
                fold_results.append((train_scores, val_scores))
                print(f"--- 完成第 {fold + 1}/{self.k_fold} 折 ---")
            
            # 计算平均性能
            avg_train_scores = np.mean([res[0] for res in fold_results], axis=0)
            avg_val_scores = np.mean([res[1] for res in fold_results], axis=0)
            self.plot_train_valid(avg_train_scores, avg_val_scores, num_epochs)
        else:
            train_loader, val_loader = self.create_data_loaders()
            print(f"从 epoch {start_epoch + 1} 开始，共 {num_epochs} 轮")
            # warmup_epochs = min(5, num_epochs//4)
            for epoch in range(start_epoch, num_epochs):
                # if epoch < warmup_epochs:
                #     for param_group in self.optimizer.param_groups:
                #         param_group['lr'] = 0.001 * (epoch + 1) / warmup_epochs
                train_loss, train_acc = self.train_epoch(train_loader, epoch, num_epochs)
                val_loss, val_acc = self.validate_epoch(val_loader)
                train_scores.append(train_acc)
                validation_scores.append(val_acc)
                
                print(  f"Epoch [{epoch+1}/{num_epochs}]: "
                        f"Train Loss: {train_loss:.4f}; Train Acc: {train_acc:.2f}% - "
                        f"Val Loss: {val_loss:.4f}; Val Acc: {val_acc:.2f}%")
                self.scheduler.step()
                # if epoch > num_epochs // 2:
                #     self.warmup_scheduler.step()
                self.save_checkpoint(epoch, train_scores, validation_scores)
                
            self.plot_train_valid(train_scores, validation_scores, num_epochs)
        
        if os.path.exists(self.CHECKPOINT_PATH):
            os.remove(self.CHECKPOINT_PATH)
            print("训练完成，检查点已删除")
    
    def create_submission(self, output_file="submission.csv"):
        self.model.eval()
        predictions = []
        test_loader = DataLoader(self.test_dataset, batch_size=32, num_workers=4, shuffle=False)
        
        with torch.no_grad():
            for images, _ in tqdm(test_loader, desc="生成预测结果"):
                images = images.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.cpu().numpy())
        
        # 创建提交文件
        submission_df = pd.DataFrame({
            'index': range(len(predictions)),
            'pred': predictions
        })
        submission_df.to_csv(output_file, index=False)
        print(f"提交文件已保存: {output_file}")
                
    def plot_train_valid(self, train_scores, validation_scores, num_epochs):
        plt.plot(range(1, num_epochs + 1), train_scores, 'o-', color='r', label='Training score')
        plt.plot(range(1, num_epochs + 1), validation_scores, 'o-', color='g', label='Validation score')
        
        plt.title('Learning Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend(loc='best')
        plt.grid()   
        plt.savefig('Score_plot.png', bbox_inches='tight')
        plt.close()
                        
if __name__ == "__main__":
    with open("config.yaml", "rb") as f:                      # read as bytes
        config = yaml.decode(f.read(), type=DLConfig)
    # data_root = "./IS_2025_OrganAMNIST"
    # model_name = 'efficientnet_med'
    classifer = Classifier(config, categories)
    classifer.build_model(model_name, unfreeze_layers=3)
    classifer.prepare_train()
    start_epoch, train_scores, validation_scores = classifer.load_checkpoint()
    classifer.train_process(
        num_epochs=20, start_epoch=start_epoch, 
        train_scores=train_scores, 
        validation_scores=validation_scores
    )
    print("生成提交文件...")
    classifer.create_submission(f"{model_name.split('_')[0]}_submission.csv")
    
    
    
    