import os
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
import numpy as np
from collections import Counter
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from dataset_generation import load_datasets, categories

torch.manual_seed(2019)
np.random.seed(2019)

CHECKPOINT_PATH = "training_checkpoint.pth"

def plot_train_valid(train_scores, validation_scores, num_epochs):
    plt.plot(range(1, num_epochs + 1), train_scores, 'o-', color='r', label='Training score')
    plt.plot(range(1, num_epochs + 1), validation_scores, 'o-', color='g', label='Validation score')
    
    plt.title('Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend(loc='best')
    plt.grid()   
    plt.savefig('Score_plot.png', bbox_inches='tight')
    plt.close()
   
def save_checkpoint(epoch, model, optimizer, scheduler, train_scores, validation_scores):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_scores': train_scores,
        'validation_scores': validation_scores,
    }, CHECKPOINT_PATH)
    print(f"检查点已保存: epoch {epoch+1}")
        
def load_checkpoint(model, optimizer, scheduler):
    if os.path.exists(CHECKPOINT_PATH):
        print("发现检查点，恢复训练...")
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return (
            checkpoint['epoch'] + 1,  # 从下一轮开始
            checkpoint['train_scores'],
            checkpoint['validation_scores']
        )
    return 0, [], []  # 没有检查点时从头开始    

def build_model(model_name, num_classes, freeze_backbone=True, in_channels=1):  # 默认改为单通道
    """
    构建并配置预训练模型
    
    参数:
        model_name (str): 模型名称 ('efficientnet_v2_s', 'regnet_y_32gf', 'vit_l_16')
        num_classes (int): 输出类别数量
        freeze_backbone (bool): 是否冻结主干网络参数
        in_channels (int): 输入通道数 (默认为1，单通道灰度图)
    
    返回:
        torch.nn.Module: 配置好的模型
    """
    # 尝试替换分类头
    head_module = None
    head_replaced = False
    
    # 加载预训练模型
    if model_name == 'efficientnet_v2_s':
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
    elif model_name == 'regnet_y_32gf':
        model = models.regnet_y_32gf(weights=models.RegNet_Y_32GF_Weights.IMAGENET1K_SWAG_E2E_V1.DEFAULT)
    elif model_name == 'vit_l_16':
        # model = models.vit_l_16(weights=models.ViT_L_16_Weights.DEFAULT)
        model = models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1)
        # 专门处理 Vision Transformer 模型
        if hasattr(model, 'heads'):
            head_module = model.heads
            if isinstance(head_module, nn.Sequential) and len(head_module) > 0:
                # 获取当前分类头的输入特征数
                if hasattr(head_module[0], 'in_features'):
                    in_features = head_module[0].in_features
                elif hasattr(head_module[0], 'in_channels'):
                    in_features = head_module[0].in_channels
                else:
                    raise AttributeError("无法确定ViT输入特征数")
                
                # 创建新的分类头
                new_head = nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(in_features, num_classes)
                )
                # 替换分类头
                model.heads = new_head
                head_replaced = True
                print(f"成功替换 {model_name} 的ViT分类头")
            else:
                raise RuntimeError(f"ViT模型 heads 属性格式异常")
        else:
            raise RuntimeError(f"ViT模型缺少 heads 属性")
    elif model_name == 'squeezenet1_1':
        model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
    elif model_name == 'shufflenet_v2_x0_5':
        model = models.shufflenet_v2_x0_5(weights=models.ShuffleNet_V2_X0_5_Weights.DEFAULT)
    elif model_name == 'mnasnet0_5':
        model = models.mnasnet0_5(weights=models.MNASNet0_5_Weights.DEFAULT)
    elif model_name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif model_name == 'resnet101':
        model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    elif model_name == 'convnext_large':
        model = models.convnext_large(weights=models.ConvNeXt_Large_Weights.DEFAULT)
    else:
        raise ValueError(f"不支持的模型: {model_name}")
    
    # 修改模型输入层以适应单通道输入
    if in_channels != 3:
        if hasattr(model, 'conv1'):
            # ResNet, EfficientNet等模型
            original_conv1 = model.conv1
            model.conv1 = nn.Conv2d(
                in_channels, 
                original_conv1.out_channels,
                kernel_size=original_conv1.kernel_size,
                stride=original_conv1.stride,
                padding=original_conv1.padding,
                bias=original_conv1.bias is not None
            )
            # 初始化新卷积层
            if in_channels == 1:
                # 对于灰度图，取原始权重的平均值
                with torch.no_grad():
                    model.conv1.weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True)
            else:
                nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')
            
            print(f"修改了 {model_name} 的第一卷积层以适应 {in_channels} 通道输入")
        elif hasattr(model, 'features') and len(model.features) > 0:
            # SqueezeNet, MobileNet等模型
            first_layer = model.features[0]
            if hasattr(first_layer, 'in_channels') and first_layer.in_channels != in_channels:
                new_first_layer = nn.Conv2d(
                    in_channels,
                    first_layer.out_channels,
                    kernel_size=first_layer.kernel_size,
                    stride=first_layer.stride,
                    padding=first_layer.padding,
                    bias=first_layer.bias is not None
                )
                # 初始化
                if in_channels == 1:
                    with torch.no_grad():
                        new_first_layer.weight.data = first_layer.weight.data.mean(dim=1, keepdim=True)
                else:
                    nn.init.kaiming_normal_(new_first_layer.weight, mode='fan_out', nonlinearity='relu')
                
                model.features[0] = new_first_layer
                print(f"修改了 {model_name} 的第一卷积层以适应 {in_channels} 通道输入")
        elif hasattr(model, 'patch_embed'):
            # ViT模型 - 特殊处理
            if hasattr(model.patch_embed, 'proj'):
                original_proj = model.patch_embed.proj
                model.patch_embed.proj = nn.Conv2d(
                    in_channels,
                    original_proj.out_channels,
                    kernel_size=original_proj.kernel_size,
                    stride=original_proj.stride,
                    padding=original_proj.padding
                )
                # 初始化
                if in_channels == 1:
                    with torch.no_grad():
                        model.patch_embed.proj.weight.data = original_proj.weight.data.mean(dim=1, keepdim=True)
                else:
                    nn.init.kaiming_normal_(model.patch_embed.proj.weight, mode='fan_out', nonlinearity='relu')
                
                print(f"修改了 {model_name} 的patch embedding层以适应 {in_channels} 通道输入")
    
    # 冻结主干网络参数
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    # 尝试常见分类头位置
    if not head_replaced:
        head_locations = [
            ('classifier', 1),      # EfficientNet 系列
            ('fc', None),           # RegNet, ResNet 等
            ('heads.head', None),   # ViT 系列
            ('classifier', None),   # 其他模型
            ('head', None),         # 其他模型
            ('output', None)        # 其他模型
        ]
        
        for location, index in head_locations:
            try:
                # 分割路径以处理嵌套属性
                parts = location.split('.')
                current_module = model
                
                # 遍历路径直到目标模块
                for part in parts[:-1]:
                    current_module = getattr(current_module, part)
                
                # 获取目标模块
                target_module = getattr(current_module, parts[-1])
                
                # 处理索引（如果指定）
                if index is not None:
                    target_module = target_module[index]
                
                # 获取输入特征数
                if hasattr(target_module, 'in_features'):
                    in_features = target_module.in_features
                elif hasattr(target_module, 'in_channels'):
                    in_features = target_module.in_channels
                else:
                    raise AttributeError("无法确定输入特征数")
                
                # 创建新的分类头
                if isinstance(target_module, nn.Conv2d):
                    # 对于卷积层分类头
                    new_head = nn.Conv2d(
                        in_channels=in_features,
                        out_channels=num_classes,
                        kernel_size=target_module.kernel_size,
                        stride=target_module.stride,
                        padding=target_module.padding
                    )
                    nn.init.kaiming_normal_(new_head.weight, mode='fan_out', nonlinearity='relu')
                else:
                    # 对于线性层分类头
                    new_head = nn.Linear(in_features, num_classes)
                    # new_head = nn.Sequential(
                    #     nn.Dropout(0.3),  # 增加dropout
                    #     nn.Linear(in_features, num_classes)
                    # )
                    nn.init.kaiming_normal_(new_head.weight, mode='fan_out', nonlinearity='relu')
                
                # 替换分类头
                if index is not None:
                    # 处理索引替换 (如 classifier[1])
                    getattr(current_module, parts[-1])[index] = new_head
                else:
                    # 直接属性替换
                    setattr(current_module, parts[-1], new_head)
                
                print(f"成功替换 {model_name} 的分类头位置: {location}{f'[{index}]' if index is not None else ''}")
                head_replaced = True
                head_module = new_head
                break
            
            except (AttributeError, TypeError, IndexError) as e:
                # 如果当前路径无效，尝试下一个
                print(f"尝试位置 {location}{f'[{index}]' if index is not None else ''} 失败: {str(e)}")
                continue
    
    # 如果所有尝试都失败，尝试通用方法
    if not head_replaced:
        try:
            # 检查模型是否有分类头属性
            if hasattr(model, 'classifier'):
                # 尝试替换整个分类头
                if isinstance(model.classifier, nn.Sequential):
                    # 对于序列结构
                    last_layer = model.classifier[-1]
                    if hasattr(last_layer, 'in_features'):
                        in_features = last_layer.in_features
                        model.classifier[-1] = nn.Linear(in_features, num_classes)
                        head_module = model.classifier[-1]
                        head_replaced = True
                        print(f"成功替换 {model_name} 的分类头位置: classifier[-1]")
                else:
                    # 对于单个模块
                    if hasattr(model.classifier, 'in_features'):
                        in_features = model.classifier.in_features
                        model.classifier = nn.Linear(in_features, num_classes)
                        head_module = model.classifier
                        head_replaced = True
                        print(f"成功替换 {model_name} 的分类头位置: classifier")
            
            elif hasattr(model, 'fc'):
                # 尝试替换全连接层
                if hasattr(model.fc, 'in_features'):
                    in_features = model.fc.in_features
                    model.fc = nn.Linear(in_features, num_classes)
                    head_module = model.fc
                    head_replaced = True
                    print(f"成功替换 {model_name} 的分类头位置: fc")
        except Exception as e:
            print(f"通用方法失败: {str(e)}")
    
    if not head_replaced:
        raise RuntimeError(f"无法找到 {model_name} 的分类头位置")
    
    # 确保分类头需要梯度
    for param in head_module.parameters():
        param.requires_grad = True
    
    return model, head_module

def create_submission(model, test_dataset, device, output_file="submission.csv"):
    model.eval()
    predictions = []
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    with torch.no_grad():
        for images, _ in tqdm(test_loader, desc="生成预测结果"):
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
    
    # 创建提交文件
    submission_df = pd.DataFrame({
        'index': range(len(predictions)),
        'pred': predictions
    })
    
    submission_df.to_csv(output_file, index=False)
    print(f"提交文件已保存: {output_file}")
    
    return submission_df
   
def get_class_weights(labels):
    class_counts = Counter(labels)
    total_samples = len(labels)
    class_weights = {}
    
    for class_idx, count in class_counts.items():
        # 使用逆频率加权，给少数类更高权重
        class_weights[class_idx] = total_samples / (len(class_counts) * count)
    
    return torch.tensor([class_weights[i] for i in sorted(class_weights.keys())])

 
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    data_root = "./IS_2025_OrganAMNIST"
    train_dataset, val_dataset, test_dataset = load_datasets(data_root, image_size=224)
    # train_dataset, val_dataset, test_dataset = load_datasets(data_root, image_size=512) # For vit
    num_classes = len(categories)

    model, head_module = build_model('regnet_y_32gf', num_classes, freeze_backbone=True, in_channels=1)
    

    num_epochs = 50
    class_weights = get_class_weights(train_dataset.labels)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.1)
    optimizer = torch.optim.Adam(head_module.parameters(), lr=0.01, weight_decay=0.01) # 只优化分类头参数
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5
    )

    start_epoch, train_scores, validation_scores = load_checkpoint(
        model, optimizer, scheduler
    )
        
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True,
        num_workers=4, 
        pin_memory=True,      # 启用内存锁页加速GPU传输 
        persistent_workers=True  # 保持子进程存活避免重复初始化
    )
    test_loader = DataLoader(val_dataset, batch_size=32, num_workers=4, pin_memory=True, persistent_workers=True, shuffle=False)

    print(f"从 epoch {start_epoch + 1} 开始，共 {num_epochs} 轮")

    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        for images, labels in tqdm(train_loader, desc="Training..."):
            valid_mask = labels != -1
            if valid_mask.sum() == 0:
                continue
            images = images[valid_mask]
            labels = labels[valid_mask]
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Verification..."):
                valid_mask = labels != -1
                if valid_mask.sum() == 0:
                    continue
                images = images[valid_mask]
                labels = labels[valid_mask]
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total if train_total > 0 else 0
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        train_scores.append(train_acc)
        validation_scores.append(val_acc)
        
        result_text = f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {running_loss/len(train_loader):.4f} Train Acc: {train_acc:.2f}% Val Acc: {val_acc:.2f}%\n"
        print(result_text)
        
        scheduler.step()
        save_checkpoint(
            epoch, model, optimizer, scheduler,
            train_scores, validation_scores,
        )

    print("生成提交文件...")
    submission_df = create_submission(model, test_dataset, device)
    plot_train_valid(train_scores, validation_scores, num_epochs)
    
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        print("训练完成，检查点已删除")