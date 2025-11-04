import os
import numpy as np
from PIL import Image
import pandas as pd
from collections import Counter
import torch
from torch.utils.data import Dataset, ConcatDataset, random_split
from torchvision import transforms
import cv2

torch.manual_seed(2019)
np.random.seed(2019)

categories = [
    "Spleen", "Right kidney", "Left kidney", 
    "Gallbladder", "Esophagus", "Liver", "Stomach", 
    "Aorta", "Inferior vena cava", "Pancreas", "Right adrenal gland"
]
label_to_idx = {label: idx for idx, label in enumerate(categories)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}

def normalize_ct(image):
    """
    将CT值归一化到[0,1]范围，保留医学意义
    """
    # CT值范围（Hounsfield单位）
    min_val = -1000  # 空气
    max_val = 3000   # 骨骼
    
    # 裁剪到医学相关范围
    image = np.clip(image, min_val, max_val)
    # 归一化
    # image = (image - min_val) / (max_val - min_val)
    image = np.log1p(image - min_val + 1)
    # 归一化到[0,1]
    image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
    return image

def enhance_vessels(image):
    """血管增强处理"""
    # 使用Frangi滤波器增强管状结构
    from skimage.filters import frangi
    vessel_enhanced = frangi(image, sigmas=range(1, 4, 1))
    
    # 融合原始图像和增强结果
    enhanced = 0.7 * image + 0.3 * vessel_enhanced
    return np.clip(enhanced, 0, 1)

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8,8)):
    """
    应用CLAHE增强对比度
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image.astype(np.uint8))

class CustomDataset(Dataset):
    def __init__(self, image_dir, label_file=None, transform=None, test_index_file=None):
        self.image_dir = image_dir
        self.transform = transform
        self.img_paths = []
        self.labels = []
        
        image_files = [
            f for f in os.listdir(image_dir) 
            if f.lower().endswith('.png')
        ]
        
        if label_file is not None:
            labels_df = pd.read_csv(label_file)
            filename_to_label = dict(zip(labels_df['file'], labels_df['label']))
            
            for img_file in image_files:
                if img_file in filename_to_label:
                    self.img_paths.append(os.path.join(image_dir, img_file))
                    self.labels.append(filename_to_label[img_file])
        else:
            if test_index_file:
                index_df = pd.read_csv(test_index_file)
                index_df = index_df.sort_values('index')
                for _, row in index_df.iterrows():
                    img_file = row['file']
                    img_path = os.path.join(image_dir, img_file)
                    if os.path.exists(img_path):
                        self.img_paths.append(img_path)
                        self.labels.append(-1)  # 测试集标签设为-1
                    else:
                        raise FileNotFoundError(f"测试图像未找到: {img_path}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('L')
        label = self.labels[idx]
        
        image_np = np.array(image, dtype=np.float32)
        image_np = normalize_ct(image_np)
        # image_np = apply_clahe(image_np)
        # image_np = enhance_vessels(image_np)
        image_np = np.clip(image_np, 0.0, 1.0)
        image = Image.fromarray((image_np * 255).astype(np.uint8))
        
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(image_size=224):
    # 训练专用变换（包含增强）
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomAffine(
            degrees=5,  # 减小旋转角度
            translate=(0.05, 0.05),
            scale=(0.95, 1.05)  # 添加轻微缩放
        ),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2),
        # transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), value='random'),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # 验证/测试专用变换（无增强）
    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    return train_transform, val_transform

def load_datasets(data_root, image_size=224, merge_train_val=False, k_fold=None):
    train_dir = os.path.join(data_root, "train", "images_train")
    val_dir = os.path.join(data_root, "val", "images_val")
    test_dir = os.path.join(data_root, "test", "images")
    
    # 假设标签文件在相应目录中
    train_label_file = os.path.join(data_root, "train", "labels_train.csv")
    val_label_file = os.path.join(data_root, "val", "labels_val.csv")
    test_idx_file = os.path.join(data_root, "test", "manifest_public.csv")
    
    # 获取变换
    train_transform, val_transform = get_transforms(image_size)
    
    # 加载数据集
    train_dataset = CustomDataset(train_dir, label_file=train_label_file, transform=train_transform)
    val_dataset = CustomDataset(val_dir, label_file=val_label_file, transform=val_transform)
    test_dataset = CustomDataset(test_dir, transform=val_transform, test_index_file=test_idx_file)
    
    if merge_train_val:
        full_dataset = ConcatDataset([train_dataset, val_dataset])
        
        if k_fold:
            print(f"使用 {k_fold}折交叉验证")
            return full_dataset, test_dataset, k_fold
        else:
            # 按8:2比例分割
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            train_dataset, val_dataset = random_split(
                full_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(2019)
            )
            print(f"合并后数据集分割: 训练集={len(train_dataset)}, 验证集={len(val_dataset)}")
    
    print(f"训练集: {len(train_dataset)} 张图片")
    print(f"验证集: {len(val_dataset)} 张图片")
    print(f"测试集: {len(test_dataset)} 张图片")
    
    # 打印类别分布
    if hasattr(train_dataset, 'labels') and -1 not in train_dataset.labels:
        label_counts = Counter(train_dataset.labels)
        print("训练集类别分布:")
        for label_idx, count in label_counts.items():
            label_name = idx_to_label.get(label_idx, f"未知({label_idx})")
            print(f"  {label_name}: {count} 张")
            
    return train_dataset, val_dataset, test_dataset

