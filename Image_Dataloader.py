import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from Image_Dataset import Train_Valid_Image_Dataset, Test_Dataset 
import gc 

class Loader:
    def __init__(self, paths):
        self.paths = paths
    
    def set_transform_list(self):
        self.transform_list = [
            transforms.Resize((224, 224)), 
            transforms.ToTensor()
        ]
        
    def set_transform(self):
        self.transform = transforms.Compose(self.transform_list)
        
    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

class Train_Loader(Loader):
    def __init__(self, paths, labels):
        super().__init__(paths)
        self.labels = labels 
        
    def set_transform_list(
        self, 
        rotation_angle, brightness, 
        contrast, saturation, hue
    ):
        super().set_transform_list()
        self.transform_list = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(rotation_angle),
            transforms.ColorJitter(
                brightness=brightness, contrast=contrast, 
                saturation=saturation, hue=hue    
            )
        ] + self.transform_list
        
    def find_max_batch_size(self, model, device):
        batch_size = 1
        step = 128  # Starting step
        data = torch.randn((batch_size, *self.train_dataset[0][0].shape)).to(device)
        while True:
            try:
                model(data)  # Forward pass
                batch_size += step
                data = torch.randn((batch_size, *self.train_dataset[0][0].shape)).to(device)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    torch.cuda.empty_cache()
                    step = step // 2  # Decrease step size
                    if step == 0:  # Converged
                        break
                else:
                    raise e
        torch.cuda.empty_cache()
        del data
        gc.collect()
        return batch_size - step
    
    def set_dataset(self):
        self.train_dataset = Train_Valid_Image_Dataset(self.paths, self.labels, transform=self.transform)
        
    def set_dataloader(self, workers):
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, 
            shuffle=True, num_workers=workers, pin_memory=True
        )
        
class Valid_Loader(Loader):
    def __init__(self, paths, labels):
        super().__init__(paths) 
        self.labels = labels 
     
    def set_dataset(self):
        self.valid_dataset = Train_Valid_Image_Dataset(self.paths, self.labels, transform=self.transform)
           
    def set_dataloader(self, workers):
        self.valid_loader = DataLoader(
            self.valid_dataset, batch_size=self.batch_size, 
            shuffle=False, num_workers=workers, pin_memory=True
        ) 
        
class Test_Loader(Loader):
    def __init__(self, paths):
        super().__init__(paths) 
    
    def set_dataset(self):
        self.test_dataset = Test_Dataset(self.paths, transform=self.transform)
           
    def set_dataloader(self, workers):
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=self.batch_size, 
            shuffle=False, num_workers=workers, pin_memory=True
        ) 