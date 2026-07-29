from PIL import Image
from torch.utils.data import Dataset

class Image_Dataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def get_image(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

class Train_Valid_Image_Dataset(Image_Dataset):
    def __init__(self, image_paths, labels, transform=None):
        super().__init__(image_paths, transform)
        self.labels = labels

    def __getitem__(self, idx):
        label = self.labels[idx]
        return self.get_image(idx), label
    
class Test_Dataset(Image_Dataset):
    def __init__(self, image_paths, transform=None):
        super().__init__(image_paths, transform)

    def __getitem__(self, idx):
        return self.get_image(idx)