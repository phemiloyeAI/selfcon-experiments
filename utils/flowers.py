import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

class FlowersDataset(Dataset):

    def __init__(self, image_path, csv_label_path, transform=None):
        self.image_path = image_path
        self.transform = transform

        self.df = pd.read_csv(csv_label_path, sep=',')
        self.targets = self.df['class_id'].to_numpy()
        self.image_names = self.df['dir'].to_numpy()
    
    def __getitem__(self, index):
        img_name = self.image_names[index]
        image = Image.open(os.path.join(self.image_path, img_name))

        if image.mode != 'RGB':
            image = image.convert('RGB')
        # image = image.resize((224, 224))
        image =  ToTensor()(image)

        if self.transform != None:
            image = self.transform(image)
        
        label = self.targets[index]

        return image, label

    def __len__(self):
        return self.image_names.shape[0]