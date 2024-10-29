import os
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset

class FGVCAircraft(Dataset):
  def __init__(self, img_path, metadata_path, transform):
    super().__init__()
    self.img_path = img_path
    self.transform = transform
    self.metadata = pd.read_csv(metadata_path)
  
  def __len__(self):
    return self.metadata.shape[0]
  
  def __getitem__(self, index):
    img_name = self.metadata['filename'].iloc[index]
    label = int(self.metadata['Labels'].iloc[index])

    img_path = os.path.join(self.img_path, img_name)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(cv2.resize(img, dsize=(64, 64), interpolation=cv2.INTER_AREA).transpose((2, 0, 1))).to(dtype=torch.float32)
  

    # img = img.transpose(2, 0, 1)
    if self.transform:
      img = self.transform(img)
    
    return img, label 


