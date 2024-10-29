import torch
from torch.utils.data import Dataset

class CUB(Dataset):
  def __init__(self, imgz_tensors_pth, metadata_pth, transform, split='train'):
    super().__init__()

    self.imgz_tensor = torch.load(imgz_tensors_pth, map_location='cpu')
    metadata = torch.load(metadata_pth, map_location='cpu')

    self.img_id_to_class_id = metadata['img_id_to_class_id']
    
    if split == 'train':
      self.img_ids = metadata['img_ids']  

    if split == 'val':
      self.img_ids = metadata['train_val_img_ids']

    if split == 'test':
      self.img_ids = metadata['test_img_ids']
    
    self.transform = transform
  
  def __len__(self):
    return self.imgz_tensor.shape[0]
  
  def __getitem__(self, index):
    img = self.imgz_tensor[index].to(dtype=torch.float32)

    img_id = self.img_ids[index]
    class_id = self.img_id_to_class_id[img_id]

    if self.transform != None:
      img = self.transform(img)
  
    
    return img, class_id