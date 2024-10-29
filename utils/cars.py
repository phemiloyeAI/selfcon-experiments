import os
import cv2
import torch
from PIL import Image 
from pathlib import Path
from scipy.io import loadmat
from torch.utils.data import Dataset


root_dir = Path('/content/drive/MyDrive/Experimental-Results-Reproduction/Self-Contrasitive-Learning/self-contrastive-learning/datasets/standfordCars')
cars_annos = root_dir / 'cars_annos.mat'
cars_test = root_dir / 'cars_test' / 'cars_test'
cars_train = root_dir / 'cars_train' / 'cars_train'

cars_annos_mat = loadmat(cars_annos)
training_images = os.listdir(cars_train)
testing_images = os.listdir(cars_test)

print(cars_annos_mat['annotations'][0])


# root_dir = Path("/content/drive/MyDrive/Experimental-Results-Reproduction/Self-Contrasitive-Learning/self-contrastive-learning/datasets/standfordCars-meta/")
# cars_annos_train = root_dir / "devkit" / "cars_train_annos.mat"
# cars_annos_test = root_dir / "cars_test_annos_withlabels (1).mat"

# cars_meta_mat = loadmat(root_dir / "devkit" / "cars_meta.mat")
# cars_annos_train_mat, cars_annos_test_mat = loadmat(cars_annos_train), loadmat(cars_annos_test)

# class_names = [arr[0] for arr in cars_meta_mat['class_names'][0]]

# training_image_label_dictionary, testing_image_label_dictionary = {}, {}

# for arr in cars_annos_train_mat['annotations'][0]:
#     image, label = arr[-1][0], arr[-2][0][0] - 1
#     training_image_label_dictionary[image] = label

# for arr in cars_annos_test_mat['annotations'][0]:
#     image, label = arr[-1][0], arr[-2][0][0] - 1
#     testing_image_label_dictionary[image] = label


# print(len(training_image_label_dictionary), len(testing_image_label_dictionary))

class StanfordCarsCustomDataset(Dataset):
    def __init__(self, directory, image_label_dict, transforms):
        super().__init__()

        self.images = [os.path.join(directory, f) for f in os.listdir(directory)]
        self.transforms = transforms
        self.image_label_dict = image_label_dict

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Get image
        image = self.images[index]
        img_pil = Image.open(image).convert('RGB')
        img_trans = self.transforms(img_pil)

        # Parse out the label from cars_meta and cars_x_annos files
        image_stem = image.split("/")[-1]
        img_label = self.image_label_dict[image_stem]

        return img_trans, img_label