import os
import requests
import zipfile
import numpy as np
from torch.utils.data import Dataset, Subset
from torchvision.transforms import Compose, ToTensor
from sklearn.model_selection import train_test_split
from PIL import Image

def download_dataset_if_not_exist(dataset_url, target_path):
    if not os.path.exists(target_path):
        print("Downloading dataset ...")
        os.makedirs(target_path)

        r = requests.get(dataset_url, stream=True)
        zip_path = os.path.join(target_path, "dataset.zip")
        with open(zip_path, "wb") as zip_file:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    zip_file.write(chunk)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(target_path)

        os.remove(zip_path)

class ClassificationDataset(Dataset):
    def __init__(self, dataset_root_dir, dataset_name, dataset_url, classes_labels, transforms=None):
        download_dataset_if_not_exist(dataset_url, dataset_root_dir)
        self.dataset_dir = os.path.join(dataset_root_dir, dataset_name)
                
        self.classes = classes_labels
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.image_paths, self.labels = self.load_dataset()
        self.transforms = Compose([ToTensor()])

    def load_dataset(self):
        image_paths = []
        labels = []

        for class_name in self.classes:
            class_dir = os.path.join(self.dataset_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                image_paths.append(img_path)
                labels.append(self.class_to_idx[class_name])
                
        return image_paths, labels
    
    def set_additional_transforms(self, transforms):
        self.transforms = Compose([ToTensor()] + transforms) if transforms is not None else Compose([ToTensor()])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path)
        if self.transforms:
            image = self.transforms(image)

        return image, label
    
    @staticmethod
    def stratified_split(dataset, test_size, valid_size, random_state=None):
        labels = np.array(dataset.labels)

        train_valid_idx, test_idx = train_test_split(
            np.arange(len(labels)),
            test_size=test_size,
            stratify=labels,
            random_state=random_state
        )

        train_idx, valid_idx = train_test_split(
            train_valid_idx,
            test_size=valid_size / (1.0 - test_size),
            stratify=labels[train_valid_idx],
            random_state=random_state
        )

        train_dataset = Subset(dataset, train_idx)
        valid_dataset = Subset(dataset, valid_idx)
        test_dataset = Subset(dataset, test_idx)

        return train_dataset, valid_dataset, test_dataset