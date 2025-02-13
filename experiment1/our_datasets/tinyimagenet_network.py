from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import pickle
from torch.utils.data import DataLoader
import torchvision.transforms as tfs
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset
import urllib.request
import zipfile
from tqdm import tqdm
import shutil
from typing import Dict, List
import glob

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url: str, output_path: str):
    with DownloadProgressBar(unit='B', unit_scale=True,
                           miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

class TinyImageNet(Dataset):
    base_url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    
    def __init__(self, root="./data", train=True, transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train
        self.tiny_imagenet_dir = '/mnt/csp/mmvision/home/lwh/DLS/'
        
        if download:
            self.download()
            
        if not self._check_exists():
            raise RuntimeError('Dataset not found. Use download=True to download it')
            
        # Load the processed data
        data_file = "tiny-imagenet_train.pkl" if train else "tiny-imagenet_val.pkl"
        data_path = os.path.join(self.tiny_imagenet_dir, data_file)
        
        with open(data_path, "rb") as f:
            dat = pickle.load(f)
        self.data = dat["data"]
        self.targets = dat["targets"]

    def __getitem__(self, item):
        data, targets = Image.fromarray(self.data[item]), self.targets[item]
        if self.transform is not None:
            data = self.transform(data)
        return data, targets

    def __len__(self):
        return len(self.data)
    
    def _check_exists(self) -> bool:
        train_path = os.path.join(self.tiny_imagenet_dir, "tiny-imagenet_train.pkl")
        val_path = os.path.join(self.tiny_imagenet_dir, "tiny-imagenet_val.pkl")
        return os.path.exists(train_path) and os.path.exists(val_path)

    def _process_train_data(self, train_dir: str) -> tuple:
        print("Processing training data...")
        classes = sorted(os.listdir(train_dir))
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        
        images = []
        labels = []
        
        for class_dir in tqdm(classes):
            class_path = os.path.join(train_dir, class_dir, 'images')
            class_idx = class_to_idx[class_dir]
            
            for img_path in glob.glob(os.path.join(class_path, '*.JPEG')):
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_array = np.array(img)
                    images.append(img_array)
                    labels.append(class_idx)
                except:
                    print(f"Error loading image: {img_path}")
                    continue
                
        return np.stack(images), np.array(labels)

    def _process_val_data(self, val_dir: str) -> tuple:
        print("Processing validation data...")
        # Read validation annotations
        val_anno_path = os.path.join(val_dir, 'val_annotations.txt')
        with open(val_anno_path, 'r') as f:
            val_anno = f.readlines()
        
        # Create mapping from image filename to class
        img_to_class = {line.split('\t')[0]: line.split('\t')[1] for line in val_anno}
        
        # Get sorted list of classes (same order as training)
        classes = sorted(set(img_to_class.values()))
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        
        images = []
        labels = []
        
        val_images_dir = os.path.join(val_dir, 'images')
        for img_name in tqdm(os.listdir(val_images_dir)):
            img_path = os.path.join(val_images_dir, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                img_array = np.array(img)
                class_name = img_to_class[img_name]
                class_idx = class_to_idx[class_name]
                
                images.append(img_array)
                labels.append(class_idx)
            except:
                print(f"Error loading image: {img_path}")
                continue
                
        return np.stack(images), np.array(labels)
    
    def download(self):
        if self._check_exists():
            print('Files already exist')
            return

        os.makedirs(self.tiny_imagenet_dir, exist_ok=True)
        
        # Download
        zip_path = os.path.join(self.root, "tiny-imagenet-200.zip")
        if not os.path.exists(zip_path):
            print('Downloading TinyImageNet...')
            download_url(self.base_url, zip_path)
        
        # Extract
        print('Extracting...')
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.root)
        
        extracted_dir = os.path.join(self.root, "tiny-imagenet-200")
        
        # Process and save as pickle
        print('Processing dataset...')
        
        # Process training data
        train_data, train_labels = self._process_train_data(
            os.path.join(extracted_dir, 'train')
        )
        train_dict = {
            'data': train_data,
            'targets': train_labels
        }
        with open(os.path.join(self.tiny_imagenet_dir, 'tiny-imagenet_train.pkl'), 'wb') as f:
            pickle.dump(train_dict, f)
        
        # Process validation data
        val_data, val_labels = self._process_val_data(
            os.path.join(extracted_dir, 'val')
        )
        val_dict = {
            'data': val_data,
            'targets': val_labels
        }
        with open(os.path.join(self.tiny_imagenet_dir, 'tiny-imagenet_val.pkl'), 'wb') as f:
            pickle.dump(val_dict, f)
        
        # Clean up
        print('Cleaning up...')
        os.remove(zip_path)
        shutil.rmtree(extracted_dir)
        
        print('Done!')

def load_tinyimagenet(
    root,
    transforms=None,
    image_size=32,
    train_batch_size=64,
    valid_batch_size=64,
    split=0.95,
    seed=42,
    rank=0,
    debug=False,
    return_dataloader=False,
):
    if transforms is None:
        transforms = tfs.Compose(
            [
                tfs.Resize((image_size, image_size)),
                tfs.ToTensor(),
                tfs.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    if train_batch_size is None:
        train_batch_size = 1
    if split is None:
        split = 0.8
        
    train_set = TinyImageNet(root, True, transforms, download=False)
    test_set = TinyImageNet(root, False, transforms, download=False)
    
    valid_subset = None
    if split < 1.0:
        labels = train_set.targets
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=1 - split, random_state=seed
        )
        for train_idx, val_idx in splitter.split(range(len(labels)), labels):
            train_subset = Subset(train_set, train_idx)
            valid_subset = Subset(train_set, val_idx)
    else:
        train_subset = train_set

    test_loader = DataLoader(test_set, batch_size=valid_batch_size, drop_last=True)
    if return_dataloader:
        train_loader = DataLoader(
            train_subset, batch_size=train_batch_size, shuffle=True, drop_last=True
        )
        valid_loader = (
            DataLoader(valid_subset, batch_size=valid_batch_size, drop_last=True)
            if valid_subset is not None
            else None
        )

        return train_loader, valid_loader, (3, image_size, image_size), 200
    return train_subset, valid_subset, (3, image_size, image_size), 200