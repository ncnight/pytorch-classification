import torch
from PIL import Image
import zfpy
from torch.utils.data import Dataset
from os.path import join
from os import listdir
import numpy as np
from PIL import Image
from torchvision import transforms

'''
structure will be 
{root_dir}/{jpeg, zfp, npy}/{train, test}/{class index}/{index.zfp, index.jpeg}
'''

class CIFAR10Dynamic(Dataset):
    
    def __init__(self, data_dir, mode='zfp', train=False, transforms=None):
        self.data_dir = data_dir
        self.transforms = transforms
        self.num_classes = 10
        self.train = train
        self.mode = mode # one of zfp, jpg, npy
        self._load_data_paths()
    
    def _load_data_paths(self):
        self.data = []
        self.labels = []
        folder = 'train' if self.train else 'test'
        for i in range(self.num_classes):
            for f in listdir(join(self.data_dir, self.mode, folder, str(i))):
                if f.split('.')[-1] == self.mode:
                    self.data.append(join(self.data_dir, self.mode, folder, str(i), f))
                    self.labels.append(i)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, y = self.data[idx], self.labels[idx]
        if self.mode == 'zfp':
            with open(img_path, 'rb') as fp:
                img = fp.read()
            img = zfpy.decompress_numpy(img)
            img = np.transpose(img, (1, 2, 0))
            img = Image.fromarray(img, mode='RGB')
        elif self.mode == 'npy':
            img = np.load(img_path)
            img = np.transpose(img, (1, 2, 0))
            img = Image.fromarray(img, mode='RGB')
        elif self.mode == 'jpg':
            img = Image.open(img_path)
        
        if self.transforms is not None:
            img = self.transforms(img)

        return img, y

def load_cifar(mode, root_dir):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if mode == 'zfp':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
    train_set = CIFAR10Dynamic(root_dir, mode=mode, train=True, transforms=transform_train)
    test_set = CIFAR10Dynamic(root_dir, mode=mode, train=False, transforms=transform_test)
    return train_set, test_set
