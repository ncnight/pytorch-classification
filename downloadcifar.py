import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from zfpy import compress_numpy
from os.path import join

data_dir = './data/profiling'

print('Downloading data...')
normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
transform = transforms.Compose(
    [transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=False, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

img_count = 0
for X, y in trainloader:
    for i in range(X.shape[0]):
        img = X[i]
        label = y[i].item()
        
        # numpy 
        img_np = img.numpy()
        np.save(join(data_dir, 'npy', 'train', str(label), str(img_count)), img_np)

        # zfp
        buf = compress_numpy(normalize(img).numpy(), precision=7)
        with open(join(data_dir, 'zfp', 'train', str(label), str(img_count)) + '.zfp', 'wb+') as fp:
            fp.write(buf)
        
        # jpg
        im = Image.fromarray(img.numpy(), mode='RGB')
        im.save(join(data_dir, 'jpg', 'train', str(label), str(img_count) + '.jpg'))

        img_count += 1
img_count = 0
for X, y in testloader:
    for i in range(X.shape[0]):
        img = X[i]
        label = y[i].item()
        
        # numpy 
        img_np = img.numpy()
        np.save(join(data_dir, 'npy', 'test', str(label), str(img_count)), img_np)

        # zfp
        buf = compress_numpy(normalize(img).numpy(), precision=7)
        with open(join(data_dir, 'zfp', 'test', str(label), str(img_count)) + '.zfp', 'wb+') as fp:
            fp.write(buf)
        
        # jpg
        im = Image.fromarray(img.numpy(), mode='RGB')
        im.save(join(data_dir, 'jpg', 'test', str(label), str(img_count) + '.jpg'))

        img_count += 1
