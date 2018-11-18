import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
from torchvision.datasets.folder import  default_loader
import torchvision.transforms as transforms

import joint_transforms

# mean and std
mean = [0.492, 0.463, 0.397]
std = [0.274, 0.266, 0.284]


def _make_dataset(dir):
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            path = os.path.join(root, fname)
            item = path
            images.append(item)
    return images


def _make_image_namelist(dir):
    images = []
    namelist = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            item_name = fname
            namelist.append(item_name)
            item_path = os.path.join(root, fname)
            images.append(item_path)
    return images, namelist


class LabelToLongTensor(object):
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            label = torch.from_numpy(pic).long()
        else:
            label = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            label = label.view(pic.size[1], pic.size[0], 1)
            label = label.transpose(0, 1).transpose(0, 2).squeeze().contiguous().long()
        return label


class Saliency(data.Dataset):

    def __init__(self, root, split='TRain', joint_transform_img=None,
                 mask_size_list = None,
                 transform=None, target_transform=LabelToLongTensor(),
                 download=False,
                 loader=default_loader):
        self.root = root
        assert split in ('TRain', 'VAl', 'test')
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform_img = joint_transform_img
        self.mask_size_list = mask_size_list
        self.loader = loader
        self.mean = mean
        self.std = std
        self.imgs = _make_dataset(os.path.join(self.root, self.split))

    def __getitem__(self, index):
        path = self.imgs[index]
        img = Image.open(path)
        ann_path = path.replace(self.split, self.split + 'annot').replace('.jpg', '.png')
        target = Image.open(ann_path)
        target = target.point(lambda i: i / 255)

        if self.joint_transform_img is not None:
            [img, target] = self.joint_transform_img([img, target])
        if self.transform is not None:
            img = self.transform(img)

        targets = []
        if self.mask_size_list is not None:
            for size in self.mask_size_list:
                transform_maks = transforms.Compose([joint_transforms.MaskResize(size)])
                [target_c] = transform_maks([target])
                targets.append(self.target_transform(target_c))

        return img, targets

    def __len__(self):
        return len(self.imgs)


class TestData(data.Dataset):

    def __init__(self, root, split='VAl', joint_transform_img=None,
                 transform=None, target_transform=None):
        self.root = root
        assert split in ('TRain', 'VAl', 'test')
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform_img = joint_transform_img
        self.mean = mean
        self.std = std
        self.imgs = _make_dataset(os.path.join(self.root, self.split))

    def __getitem__(self, index):
        path = self.imgs[index]
        img = Image.open(path)
        ann_path = path.replace(self.split, self.split + 'annot').replace('.jpg', '.png')
        target = Image.open(ann_path)

        if self.joint_transform_img is not None:
            [img] = self.joint_transform_img([img])
        if self.transform is not None:
            img = self.transform(img)

        transform_maks = transforms.Compose([joint_transforms.MaskResize(112)])
        [target_c] = transform_maks([target])
        target = self.target_transform(target_c)

        return img, target

    def __len__(self):
        return len(self.imgs)


class TestImage(data.Dataset):

    def __init__(self, root, split='train', joint_transform=None,
                 transform=None, target_transform=LabelToLongTensor(),
                 download=False,
                 loader=default_loader):
        self.root = root
        assert split in ('train', 'VAl', 'test')
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform
        self.loader = loader
        self.mean = mean
        self.std = std
        self.imgs, self.nlist = _make_image_namelist(os.path.join(self.root, self.split))

    def __getitem__(self, index):
        path = self.imgs[index]
        img = Image.open(path)
        nlist = self.nlist[index]
        # target = Image.open(path.replace(self.split, self.split + 'annot'))

        if self.joint_transform is not None:
            [img] = self.joint_transform([img])

        if self.transform is not None:
            img = self.transform(img)

        # target = self.target_transform(target)
        return img, nlist

    def __len__(self):
        return len(self.imgs)

