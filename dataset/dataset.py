import torch
from torch.utils import data
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import csv
import cv2

# subsets = ['living_room', 'bedroom', 'kitchen', 'dining_room',
#             'bathroom', 'home_office', 'house']

class SceneParsingDataset(data.Dataset):
    '''
        Dataset subclass to load MIT Scene Parsing Dataset
        Parameters
        ---------
        img_path : str
            relative path to image directory
        Attributes
        ---------
        img_path : list
            list of relative paths to each images
        '''
    def __init__(self, img_path, is_train=True):
        self.img_path = [os.path.join(img_path, x) for x in os.listdir(img_path)]

        self.is_train = is_train

    def __getitem__(self, i):
        img = Image.open(self.img_path[i]).convert('RGB')
        seg_path = self.get_annotation_path(self.img_path[i])
        seg = Image.open(seg_path)
        img, seg = self.transform(img, seg)
        return img, seg

    def __len__(self):
        return len(self.img_path)

    def get_annotation_path(self, img_path):
        # annotations/name.png 형태로 들어있다.
        seg_path = img_path.replace('images','annotations').replace('.jpg', '.png')
        return seg_path

    def get_new_size(self, img, max_dim=800):
        w, h = img.size
        ratio = 1.0
        if w > max_dim:
            ratio = max_dim / w
        if h > max_dim:
            ratio = min(max_dim / h, ratio)
        nw = int(w * ratio + 0.5)
        nh = int(h * ratio + 0.5)
        return nw, nh

    def transform(self, img, seg):
        '''
        Perform transformations on image and segmentation. The transformation
        must semantically match between image and segmentation.
        Parameters
        ---------
        img : PIL Image
            image
        seg : PIL Image
            segmentation
        Returns
        ---------
        img : torch.Tensor
            image after transformations
        seg : torch.Tensor
            segmentation matched with image
        '''
        w, h = self.get_new_size(img, max_dim=200)
        #         seg = torch.from_numpy(np.array(seg.resize((w, h)))).long()
        #         seg = torch.from_numpy(np.array(seg)).long()
        seg = transforms.Resize((h, w))(seg)
        img = transforms.Resize((h, w))(img)

        if self.is_train:
            if np.random.rand() > 0.5:
                seg = transforms.functional.hflip(seg)
                img = transforms.functional.hflip(img)

            if np.random.rand() > 0.5:
                img = transforms.functional.adjust_brightness(img, np.random.uniform(0.8, 1.2))

            if np.random.rand() > 0.5:
                img = transforms.functional.adjust_gamma(img, np.random.uniform(0.8, 1.2))

        seg = torch.from_numpy(np.array(seg)).long()
        seg -= 1
        img = transforms.ToTensor()(img)

        return img, seg


if __name__ == '__main__':
    # Load datasets
    root_path = '../data/ADEChallengeData2016/images/'
    folders = {
        'train': 'training',
        'val': 'validation'
    }
    datasets = {
        split: SceneParsingDataset(os.path.join(root_path, folder), is_train=split == 'train')
        for split, folder in folders.items()
    }
    print(datasets)

    datasizes = {
        split: len(datasets[split])
        for split, folder in folders.items()
    }
    print(datasizes)

    dataloaders = {
        split: data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=12)
        for split, dataset in datasets.items()
    }
    print(dataloaders['train'])


    # -1~150 labeling
    for (img, seg) in dataloaders['val']:
        print(img.size(), seg.size())
    #     print(seg.size())
    #     seg_np=seg.view(seg.size(1),seg.size(2),1).numpy()
    #     seg_np=np.clip(seg_np, 0, 255)
    #     print(seg_np)
    #     cv2.imshow('123',seg_np/255)
    #     cv2.waitKey(0)
