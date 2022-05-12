#!/usr/bin/env python
from mimetypes import init
import os
import pprint
import time
import warnings
import PIL
import numpy as np
import argparse
import torch
import torchvision
import torch.utils.data.distributed
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms
from torchvision import datasets
from termcolor import cprint
import cv2
from csv import writer
from pathlib import Path

from inference.network_inf import builder_inf
from list_gen import preprocess_faces
from utils import utils
import sys
sys.path.append("..")
sys.path.append("../../")


# parse the args


class ImgInfLoader(data.Dataset):
    def __init__(self, ann_file, transform=None):
        self.ann_file = ann_file
        self.transform = self.get_transformer()
        cprint('=> preparing dataset for inference ...')
        self.init()

    def get_transformer(self):
        return transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.]),
        ])

    def init(self):
        with open(self.ann_file) as f:
            self.imgs = f.readlines()

    def __getitem__(self, index):
        ls = self.imgs[index].strip().split()
        # change here
        img_path = ls[0]
        if not os.path.isfile(img_path):
            raise Exception('{} does not exist'.format(img_path))
            exit(1)
        img = cv2.imread(img_path)
        if img is None:
            raise Exception('{} is empty'.format(img_path))
            exit(1)
        _img = cv2.flip(img, 1)
        _img = PIL.Image.fromarray(_img)
        img = PIL.Image.fromarray(img)
        return [self.transform(img), self.transform(_img)], img_path

    def __len__(self):
        return len(self.imgs)


class Magface:

    def __init__(self) -> None:

        # '/home/chaki/Projects/MagFace/weights/magface_epoch_00025.pth'
        self.resume = "/home/chaki/Projects/MagFace/tmp/00025.pth"
        self.cpu_mode = False
        self.print = 100
        self.embedding_size = 512
        self.batch_size = 1
        self.workers = 4
        self.arch = 'iresnet50'
        self.load_model()

    def load_model(self):
        model = builder_inf(self)
        if not self.cpu_mode:
            model = model.cuda()
        self.model = model.eval()


def process_faces(path: str):

    cprint('=> modeling the network ...', 'green')
    magFace = Magface()
    cprint('=> building the dataloader ...', 'green')
    cprint('=> starting inference engine ...', 'green')

    dirs = [os.path.join(path, dir) for dir in os.listdir(
        path) if os.path.isdir(os.path.join(path, dir))]
    columns = ["id", "path", "feature_512"]
    fold = Path("db_files/") / "data_not_normalized.csv"
    utils.append_list_as_row(fold, columns)
    for directory in dirs:
        id = directory.split("/")[-1]

        feat_list = os.path.join(directory, "feat.list")
        inf_list = os.path.join(directory, "img.list")

        cprint('=> embedding features will be saved into {}'.format(fold))

        inf_dataset = ImgInfLoader(ann_file=inf_list)
        inf_loader = torch.utils.data.DataLoader(
            inf_dataset, batch_size=1, pin_memory=True, shuffle=False)

        batch_time = utils.AverageMeter('Time', ':6.3f')
        data_time = utils.AverageMeter('Data', ':6.3f')
        progress = utils.ProgressMeter(
            len(inf_loader), [batch_time, data_time], prefix="Extract Features: ")

        # switch to evaluate mode
        with torch.no_grad():
            end = time.time()

            for i, (input, img_paths) in enumerate(inf_loader):
                # measure data loading time
                data_time.update(time.time() - end)
                image = input[0].cuda()
                # compute output
                embedding_feat = magFace.model(image)
                # embedding_feat = F.normalize(embedding_feat, p=2, dim=1)

                _feat = embedding_feat.data.cpu().numpy()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % 10 == 0:
                    progress.display(i)

                person_info = [id, img_paths[0], list(_feat[0])]
                utils.append_list_as_row(fold, person_info)

    # close


if __name__ == '__main__':
    cprint('=> parse the args ...', 'green')
    path = "/home/chaki/Projects/gods_eye/output/ptz_good_test_2/"
    preprocess_faces(path)
    process_faces(path)
