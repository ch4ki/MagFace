#!/usr/bin/env python
import dlib
from mimetypes import init
import os
import pprint
import time
import warnings
import PIL
import numpy as np
import argparse
import pandas as pd
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
from torch import nn
from tqdm import tqdm
from inference.network_inf import builder_inf
from utils import utils
import sys

sys.path.append("..")
sys.path.append("../../")


def preprocess_faces(path):
    peoples_dir = path  # os.path.join(os.getcwd(), path)
    peoples = os.listdir(path)

    save_path = "img_list"
    for people in peoples:
        person = os.path.join(peoples_dir, people)
        fio = open(f"{save_path}/img.list", 'a+')
        if person.endswith((".png", ".jpg")):
            fio.write('{} \n'.format(person))
        fio.close()


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
        return [self.transform(img), self.transform(_img)], os.path.basename(img_path).split(".")[0]

    def __len__(self):
        return len(self.imgs)


detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(
    "/home/chaki/Projects/dlib_face_recognition/shape_predictor_5_face_landmarks.dat")


class Magface:

    def __init__(self) -> None:
        # "/home/chaki/Projects/MagFace/tmp/00025.pth"
        # '/home/chaki/Projects/MagFace/weights/magface_epoch_00025.pth'
        self.resume = "tmp/magface_epoch_00025.pth"
        self.cpu_mode = False
        self.print = 100
        self.embedding_size = 512
        self.batch_size = 1
        self.workers = 4
        self.arch = 'iresnet100'
        self.load_model()

    def load_model(self):
        model = builder_inf(self)
        if not self.cpu_mode:
            model = model.cuda()
        self.model = model.eval()


def process_faces(path: str, save_path: str):

    cprint('=> modeling the network ...', 'green')
    magFace = Magface()
    cprint('=> building the dataloader ...', 'green')
    cprint('=> starting inference engine ...', 'green')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # magFace= nn.DataParallel(magFace)
    # magFace.to(device)
    # [os.path.join(path, dir) for dir in os.listdir(path) if os.path.isdir(os.path.join(path, dir))]
    dirs = [Path("img_list")]

    columns = ["path", "feature"]
    fold = Path(save_path)
    # utils.append_list_as_row(fold, columns)
    df = pd.DataFrame()

    for directory in dirs:
        inf_list = os.path.join(directory, "img.list")
        cprint('=> embedding features will be saved into {}'.format(fold))
        inf_dataset = ImgInfLoader(ann_file=inf_list)
        inf_loader = torch.utils.data.DataLoader(
            inf_dataset, batch_size=1, num_workers=1, shuffle=False)
        batch_time = utils.AverageMeter('Time', ':6.3f')
        data_time = utils.AverageMeter('Data', ':6.3f')
        progress = utils.ProgressMeter(
            len(inf_loader), [batch_time, data_time], prefix="Extract Features: ")

        # switch to evaluate mode
        with torch.no_grad():
            end = time.time()
            for i, (input, img_paths) in enumerate(tqdm(inf_loader)):
                # measure data loading time
                data_time.update(time.time() - end)
                image = input[0].cuda()
                print(image)
                # compute output
                img1_aligned = cv2.cvtColor(img1_aligned, cv2.COLOR_BGR2RGB)
                img1_aligned = PIL.Image.fromarray(img1_aligned)

                img1_aligned = composer(img1_aligned)
                img1_aligned = img1_aligned.unsqueeze(dim=0).cuda()

                embedding_feat = magFace.model(img1_aligned)
                # embedding_feat = F.normalize(embedding_feat, p=2, dim=1)
                _feat = embedding_feat.data.cpu().numpy()
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                data = {'path': img_paths, 'feature': list(_feat)}
                df = pd.DataFrame(data)
                df["feature"] = df["feature"].apply(lambda x: list(x))
                df.to_csv(fold, mode='a', header=False, index=False)


if __name__ == '__main__':
    cprint('=> parse the args ...', 'green')
    # path = "/home/chaki/Projects/gods_eye/output/ptz_good_test_2/"
    input_path = '/home/marina/Projects/new_sample_pictures'
    save_path = "db_files/faiss.csv"
    # preprocess_faces(input_path)
    process_faces(input_path, save_path)
