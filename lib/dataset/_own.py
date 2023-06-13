from __future__ import print_function, absolute_import
import torch.utils.data as data
import os
import numpy as np
import cv2

class _OWN(data.Dataset):

    def __init__(self, config, data_path: str, data_label_path: str, data_file_name: str, phase: str):

        if not (phase in ['train', 'val', 'test']):
            raise AssertionError(phase)
        
        self.root = config.DATASET.ROOT
        self.inp_h = config.MODEL.IMAGE_SIZE.H
        self.inp_w = config.MODEL.IMAGE_SIZE.W
        self.mean = np.array(config.DATASET.MEAN, dtype=np.float32)
        self.std = np.array(config.DATASET.STD, dtype=np.float32)

        self.labels = []
        self.phase = phase

        img_id_gt_txt = os.path.join(data_label_path, data_file_name + "_" + phase + ".txt")

        with open(img_id_gt_txt, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').split(None, 1)
                img_path = os.path.join(data_path, line[0])
                if os.path.exists(img_path) and os.stat(img_path).st_size > 0 and line[1]:
                    self.labels.append({img_path: line[1]})

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        img_path = list(self.labels[idx].keys())[0]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_h, img_w = img.shape

        img = cv2.resize(img, (0,0), fx=self.inp_w / img_w, fy=self.inp_h / img_h, interpolation=cv2.INTER_CUBIC)
        img = np.reshape(img, (self.inp_h, self.inp_w, 1))

        img = img.astype(np.float32)
        img = (img/255. - self.mean) / self.std
        img = img.transpose([2, 0, 1])

        return img, idx








