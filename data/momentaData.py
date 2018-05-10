#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import warnings
import json

sys.path.append("../")
from lane import *


def get_target(target_json_path):
    """
    tuple: [laneFunction], [laneSet]
    max order is 3
    """
    with open(target_json_path, 'r') as jsonfile:
        str_json = json.load(jsonfile)

    lines = []
    lines_x = []
    lines_y = []

    for i in str_json['Lines']:
        x_point = []
        y_point = []
        for j in i['cpoints']:
            x_point.append(int(j['x']))
            y_point.append(int(j['y']))
        x = np.array(x_point)
        y = np.array(y_point)
        lines_x.append(x)
        lines_y.append(y)
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            n = 1
            try:
                z = np.polyfit(y, x, 1)
                n = 1
            except np.RankWarning:
                try:
                    z = np.polyfit(y, x, 2)
                    n = 2
                except np.RankWarning:
                    warnings.simplefilter('ignore', np.RankWarning)
                    z = np.polyfit(y,x,3)
                    n = 3
            if n==1:
                lines.append([0, 0, z[0], z[1]])
            elif n==2:
                lines.append([0, z[0], z[1], z[2]])
            else:
                lines.append(z)
    return [lines, lines_x, lines_y]


class momentaDataset(data.Dataset):
    def __init__(self, root, fix_y_seq, datasetName='momenta'):
        self.root = root # MOMENTA_ROOT
        self.fix_y_seq = fix_y_seq
        self.targetPath = osp.join(self.root, 'results_lane')
        self.imgPath = osp.join(self.root, 'images')
        self.name = datasetName
        self.imgs = []
        self.targets =[]

        for root, dirs, files in os.walk(self.imgPath):
            for filename in files:
                target_path = osp.join(self.targetPath, filename[:-3]+'json')
                if os.path.exists(target_path):
                    with open(target_path, 'r') as jsonfile:
                        str_json = json.load(jsonfile)
                        if str_json['Skip'] == True:
                            self.img.append(osp.join(self.imgPath, filename))
                            self.targets.append(target_path)

    def __getitem__(self, index):

        img_path = self.img[index]
        target_path = self.target[index]

        img = cv2.imread(img_path)
        height, width, channels = img.shape
        target = get_target(target_path)

        return img, target

    def __len__(self):

        return len(self.img)

    def __get_target(self, target_path):
        """
        tuple: [laneFunction], [laneSet]
        max order is 3
        """
        with open(target_path, 'r') as jsonfile:
            str_json = json.load(jsonfile)

        lanesFunc = []
        laneset = LaneSet()

        for l in str_json['Lines']:
            lane = LaneLine()
            x = []
            y = []
            if l['skip']==True: continue
            # TODO
            #lane.is_sushi_solid_ = (l['dashed']==solid) # TODO
            for p in i['cpoints']:
                x.append(int(point['x']))
                y.append(int(point['y']))
            x = np.array(x)
            y = np.array(y)
            y_end = min(y)
            lane.end_pos_ = y_end
            # lane fit (mat order=3)
            n = 1
            z = None
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    z = np.polyfit(y, x, 1)
                    n = 1
                except np.RankWarning:
                    try:
                        z = np.polyfit(y, x, 2)
                        n = 2
                    except np.RankWarning:
                        warnings.simplefilter('ignore', np.RankWarning)
                        z = np.polyfit(y,x,3)
                        n = 3

            if n==1:   z = [0,    0, z[0], z[1]]
            elif n==2: z = [0, z[0], z[1], z[2]]
            lanesFunc.append(z)

            for y_pos in self.fix_y_seq[::-1]:
                if y_pos>y_end:
                    x_pos = z[0]*y_pos**3 + z[1]*y_pos**2 + z[2]*y_pos + z[3]
                    if x_pos >=0  and x_pos< self.image_width-1:
                        lane.AddPoint(LanePoint(x_pos, y_pos))
                        lane.start_pos_ = y_pos

            laneset.append(lane)
        return lanesFunc, laneset











