#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import math
import numpy as np

BigNum = 100

def calc_loss_loc( pred, #shape: (pts_per_lane,)
                   fix_y_seq,
                   lane_target,
                   loss_loc,
                   idx, h, w):
    y_end_pos = max(pred[len(fix_y_seq)], lane_target.end_pos_)
    for y in range(y_end_pos, len(fix_y_seq)):
        loss_loc[idx, h, w] += math.fabs(pred[y]-fix_y_seq[y])

def calc_loss_lane_fit(x_array, y_array, max_order=5):
    # max order = 3
    z = None
    order = 1

    while order <= max_order:
        z, res, rank, _ = np.polyfit(y, x, order, full=True)
        if rank != order:
            z = None
            order += 1
        else: break

    if z = None:
        loss_lane_fit[idx, h, w] = BigNum
        loss_iou[idx, h, w] = BigNum
    else:
        loss_lane_fit[idx, h, w] = res
        calc_loss_iou(loss_iou, )


def calc_loss_iou(loss_iou, pred, target):
    pic1 = pic2 = np.zeros([config_['image_width'], config_['image_height']])
    l1_func
    l2_func

    for h in range(math.floor(l1.start_pos_), math.floor(l1.end_pos_), -1):
        cross_x = math.floor(l1_func[0]*h**3 + l1_func[1]*h**2 + l1_func[2]*h + l1_func[3])
        for w in range(corss_x-15, cross_x+15):
            if w>=0: pic1[w,h]=1

    for h in range(math.floor(l2.start_pos_), math.floor(l2.end_pos_), -1):
        cross_x = math.floor(l2_func[0]*h**3 + l2_func[1]*h**2 + l2_func[2]*h + l2_func[3])
        for w in range(corss_x-15, cross_x+15):
            if w>=0: pic2[w,h]=1
    union =30 (math.floor(l1.start_pos_)-math.floor(l1.end_pos_)+math.floor(l2.start_pos_)-math.floor(l2.end_pos_))
    intersection = 0
    for h in range(config_['image_height']):
        for w in range(config_['image_width']):
            if(pic1[w,h]==1 and pic2[w,h]==1):
                intersection += 1
    iou = intersection / union
    return iou


class LaneDetectionLoss(nn.Module):
    """
    result : [batch_size, laneset]
    target : [batch_size, laneset, lanefunc]

    Objective Loss:
        Loss = (Lconf + l(i,j)*alpha*Lloc)/N
    """

    def __init__(self, dis_thresh=40, neg_mining=0.3, image_height=288, image_width=512, fix_y_seq, bin_h=18, bin_w=32, use_gpu=True):
        super(LaneDetectionLoss, self).__init__()
        self.dis_thresh = dis_thresh
        self.neg_mining = neg_mining
        self.image_height = image_height
        self.image_width  = image_width
        self.fix_y_seq = fix_y_seq
        self.bin_h = bin_h
        self.bin_w = bin_w
        self.use_gpu = use_gpu

    def forward(self, predictions, targets):
        """
        Lane Detection Loss
        Args:
            predictions (tuple):
                loc_data [batch_size, pts_per_lane+1, lr_bin_num, d_bin_num]
                cls_data [batch_size, 1,              lr_bin_num, d_bin_num]
                xushi_cls_data [batch_size, 1,        lr_bin_num, d_bin_num]

            targets (tensor): Ground truth boxes and
                [batch_size, laneset]

        return: loss_loc, loss_conf, loss_fit, loss_end_pos
        """
        loc_data, cls_data, xushi_cls_data = predictions
        pts_per_lane = loc_data.size(1) - 1
        assert(pts_per_lane == len(self.fix_y_seq))
        assert(self.bin_h == loc_data.size(2))
        assert(self.bin_w == loc_data.size(3))
        batch_size   = loc_data.size(0)

        loss_loc      = torch.Tensor(batch_size, bin_h, bin_w)
        loss_conf     = torch.Tensor(batch_size, bin_h, bin_w)
        loss_lane_fit = torch.Tensor(batch_size, bin_h, bin_w)
        loss_end_pos  = torch.Tensor(batch_size, bin_h, bin_w)

        for idx in range(batch_size):
            loc_d = loc_data[idx, :, :, :]    # shape=(pts_per_lane+1, bin_h, bin_w)
            cls_d = cls_data[idx, :, :, :]
            xushi_cls_d = xushi_cls_data[idx, :, :, :]
            target = targets[idx, :]     # laneset

            # find positive samples
            # return matched anchor and matched lane ID
            anchor = np.zeros(self.bin_h, self.bin_w)
            anchor -= 1
            for h in range(self.bin_h):
                for w in range(self.bin_w):
                    y_pos = int((pts_per_lane / self.bin_h) * h)
                    y_img = self.fix_y_seq[y_pos]
                    x_img = int(self.image_width / self.bin_w * (w+1))
                    delta_x = loc_d[y_pos, h, w]
                    min_dis = self.dis_thresh
                    for i in range(target.lane_idx):
                        lane = target.GetLane(i)
                        for point in lane.points_:
                            if point.y == y_img:
                                dis = math.floor(point.x-(x_img+delta_x))
                                if dis < min_dis:
                                    anchor[h,w] = target.lane_idx[i]
                                    min_dis = dis
                                break

            # nms
            # get a anchor which positive:negtive = 1:3
            # return anchor
            # how to choose negtive anchor by dis or ?


            # calculate loss
            for h in range(bin_h):
                for w in range(bin_w):
                    if anchor[h,w] >= 0: # is positive
                        loss_loc = calc_loss_loc(loc_d[:,h,w],
                                                 self.fix_y_seq,
                                                 target.GetLane(anchor[h,w]),
                                                 loss_loc,
                                                 idx, h, w)

                        loss_ce = F.cross_entropy(cls_d[idx,h,w], targets.GetLane(anchor[h,w]).is_xushi_solid)

                        # return lane_fit
                        loss_lane_fit = calc_lane_fit(loc_d[:,h,d],
                                                      self.fix_y_seq)

                        # if turn a BigNum
                        # if fit, return least-squares and iou loss
                        # if not fit, return least-squares
                        loss_fit_and_iou = calc_loss_iou(loc_d, targets.GetLane(anchor[h,w]))


                    else:
                        loss_ce =
                             F.cross_entropy(cls_data[batch_size,0,h,w], targets.GetLane(anchor[h,w]).is_xushi_solid)




        return loss_l, loss_c













