#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import division
import numpy as np
import math
import sys
import os
import cv2
import json
import matplotlib.pyplot as plt
import warnings

from .LaneCommon import *

EPS = 1e-6
BIG_Num = config_['image_width']

y_step = config_['image_height'] * 1.0 / config_['pts_per_lane']
fix_y_seq_ = []
y = config_['image_height'] - 1
while y >= 0:
    fix_y_seq_.append(y)
    y -= y_step

# LaneLine& l
def lane_fit(l):
    line_func = []
    x_point = []
    y_point = []
    for point in l.points_:
        x_point.append(point.x)
        y_point.append(point.y)
    x_point = np.array(x_point)
    y_point = np.array(y_point)
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        n = 1
        try:
            z = np.polyfit(y_point, x_point, 1)
            n = 1
        except np.RankWarning:
            try:
                z = np.polyfit(y_point, x_point, 2)
                n = 2
            except np.RankWarning:
                warnings.simplefilter('ignore', np.RankWarning)
                z = np.polyfit(y_point, x_point, 3)
                n = 3
        if n == 1:
            line_func.append([0, 0, z[0], z[1]])
        elif n == 2:
            line_func.append([0, z[0], z[1], z[2]])
        else:
            lines.append(z)

    return line_func

# LanePoint p, LaneLine l
def dis_between_point_and_lane(p, l):
    line_func = lane_fit(l)

    if y_pos < fix_y_seq_[-1] or y_pos > fix_y_seq_[0]:
        return BIG_Num
    else:
        cross_x = line_func[0]*y_pos**3 + line_func[1]*y_pos**2 + line_func[2]*y_pos + line_func[3]
        return math.fabs(cross_x-p.x) / config_['image_width']

# LaneLine& l1, LaneLine& l2
# l1 and l2 must matched
def dis_between_lanes(l1, l2):
    start_pos_ = min(l1.start_pos_, l2.start_pos_)  # start point is in the below of picture
    end_pos = max(l1.end_pos_, l2.end_pos_)
    l1_func = lane_fit(l1)
    l2_func = lane_fit(l2)
    dis = 0.0

    for y in fix_y_seq_:
        if y<start_pos_ and y>end_pos_:
            cross_x1 = l1_func[0]*y**3 + l1_func[1]*y**2 + l1_func[2]*y + l1_func[3]
            cross_x2 = l2_func[0]*y**3 + l2_func[1]*y**2 + l2_func[2]*y + l2_func[3]
            dis += math.fabs(cross_x1-cross_x2)/config_['image_width']
        elif (y<l1.start_pos_ and y>l2.start_pos_) or (y>l2.start_pos_ and y<l1.start_pos_):
            dis += BIG_Num
        elif (y<l1.end_pos_ and y>l2.end_pos_) or (y>l2.end_pos_ and y<l2.end_pos_):
            dis += BIG_Num

    return dis

# LaneLine& l1, LaneLine& l2
def calc_iou_between_lanes(l1, l2):
    pic1 = pic2 = np.zeros([config_['image_width'], config_['image_height']])
    l1_func = lane_fit(l1)
    l2_func = lane_fit(l2)

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
    iou = intersection/union
    return iou


# assume the width of line is 30 pixel
# LaneLine& l1, LaneLine& l2
# l1-predict line
# l2-gt line
def check_match_between_predictLane_and_gtLane(l1, l2):
    if calc_iou_between_lanes(l1,l2) < config_['match_thresh']: # TODO
        return True
    else: return False



def calc_x_cross(p1, p2, x):  # const LanePoint &p1, const LanePoint &p2, float x
    if math.fabs(p1.x - p2.x)<EPS: return -1
    k = (p1.y - p2.y)/(p1.x - p2.x)
    b = p1.y - k * p1.x
    return k*x+b

def calc_y_cross(p1, p2, y): # const LanePoint &p1, const LanePoint&p2, float y
    if math.fabs(p1.y - p2.y)<EPS: return -1
    k = (p1.x-p2.x)/(p1.y-p2.y)
    b = p1.x - k*p1.y
    return k*y+b

def compare_lane_by_point_num(l1, l2):  # const LaneLine& l1, const LaneLine& L2
    return l1.PointNum() < l2.PointNum() # 降序排列

def project(lane, point_step):  # const LaneLine& lane, int point_step
    res = {} #{int:float}
    for point in lane.points_:
        x = point.x
        y = point.y
        key = math.floor(1.0 * y / point_step + 0.5)
        res[key]=x

    return res

# const LaneLine& l1, const LaneLine& l2, int point_per_lane, int point_step
def check_same_lane(l1, l2, point_per_lane, point_step):
    same_lane_thresh = 4 # TODO
    l1_proj = project(l1, point_step)
    l2_proj = project(l2, point_step)
    dis = []
    dis_total = 0.0
    for i in range(point_per_lane):
        if (i not in l1_proj) or (i not in l2_proj): continue
        t = math.fabs(l1_proj[i]-l2_proj[i])
        dis.append(t)
        dis_total += t
    if len(dis)==0: return False
    dis_total = dis_total / len(dis)
    if dis_total < same_lane_thresh: return True
    else: return False


def compare_lane_by_score(l1, l2):
    if l1.GetScore() < l2.GetScore(): return 1# 降序排列
    elif l1.GetScore() > l2.GetScore(): return -1
    else: return 0

# const float* x_array, const float* y_array, int count, int order, float* coefficients
def polyfit(x_array, y_array, count, order, coefficients):
    maxOrder = 5
    if count <= order:
        return False
    if order > maxOrder:
        return False
    if len(x_array)==0 or len(y_array)==0 or len(coefficients)==0: return False

    B = np.zeros(maxOrder+1)
    P = np.zeros(((maxOrder+1)*2)+1)
    A = np.zeros((maxOrder+1)*2*(maxOrder+1))

    x = 0.0
    y = 0.0
    powx = 0.0

    # Identify the colum vector
    for ii in range(count):
        x = x_array[ii]
        y = y_array[ii]
        powx = 1
        for jj in range(order+1):
            B[jj] = B[jj] + (y * powx)
            powx = powx * x

    # Initialize the powx array
    P[0] = count

    for ii in range(count):
        x    = x_array[ii]
        powx = x_array[ii]
        for jj in range(1, 2*(order+1)+1):
                P[jj] = P[jj] + powx
                powx = powx * x

    for ii in range(order+1):
        for jj in range(order+1):
            A[(ii * (2 * (order + 1))) + jj] = P[ii+jj]
        A[(ii*(2 * (order + 1))) + (ii + (order + 1))] = 1

    for ii in range(order+1):
        x = A[(ii * (2 * (order + 1))) + ii]
        if x != 0:
            for kk in range(2 * (order + 1)):
                A[(ii * (2 * (order + 1))) + kk] =A[(ii * (2 * (order + 1))) + kk] / x
            for jj in range(order+1):
                if jj-ii != 0:
                    y = A[(jj * (2 * (order + 1))) + ii]
                    for kk in range(2 * (order + 1)):
                        A[(jj * (2 * (order + 1))) + kk] = A[(jj * (2 * (order + 1))) + kk] -y * A[(ii * (2 * (order + 1))) + kk]
        else: return False    # Cannot work with singular matrices

    # Calculate and Identify the coefficients
    for ii in range(order+1):
        for jj in range(order+1):
            x = 0
            for kk in range(order+1):
                x = x+ (A[(ii * (2 * (order + 1))) + (kk + (order + 1))] *B[kk])
            coefficients[ii]=x

    return True


# const LaneLine& lane
def find_k(lane):
    order = 1
    point_num = len(lane.points_)
    x_seq = []
    y_seq = []
    t = 0
    for i in range(len(lane.points_)):
        x_seq.append(lane.points_[i].x)
        y_seq.append(lane.points_[i].y)

    coefficients =[]



def calc_err_dis_without_pos(l1, l2):
    start_y = min(l1.GetPoint(0).y, l2.GetPoint(0).y)
    end_y = max(l1.LastPoint().y, l2.LastPoint().y)

    pts1 = l1.GetPoints()
    pts2 = l2.GetPoints()

    s1 = 0
    for _ in range(len(pts1)):
        if(pts1[s1].y <= start_y): break
        s1+=1

    s2 = 0
    for _ in range(len(pts2)):
        if(pts2[s2].y <= start_y): break
        s2+=1

    dis = 0.0
    count = 0

    while(s1<len(pts1) and s2<len(pts2)):
        dis += math.fabs(pts1[s1].x - pts2[s2].x)
        count += 1
        s1+=1
        s2+=1
    if count > 0 : return dis/count
    else: return sys.float_info.max



# const LaneLine& l1, const LaneLine& l2
def calc_err_dis_with_pos(l1, l2):
    max_start_pos = max(l1.start_pos_, l2.start_pos_)
    min_end_pos = min(l1.end_pos_, l2.end_pos_)

    if min_end_pos <= max_start_pos or max_start_pos<0 or min_end_pos<1:
        return sys.float_info.max

    dis = 0.0
    for i in range(max_start_pos, min_end_pos):
        dis += math.fabs(l1.points_[i-l1.start_pos_].x - l2.points_[i-l2.start_pos_].x)
    dis /= (min_end_pos - max_start_pos)

    # consider Y lane, we must let their distance become larger
    dis_start = math.fabs(l1.points_[max_start_pos - l1.start_pos_].x - l2.points_[max_start_pos - l2.start_pos_].x)
    dis = max(dis, dis_start)
    dis_end = math.fabs(l1.points_[min_end_pos-1-l1.start_pos_].x - l2.points_[min_end_pos-1-l2.start_pos_].x)
    dis = max(dis, dis_start)

    return dis


# const LaneLine& l, int idx_in, float y_in
class LaneWithCrossK:
    def __init__(self, l, idx_in, y_in):
        self.lane = l
        self.idx = idx_in
        self.y = y_in
        self.cross_x = calc_y_cross(self.lane.points_[0], self.lane.points_[1], self.y)
        self.k = (self.lane.points_[1].x -self.lane.points_[0].x) / (self.lane.points_[1].y - self.lane.points_[0].y)

# const LaneWithCrossK& l1, const LaneWithCrossK& l2
def compare_LaneWithCrossK(l1, l2):
    eps = 2.0
    x1_first = l1.cross_x
    x2_first = l2.cross_x
    if(math.fabs(x1_first-x2_first)>eps):
        return x1_first > x2_first  # 升序排列
    else:
        x1_last = l1.lane.LastPoint().x
        x2_last = l2.lane.LastPoint().x
        return x1_last > x2_last

class LaneUtil:
    # LaneSet& lane_set, float thresh
    def nms_without_pos(self, lane_set, thresh):
        lanes_sorted = lane_set.GetLanes()
        if(len(lanes_sorted) == 0): return #TODO

        # sort
        lanes_sorted.sort(compare_lane_by_score) # 根据分数降序排列
        lanes_result = []

        # selected
        selected =np.zeros(len(lanes_sorted), dtype=bool)

        for n in range(len(lanes_sorted)):
            if(selected[n]): continue
            lanes_result.append(lanes_sorted[n])
            selected[n] = True
            for t in range(n+1, len(lanes_sorted)):
                if(calc_err_dis_without_pos(lanes_sorted[n], lanes_sorted[t])<thresh):
                    selected[t] = True

        lane_set.SetLanes(lanes_result)
        return lane_set

    # LaneSet& lane_set, float thresh
    def nms_with_pos(self, lane_set, thresh):
        lanes_sorted = lane_set.GetLanes()
        if(len(lanes_sorted) == 0): return #TODO

        # sort
        lanes_sorted.sort(compare_lane_by_score) # 根据分数降序排列
        lanes_result = []

        # selected
        selected =np.zeros(len(lanes_sorted), dtype=bool)

        for n in range(len(lanes_sorted)):
            if(selected[n]): continue
            lanes_result.append(lanes_sorted[n])
            selected[n] = True

            for t in range(n+1, len(lanes_sorted)):
                if(calc_err_dis_with_pos(lanes_sorted[n], lanes_sorted[t])<=thresh):
                    selected[t] = True
        lane_set.SetLanes(lanes_result)
        return lane_set


    # LaneSet& lane_set
    def ordering(self, lane_set):
        # order lane from left to right
        lanes_crossk = []
        cross_y = float(lane_set.image_height_ -1)
        for i in range(len(lane_set.lanes_)):
            lanes_crossk.append(LaneWithCrossK(lane_set.lanes_[i], i, cross_y))
        lanes_crossk.sort(compare_LaneWithCrossK)

        # find current lanes
        right_pos = len(lanes_crossk)
        for i in range(len(lanes_crossk)):
            cp_lane = lanes_crossk[i]
            if cp_lane.k >0:
                right_pos = i
                break

        # assign lane index
        lane_idx = np.zeros(len(lanes_crossk))
        idx = -1

        for i in range(0, right_pos)[::-1]:
            lane_idx[i] = idx
            idx -= 1

        idx = 1
        for i in range(right_pos, len(lane_idx)):
            lane_idx[i] = idx
            idx += 1

        lanes_final = []
        for i in range(len(lanes_crossk)):
            lanes_final.append(lanes_crossk[i])
        lane_set.SetLanes(lanes_final, lane_idx)

        return lane_set











