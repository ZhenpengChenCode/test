#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import division
import numpy as np
import math

class LanePoint:
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y

class LaneLine:
    def __init__(self, score_=0, xushi_score_=-1, is_xushi_solid_=True, start_pos_=0, end_pos_=0):
        self.score_ = score_
        self.xushi_score_ = xushi_score_
        self.is_xushi_solid_ = is_xushi_solid_
        self.start_pos_ = start_pos_
        self.end_pos_ = end_pos_
        self.points_ = []
    def Scale(self, sx, sy):
        for i in self.points_:
            i.x *= sx
            i.y *= sy
    def AddPoint(self, p):
        self.points_.append(p) # LanePoint p
    def PointNum(self):
        return len(self.points_)
    def GetPoints(self):
        return self.points_
    def GetPoints(self, index):
        return self.points_[index]
    def LastPoint(self):
        if len(self.points_)>0:
            return self.points_[-1]
        else: return None
    def SetScore(self, s):
        self.score_ = s
    def SetXushiScore(self, s):
        self.xushi_score_ = s
    def GetScore(self):
        return self.score_
    def GetXushiScore(self):
        return self.xushi_score_
    def GetXushiType(self):
        return self.is_xushi_solid

class LaneSet:
    def __init__(self, image_width_=512, image_height_=288, lanes_=[], lane_idx_=[]):
        self.image_width_ = image_width_
        self.image_height_ = image_height_
        self.lanes_ = lanes_
        self.lane_idx_ = lane_idx_
    def Scale(self, sx, sy):
        self.image_width_ *= sx
        self.image_height_ *= sy
        for lane in self.lanes_:
            lane.Scale(sx, sy)
    def AddLane(self, lane):
        self.lanes_.append(lane)
        self.lane_idx_.append(0)
    def AddLane(self, lane, idx):
        self.lanes_.append(lane)
        self.lane_idx_.append(idx)
    def SetLanes(self, lanes):
        self.lanes_ = lanes
        self.lane_idx_ = np.zeros(len(lanes))
    def SetLanes(self, lanes, lane_idx):
        self.lanes_ = lanes
        self.lane_idx_ = lane_idx
    def GetLanes(self):
        return self.lanes_
    def GetLane(self, i):
        assert(i>=0 and i < len(self.lanes_))
        return self.lanes_[i]
    def GetLaneIdx(self, i):
        assert(i>=0 and i < len(self.lanes_))
        return self.lane_idx_[i]
    def LaneNum(self):
        return len(self.lanes_)
    def ImageHeight(self):
        return self.image_height_
    def ImageWidth(self):
        return self.image_width_

