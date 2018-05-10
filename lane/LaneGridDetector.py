#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import math
import json

from .LaneCommon import *
from .LaneUtil import *



class LaneGridConfig:
    def __init__(self, config_):
        self.pts_per_lane = config_['pts_per_lane']
        self.d_bin_num = config_['d_bin_num']
        self.lr_bin_num = config_['lr_bin_num']
        self.cls_thresh = config_['cls_thresh']
        self.nms_thresh = config_['nms_thresh']
        self.xushi_cls_thresh = config_['xushi_cls_thresh']


class LaneGridDetector:
    def __init__(self, mode = 'train', input_width=512, input_height=288, input_channel=3):
        self.mode_ = mode
        self.input_width_ = input_width
        self.input_height_ = input_height
        self.input_channel_ = input_channel
        self.config_ = LaneGridConfig(config_)
        if self.config_.xushi_cls_thresh >= 0:
            self.predict_xushi_ = True
        else:  self.predict_xushi_ = False

        y_step = self.input_height_ * 1.0 /self.config_.pts_per_lane
        self.fix_y_seq_ = []
        y = self.input_height_-1
        while(y>=0):
            self.fix_y_seq_.append(y)
            y -= y_step

    # preprocess
    # const unsigned char* image_data, int image_width, int image_height
    # int image_channel, DataBlob& input_blob, float& scale_x, float& scale_y
    def PrepareInputData(self, image_data, image_width, image_height, image_channel, input_blob, scale_x, scale_y):

        # TODO
        if (image_data==NULL or input_blob.data):
            return False
        allocated_resize = False
        # resize_data = NULL

        scale_x = self.input_width_ * 1.0 / image_width
        scale_y = self.input_height_ * 1.0 / image_height

    # const DataBlob& up_blob, const DataBlob& down_blob
    # const DataBlob& cls_blob, const DataBlob& xushi_cls_blob

    def ProcessOutputBlob(self, up_blob, down_blob, cls_blob, xushi_cls_blob):
        # up_blob   [1,73,18,32]
        # down_blob [1,72,18,32]
        # cls_blob  [1, 2,18,32]
        # xushi_cls_blob [1,2,18,32]
        assert(cls_blob[0] == 1)
        assert(cls_blob[1] == 2)
        assert(cls_blob[2] == self.config_.lr_bin_num)
        assert(cls_blob[3] == self.config_.d_bin_num)
        assert(up_blob[1] == self.config_.pts_per_lane+1)
        assert(down_blob[1] == self.config_.pts_per_lane)

        bin_h = self.config_.lr_bin_num
        bin_w = self.config_.d_bin_num

        up_data = up_blob.flatten()
        down_data = down_blob.flatten()
        cls_data = cls_blob.flatten()[bin_h*bin_w:]
        xushi_cls_data = xushi_cls_blob.flatten()

        if self.predict_xushi_:
            xushi_cls_data = xushi_cls_data[bin_h*bin_w:]

        step_w = self.input_width_ * 1.0 / bin_w
        step_h = self.input_height_ * 1.0 / bin_h
        lane_set = LaneSet(image_width_=self.input_width_, image_height_=self.input_height_)

        cls_pointer = 0
        xushi_cls_pointer = 0
        up_pointer = 0
        down_pointer = 0

        for h in range(bin_h):
            for w in range(bin_w):
                c_x = (1.0 * w + 0.5) * step_w
                c_y = (1.0 * h + 0.5) * step_h

                prob = cls_data[cls_pointer]
                cls_pointer += 1

                xushi_prob = -1
                if self.predict_xushi_:
                    xushi_prob = xushi_cls_data[xushi_cls_pointer]
                    xushi_cls_pointer += 1

                if prob < self.config_.cls_thresh: continue

                up_lane =  LaneLine()
                down_lane = LaneLine()

                up_offset = int(h * bin_w + w)
                down_offset = int(h * bin_w + w)
                y_pos = int((bin_h -1 - h) * (self.config_.pts_per_lane / bin_h))
                relative_end_pos = up_data[up_offset]
                end_pos = y_pos
                start_pos = y_pos
                for i in range(self.config_.pts_per_lane):
                    if (i>=relative_end_pos or y_pos + i >= len(self.fix_y_seq_)): break
                    x_offset = int(up_offset + (1 + i) * bin_h * bin_w)
                    rela_x = float(up_data[x_offset])
                    p = LanePoint(c_x + rela_x, self.fix_y_seq_[y_pos+i])
                    end_pos = y_pos + i +1
                    up_lane.AddPoint(p)
                for i in range(y_pos):
                    x_offset = int(down_offset + i * bin_h * bin_w)
                    rela_x = float(down_data[x_offset])
                    p = LanePoint(c_x + rela_x, self.fix_y_seq_[y_pos-1-i])
                    start_pos = y_pos - 1 - i
                    down_lane.AddPoint(p)
                    if (p.x<0 or p.x>self.input_width_ or p.y<0 or p.y>self.input_heght_): break
                if(up_lane.PointNum() + down_lane.PointNum() >=2):
                    lane = LaneLine()
                    if(down_lane.PointNum()>0):
                        for it in range(len(down_lane.points_))[::-1]:
                            lane.AddPoint(down_lane.points_[it])
                    if(up_lane.PointNum()>0):
                        for it in range(len(up_lane.points_)):
                            lane.AddPoint(up_lane.points_[it])

                    lane.score_ = prob
                    lane.start_pos_ = start_pos
                    lane.end_pos_ = end_pos
                    lane.xushi_score_ = xushi_prob
                    lane.is_xushi_solid_ = True
                    if(self.predict_xushi and self.xushi_porb < self.config_.xushi_cls_thresh):
                        lane.is_xushi_solid_ = False

                    lane_set.AddLane(lane)

        lane_set = LaneUtil.nms_with_pos(lane_set, self.config_.nms_thresh)

        lane_set = LaneUtil.ordering(lane_set)

        return lane_set

    # const unsigned char* image_data, int image_width, int image_height, int
    # iamge_channel, LaneSet& lane_set
    def Predict(image_data, image_width, image_height, image_channel, lane_set):

        scale_x = 0
        scale_y = 0

        if self.predict_xushi_:
            xushi_cls_blob=None #TODO
            lane_set = LaneGridDetector.ProcessOutputBlob(up_blob, down_blob, cls_blob, xushi_cls_blob)
        else:
            xushi_cls_blob=None  # TODO
            lane_set = LaneGridDetector.ProcessOutputBlob(up_blob, down_blob, cls_blob, xushi_cls_blob)

        lane_set.Scale(1.0/scale_x, 1.0/scale_y)
        return True
