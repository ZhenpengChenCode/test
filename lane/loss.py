#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import math
import json
from LaneCommon import *
from LaneGridDetector import *
from LaneUtil import *

# loss=precitsion+recall+location_loss

# LaneSet& laneset1, LaneSet& laneset2
def calc_loss(laneset1, laneset2):
    loss = 0
    loss_loc = 0
    loss_precision = 0
    loss_call = 0

    for predict_lane in laneset1:
        for gt_lane in laneset2:
            if check_match_between_lanes(predict_lane, gt_lane):
                loss_loc += dis_between_lanes(predict_lane, gt_lane)











