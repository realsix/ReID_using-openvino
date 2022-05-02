#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import time

import cv2
import numpy as np
# from ReID_demo import run_demo

start_time = time.time()
video = "my_data/video/1/9.mp4"
result_video = "my_data/video/1/result_9.mp4"

my_video = ["video/"]

cap = cv2.VideoCapture(video)

fps_video = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
videoWriter = cv2.VideoWriter(result_video, fourcc, fps_video, (frame_width, frame_height))

my_npy_name = './my_data/running_result/1/npy/location.npy'
my_rm_npy_name = './my_data/running_result/1/npy/rm.npy'

a = np.load(my_npy_name, allow_pickle=True)
graphTable = a.tolist()

b = np.load(my_rm_npy_name, allow_pickle=True)
raw_message = b.tolist()
# print(raw_message)
print(len(raw_message))

raw_message = raw_message[:150]
# print(graphTable)
# raw_message = run_demo()


# p[2] : [frame_num,person_num]
# if red, both should be matched
# labels = [p[0] for p in raw_message]
frame_appear = [p[2][0] for p in raw_message]  # 检索的符合条件的所有帧数
person_num_appear = [p[2][1] for p in raw_message]  # 检索的符合条件的所有帧数里面的人，不能打乱，否则出错

count = 0  # count: 视频的第几帧

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        count += 1
        for detection in graphTable:
            frame_num = detection[1]

            # 如果帧数不匹配，就跳过
            if frame_num != count:
                continue

            x1, y1, x2, y2 = detection[0]

            i = detection[2]  # 同一帧里面的第几个人

            # 两个约束条件：1.这一帧在检索的top-k里面 2. 可能这一帧里面有很多个人，这个人要匹配
            # frame_appear.index(frame_num) 在【检索的符合条件的所有帧数】这个列表里面寻找这一帧的索引，利用这个索引找出这一帧里面的这一个人
            isred = True if frame_num in frame_appear and i == person_num_appear[
                frame_appear.index(frame_num)] else False

            if isred:
                color = (0, 0, 255)  # red
            else:
                color = (0, 255, 0)  # green

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 5)
            videoWriter.write(frame)
    else:
        videoWriter.release()
        break
end_time = time.time() - start_time
print('[visualize] Finished! Used time:{0:2f}s.'.format(end_time))
