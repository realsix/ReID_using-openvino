import argparse
import time
import queue
from threading import Thread

import os
import random

import numpy as np
import cv2 as cv

from utils.network_wrappers import Detector
from utils.misc import read_py_config, check_pressed_keys
from utils.video import MulticamCapture, NormalizerCLAHE
from openvino.inference_engine import IECore  # pylint: disable=import-error,E0611

from tracker import Tracker
import monitors

print('Creating Inference Engine')
ie = IECore()


class FramesThreadBody:
    def __init__(self, capture, max_queue_length=2):
        self.process = True
        self.frames_queue = queue.Queue()
        self.capture = capture
        self.max_queue_length = max_queue_length

    def __call__(self):
        self.count = 0
        while self.process:
            self.count += 1
            if self.count > 1000000:
                self.count = 0
            if self.frames_queue.qsize() > self.max_queue_length:
                time.sleep(0.1)
                continue
            has_frames, frames = self.capture.get_frames()
            if not has_frames and self.frames_queue.empty():
                self.process = False
                break
            if has_frames:
                self.frames_queue.put(frames)


def from_video_get_person(params, config, capture, detector):
    frame_number = 0
    key = -1
    locate = []
    if config.normalizer_config.enabled:
        capture.add_transform(
            NormalizerCLAHE(
                config.normalizer_config.clip_limit,
                config.normalizer_config.tile_size,
            )
        )

    thread_body = FramesThreadBody(capture, max_queue_length=len(capture.captures) * 2)
    frames_thread = Thread(target=thread_body)
    frames_thread.start()

    frames_read = False
    set_output_params = False

    prev_frames = thread_body.frames_queue.get()
    detector.run_async(prev_frames, frame_number)
    presenter = monitors.Presenter(params.utilization_monitors, 0)

    while thread_body.process:
        key = check_pressed_keys(key)
        if key == 27:
            break
        presenter.handleKey(key)

        skip = params.skip  # 跳帧分析

        try:
            if thread_body.count % skip == 0:
                frames = thread_body.frames_queue.get_nowait()
            else:
                frames = thread_body.frames_queue.get_nowait()
                continue
        except queue.Empty:
            frames = None

        if frames is None:
            continue

        all_detections = detector.wait_and_grab()
        frame_number += skip
        detector.run_async(frames, frame_number)

        for video in range(len(all_detections)):
            for i, detections in enumerate(all_detections[video]):
                possible = detections[1]
                x1, y1, x2, y2 = detections[0]
                if possible > params.t_detector:
                    cut = frames[0][y1:y2, x1:x2]
                    out_path = params.output_pic
                    name = os.path.basename(params.input[video]).split('.')[0] + '_' + str(frame_number) + '_' + str(i)
                    cv.imwrite(out_path + name + '.jpg', cut)
                    locate.append([detections[0], frame_number, i, 0])
    if params.is_save:
        locate_arr = np.array(locate)
        np.save('location.npy', locate_arr)
        print("Finished saving.")

    print('Finished writing.')


def run_demo():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='Moving tracking in the period of pandemic \
                                                     with ReID using openvino')
    parser.add_argument('-i', '--input', required=True, nargs='+',
                        help='Input sources (indexes of cameras or paths to video files)')
    parser.add_argument('--m_reid_xml', type=str, default='model/hr18.xml',
                        help='Path to the object re-identification model_xml')
    parser.add_argument('--m_reid_bin', type=str, default='model/hr18.bin',
                        help='Path to the object re-identification model_xml')
    parser.add_argument('--loop', default=False, action='store_true',
                        help='Optional. Enable reading the input in a loop')
    parser.add_argument('-m', '--m_detector', type=str, default='model/detection.xml',
                        help='Path to the object detection model_xml')
    parser.add_argument('--config', type=str, default=os.path.join(current_dir, 'configs/person.py'), required=False,
                        help='Configuration file')
    parser.add_argument('--output_pic', type=str, default='./ReID_demo/gallery/g/', help='Path to output picture')
    parser.add_argument('--t_detector', type=float, default=0.9,
                        help='Threshold for the object detection model')
    parser.add_argument('--output_video', type=str, default='', required=False,
                        help='Optional. Path to output video')
    parser.add_argument('-l', '--cpu_extension',
                        help='MKLDNN (CPU)-targeted custom layers.Absolute \
                                  path to a shared library with the kernels impl.',
                        type=str, default=None)
    parser.add_argument('-d', '--device', type=str, default='CPU')
    parser.add_argument('-u', '--utilization_monitors', default='', type=str,
                        help='Optional. List of monitors to show initially.')
    parser.add_argument('--top_k', default=10, type=int,
                        help='Top k indexes in gallery feature.')
    parser.add_argument('--is_save', default=False, type=bool,
                        help='Save all detections')
    parser.add_argument('--skip', default=5, type=int,
                        help='Skip n frames when analyzing.')
    args = parser.parse_args()

    if len(args.config):
        # print('Reading configuration file {}'.format(args.config))
        config = read_py_config(args.config)
    else:
        raise FileNotFoundError

    random.seed(config.random_seed)

    capture = MulticamCapture(args.input, args.loop)

    object_detector = Detector(ie, args.m_detector,
                               config.obj_det.trg_classes,
                               args.t_detector,
                               args.device, args.cpu_extension,
                               capture.get_num_sources())

    from_video_get_person(args, config, capture, object_detector)

    tracker = Tracker(ie, args, data_dir='ReID_demo', device='CPU')
    fm = tracker.get_feature_map()
    raw_message = tracker.compute_distance(fm)

    # print(raw_message)

    return raw_message


if __name__ == '__main__':
    message = run_demo()
    print(message)
