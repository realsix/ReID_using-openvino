import argparse
import queue
from threading import Thread
import random


import cv2 as cv

from utils.network_wrappers import Detector
from utils.misc import read_py_config, check_pressed_keys
from utils.video import MulticamCapture, NormalizerCLAHE
from openvino.inference_engine import IECore  # pylint: disable=import-error,E0611

from tracker import Tracker
import monitors

from path_use import *


# 初始化推理器
print('[ReID_demo] Creating Inference Engine')
ie = IECore()


class FramesThreadBody:
    """
    multi_camera_multi_target_tracking_demo.py中copy来的
    """
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


def from_video_get_person(params, config, capture, detector, place_list, video_list):
    """
    魔改 multi_camera_multi_target_tracking_demo.py中 的def run()

    :param params: 输入的args
    :param config:
    :param capture:
    :param detector:
    :param place_list:
    :param video_list:
    :return:
    """
    print("[ReID_demo][INFO] Video Writing……")
    # initialize parameters
    frame_number = 0  # 初始第几帧
    key = -1  # 对摄像头模式，按ESC键退出程序
    skip = params.skip  # 用于跳帧
    locate = []

    out_path = os.path.join(params.data_output, 'gallery/g/')
    print(f"[ReID_demo][INFO] output_path:{out_path}")

    # config.normalizer_config.enabled: usually False
    # 通常是False， 不知道什么用
    if config.normalizer_config.enabled:
        capture.add_transform(
            NormalizerCLAHE(
                config.normalizer_config.clip_limit,
                config.normalizer_config.tile_size,
            )
        )

    # 图像线程主体？queue？
    thread_body = FramesThreadBody(capture, max_queue_length=len(capture.captures) * 2)
    frames_thread = Thread(target=thread_body)
    frames_thread.start()

    # 以下几行不懂
    prev_frames = thread_body.frames_queue.get()
    detector.run_async(prev_frames, frame_number)
    presenter = monitors.Presenter(params.utilization_monitors, 0)

    # thread_body.process: 是否存在图像
    while thread_body.process:
        key = check_pressed_keys(key)  # 按到key对应的按键退出程序
        if key == 27:
            break
        presenter.handleKey(key)

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
        # print(f"[ReID_demo][INFO] thread body count: {thread_body.count}, all_detections: {all_detections}")

        frame_number += skip
        detector.run_async(frames, frame_number)

        zip_detections = list(zip(all_detections, frames))

        for video_num, video_message in enumerate(zip_detections):
            all_d = video_message[0]
            picture = video_message[1]

            for i, detections in enumerate(all_d):
                possible = detections[1]
                x1, y1, x2, y2 = detections[0]
                if possible > params.t_detector:
                    cut = picture[y1:y2, x1:x2]

                    name = place_list[video_num] + '_'
                    name += os.path.basename(video_list[video_num]).split('.')[0]
                    name += '_' + str(frame_number) + '_' + str(i)
                    name = os.path.join(out_path, name + '.jpg')

                    # print(f'[INFO] jpg_name: {name}')
                    cv.imwrite(name, cut)
                    locate.append([detections[0], frame_number, i])
    if params.is_save:
        locate_array = np.array(locate, dtype=object)

        timestamp = get_timestamp()
        my_npy_name = os.path.join(params.data_output, 'npy', timestamp + 'location.npy')

        # my_npy_name = self.params.output_npy_path + 'location.npy'
        # name_count = 0
        # while os.path.isfile(my_npy_name):
        #     my_npy_name = params.output_npy_path + '/location_' + str(name_count) + '.npy'
        #     name_count += 1

        np.save(my_npy_name, locate_array)
        print(f"[ReID_demo][INFO] {my_npy_name} saves successfully.")
        return my_npy_name
    print("[ReID_demo][INFO] Video Writing Finished!")
    return False


def run_demo(args):
    if len(args.config):
        print('[ReID_demo] Reading configuration file {}'.format(args.config))
        config = read_py_config(args.config)
    else:
        raise FileNotFoundError

    random.seed(config.random_seed)
    if args.is_only_txt:
        from_npy_get_txt('', args=args)
        return True

    # 从视频路径中获取视频的列表
    input_ = args.video_input[0]
    input_video, place_list = get_video_name(input_)

    # print(input_video)
    # print(input_video[0].split('\\')[-2])

    # 创建输出文件夹，并将目标图片复制到query中
    output_video_path = args.data_output
    get_target(target_path=args.target_input[0], output_path=output_video_path)

    capture = MulticamCapture(input_video, args.loop)
    object_detector = Detector(ie, args.m_detector,
                               config.obj_det.trg_classes,
                               args.t_detector,
                               args.device, args.cpu_extension,
                               capture.get_num_sources())

    from_video_get_person(args, config, capture, object_detector, place_list=place_list, video_list=input_video)
    # 是否开始追踪，即是否开启 Track
    if args.is_tracker:
        tracker = Tracker(ie, args, data_dir=output_video_path, device='CPU')
        fm = tracker.get_feature_map()
        raw_message = tracker.compute_distance(fm)
        # 开启Track后，是否保存数据为txt
        if args.is_txt:
            from_npy_get_txt(raw_message=raw_message, args=args)
        return raw_message
    else:
        return False


def my_args():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='Moving tracking in the period of pandemic '
                                                 'with ReID using openvino')
    # 输入
    parser.add_argument('-v', '--video_input', required=True, nargs='+',
                        help='请输入视频所在路径，如: ./video_input')
    parser.add_argument('-t', '--target_input', required=True, nargs='+',
                        help='请输入目标图片所在路径，如: ./target_input/target_person.jpg')
    parser.add_argument('--npy_path', type=str, default='./data_output/2',
                        help='生成的npy文件的绝对路径，如：F:/python/rm.npy')
    parser.add_argument('--json_path', type=str, default='./configs/configs.json',
                        help='存储输入视频信息的所在路径，如：./configs/configs.json')

    # 输出
    parser.add_argument('--data_output', type=str, default='./data_output/2',
                        help='生成的npy文件以及中间检测用的数据')

    # 模型：model/...
    parser.add_argument('--m_reid_xml', type=str, default='./model/hr18.xml',
                        help='xml所在位置')
    parser.add_argument('--m_reid_bin', type=str, default='./model/hr18.bin',
                        help='bin所在文件位置')
    parser.add_argument('--loop', default=False, action='store_true',
                        help='Optional. Enable reading the input in a loop')
    parser.add_argument('-m', '--m_detector', type=str, default='./model/detection.xml',
                        help='Path to the object detection model_xml')
    # 默认设置
    parser.add_argument('--config', type=str, default=os.path.join(current_dir, 'configs/person.py'), required=False,
                        help='Configuration file')

    # 其余选项
    parser.add_argument('--t_detector', type=float, default=0.95,
                        help='Threshold for the object detection model')
    parser.add_argument('-l', '--cpu_extension',
                        help='MKLDNN (CPU)-targeted custom layers.Absolute path to '
                             'a shared library with the kernels impl.',
                        type=str, default=None)
    parser.add_argument('-d', '--device', type=str, default='CPU',
                        help='CPU、MYRIAD')
    parser.add_argument('-u', '--utilization_monitors', default='', type=str,
                        help='Optional. List of monitors to show initially.')

    parser.add_argument('--top_k', default=-1, type=int,
                        help='Top k indexes in gallery feature.')
    parser.add_argument('--is_save', default=True, type=bool,
                        help='Save all detections.')
    parser.add_argument('--is_tracker', default=True, type=bool,
                        help='default is False, for collecting gallery')
    parser.add_argument('--is_txt', default=True, type=bool,
                        help='default is True, for getting information txt')
    parser.add_argument('--is_only_txt', default=False, type=bool,
                        help='Get information map txt via .npy file.')
    parser.add_argument('--skip', default=10, type=int,
                        help='Skip n frames when analyzing.')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    start_time = time.time()
    my_arg = my_args()
    my_message = run_demo(my_arg)

    print("[ReID_demo] Using time: {:02f} s".format(time.time() - start_time))
