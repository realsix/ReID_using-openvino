import time
import os
import shutil
import json
import numpy as np


def second_to_hour(sec):
    """
    秒数转为时间数，如14*60*60 -》 14：00：00
    :param sec: 输入的秒数
    :return: 返回数组，即[小时、分钟、秒]
    """
    minute, second = divmod(sec, 60)
    hour, minute = divmod(minute, 60)
    return [int(hour), int(minute), int(second)]


def hour_to_second(hour, minute, second):
    """
    计算一天某个时刻对应的秒数，如14：00：00 -》（14*60 + 0*60）*60 + 0
    :param hour: 输入的小时数
    :param minute: 输入的分钟数
    :param second: 输入的秒数
    :return: 整数形式
    """
    return int((hour * 60 + minute) * 60 + second)


def get_timestamp():
    """
    获得时间戳的函数，返回时间戳的字符串，如 2022_05_02_14_26_36_
    :return: str
    """
    now_time = round(time.time() * 1000)
    timestamp_str = time.strftime('%Y_%m_%d_%H_%M_%S_', time.localtime(now_time / 1000))
    return timestamp_str


def get_video_name(video_path):
    """
    获得输入路径的各视频的列表
    :param video_path: 输入的视频所在路径
    :return: list
    """
    video_list = []
    video_type = ['avi', 'mp4']
    place_list = []
    dir_count = 0
    for root, dirs, files in os.walk(video_path):
        if dir_count == 0:
            place_list = dirs
            dir_count += 1
        if len(files) > 0:
            for f in files:
                if f[-3:] not in video_type:
                    continue
                else:
                    file_path = os.path.join(root, f)
                    video_list.append(file_path)
                    # print(file_path)

    return video_list, place_list


def get_output_path(output_path):
    """
    生成中间存储的文件夹, 创建如：
    <save_path>_|
                |_query
                      |_q
                |_gallery
                        |_g
                |_npy
                |_message
    的目录文件夹
    :param output_path:
    :return: None
    """
    now_path = output_path
    first_directory = ['query', 'gallery', 'npy', 'message']
    second_directory = ['q', 'g']
    if not os.path.exists(now_path):
        os.mkdir(now_path)

    for i, dire in enumerate(first_directory):
        now_path = os.path.join(output_path, dire)
        if not os.path.exists(now_path):
            os.mkdir(now_path)
            if i <= 1:
                os.mkdir(os.path.join(now_path, second_directory[i]))


def get_target(target_path, output_path):
    """
    用来转移需要检测的目标图片，原路径：target_path, 复制到的路径output_path
    :param target_path: 目标图片初始所在路径
    :param output_path: 复制后所到的路径
    :return:
    """
    # 判断复制后的路径是否存在
    get_output_path(output_path)
    # 查看图片是否符合格式
    picture_type = ['png', 'jpg', 'jpeg']
    picture_name = []
    # 获取图片名称
    for root, dirs, files in os.walk(target_path):
        # 如果原路径下没有文件，则报错
        if len(files) > 0:
            # 只获取一个目标人物图片
            if len(picture_name) == 1:
                break
            # 判断是否符合图片格式
            for f in files:
                if f[-3:] in picture_type:
                    picture_name.append(f)
                    break
        else:
            return False

    origin_name = os.path.join(target_path, picture_name[0])
    after_name = os.path.join(output_path, 'query/q', picture_name[0])

    if os.path.exists(os.path.join(output_path, 'query/q')):
        shutil.copy(origin_name, after_name)


def from_npy_get_txt(raw_message, args):
    start_time = time.time()
    raw_message = raw_message  # 获得生肉信息
    if args.is_only_txt:
        my_rm_npy_name = args.npy_path
        b = np.load(my_rm_npy_name, allow_pickle=True)
        raw_message = b.tolist()

    json_name = args.json_path
    with open(json_name, encoding='utf-8') as f:
        json_data = json.load(f)
        information_map = json_data['place']
        f.close()

    fps = 30  # 设置帧率
    unsorted_message = []  # 未经排序的数据信息
    message = []  # 最终排序后每一帧包含坐标的信息

    timestamp = get_timestamp()
    writing_txt_name = os.path.join(args.data_output, 'message', timestamp+'message.txt')

    # 处理生肉信息
    for i in raw_message:
        label = i[0]  # 获得当前帧人体ID
        camera_id = str(i[1])  # 获得摄像头机位号码
        camera_time = i[2][0]  # 获得当前帧位于视频中哪一位置
        place = i[3]  # 当前图片所在的地点

        appear_in_second = information_map[place][camera_id] + (camera_time / fps)
        # print('location: ', place)
        unsorted_message.append([label, appear_in_second, place])

    unsorted_message.sort(key=lambda x: (x[0], x[1]))  # 进行排序

    for j in unsorted_message:
        my_string = "label_{} appeared at {}:{}:{} in {}".format(j[0], *second_to_hour(j[1]), j[2])
        message.append(my_string + "\n")
        # print(my_string)

    with open(writing_txt_name, 'w') as txt:
        txt.writelines(message)
    print(f"[txt] {writing_txt_name} saved successfully.")
    end_time = time.time() - start_time
    print("[txt] Used time:{:2f}s.".format(end_time))


if __name__ == "__main__":
    tar_ = './target_input'
    out_ = './output_path'
    get_target(tar_, out_)
