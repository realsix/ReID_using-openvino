import time


def main():
    # Initialize model
    m = None
    dictr = {'ID_0': {'time_count': 2,
                      'camera_1': 0, 'start_time_1': 34, 'end_time_1': 35, 'continuing_time_1': 1,
                      'camera_2': 1, 'start_time_2': 25, 'end_time_2': 36,  'continuing_time_2': 11},
             'ID_1': {'time_count': 2,
                      'camera_1': 3, 'start_time_1': 28, 'end_time_1': 35, 'continuing_time_1': 7,
                      'camera_2': 1, 'start_time_2': 25, 'end_time_2': 36,  'continuing_time_2': 11},
             }
    my_memory = {}
    ID_memory_dict = {0: [-1]}
    # {0:[-1],
    #  1:[1, 2, 3, 4],
    #  2:[1, 2, 4, 5]}  3结束，5新增，1，2持续
    id_index = 1
    # 进入循环
    while True:
        m.detect()  # 开始检测
        if m.ID != None:  # 如果检测到的ID存在
            if len(m.ID) > 0:  # 如果存在多个ID
                ID_memory_dict[str(id_index)] = m.ID  # 保存当前ID
                for mm_ID, mm_camera in m.ID:  # 对每个ID进行处理
                    if id_index >= 1:
                        if mm_ID in ID_memory_dict[str(id_index-1)]:  # 如果当前ID在上一轮获取的ID中
                            if str(mm_ID) in my_memory:  # 如果这个ID已经记录过的话，跳过，即处于持续时间
                                continue
                        else:  # 如果不在上一轮获取的ID中，1是新的人，2是人不在这里
                            if str(mm_ID) not in my_memory:  # 1 是新的人，即新的ID，处于开始时间内，开始记录
                                my_memory[str(mm_ID)] = {}  # ID单独记录
                                # 记录time_count, 即出现的次数
                                my_memory[str(mm_ID)]['time_count'] = my_memory[str(mm_ID)].setdefault('time_count',
                                                                                                       0) + 1
                                start_time_name = 'start_time_' + my_memory[str(mm_ID)]['time_count']  # start_time_0
                                camera_name = 'camera_' + my_memory[str(mm_ID)]['time_count']  # camera_1 第一次出现的视频
                                # 对应的初始时间值
                                my_memory[str(mm_ID)][start_time_name] = my_memory[str(mm_ID)].setdefault(
                                    start_time_name, time.time())
                                # 对应的视频
                                my_memory[str(mm_ID)][camera_name] = my_memory[str(mm_ID)].setdefault(camera_name,
                                                                                                      str(mm_camera))
                            elif str(mm_ID) in my_memory:  # 2 是已经存在了的，则上一次持续时间结束
                                start_time_name = 'start_time_' + my_memory[str(mm_ID)]['time_count']  # start_time_0
                                end_time_name = 'end_time_' + my_memory[str(mm_ID)]['time_count']  # end_time_0
                                # 2.1 如果字典中不存在上一次time_count的结束时间，进行结束运算
                                if end_time_name not in my_memory[str(mm_ID)]:
                                    # continuing_time_0
                                    continuing_time_name = 'continuing_time_' + my_memory[str(mm_ID)]['time_count']
                                    # end_time_name: time.time()
                                    my_memory[str(mm_ID)][end_time_name] = my_memory[str(mm_ID)].setdefault(end_time_name,
                                                                                                            time.time())
                                    # continuing_time: end_time - start_time
                                    continuing_time = my_memory[str(mm_ID)][end_time_name] - my_memory[str(mm_ID)][start_time_name]
                                    my_memory[str(mm_ID)][continuing_time_name] = my_memory[str(mm_ID)].setdefault(continuing_time_name,
                                                                                                                   continuing_time)
                                else:  # 2.2 如果存在，则开始新的记录运算
                                    # 记录time_count, 即出现的次数
                                    my_memory[str(mm_ID)]['time_count'] = my_memory[str(mm_ID)].setdefault('time_count', 0) + 1
                                    # start_time_0
                                    start_time_name = 'start_time_' + my_memory[str(mm_ID)]['time_count']
                                    # camera_1 第一次出现的视频
                                    camera_name = 'camera_' + my_memory[str(mm_ID)]['time_count']
                                    # 对应的初始时间值
                                    my_memory[str(mm_ID)][start_time_name] = my_memory[str(mm_ID)].setdefault(
                                        start_time_name, time.time())
                                    # 对应的视频
                                    my_memory[str(mm_ID)][camera_name] = my_memory[str(mm_ID)].setdefault(camera_name,
                                                                                                          str(mm_camera))
        id_index += 1


if __name__ == "__main__":
    main()
